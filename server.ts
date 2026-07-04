import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { randomUUID } from 'crypto';
import { GoogleGenAI, Type } from '@google/genai';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { createRequire } from 'module';
import * as fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Dynamic check to find .env in root folder, even if running from dist/
let envPath = path.resolve(__dirname, '.env');
if (!fs.existsSync(envPath)) {
  envPath = path.resolve(__dirname, '..', '.env');
}
dotenv.config({ path: envPath });

const require = createRequire(import.meta.url);
const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');

function cleanEnvVar(val: string | undefined): string | undefined {
  if (!val) return undefined;
  let s = val.trim();
  if (s.startsWith('"') && s.endsWith('"')) {
    s = s.substring(1, s.length - 1);
  }
  if (s.startsWith("'") && s.endsWith("'")) {
    s = s.substring(1, s.length - 1);
  }
  return s;
}

function initFirebase() {
  if (getApps().length > 0) return;

  const serviceAccountPath = cleanEnvVar(process.env.FIREBASE_SERVICE_ACCOUNT_PATH);
  if (serviceAccountPath) {
    try {
      const raw = fs.readFileSync(serviceAccountPath, 'utf8');
      const serviceAccount = JSON.parse(raw);
      initializeApp({
        credential: cert(serviceAccount),
      });
      console.log('✅ Firebase initialized via service account file');
      return;
    } catch (e: any) {
      console.error('❌ Failed to read service account file:', e.message);
      process.exit(1);
    }
  }

  // Option 2: Individual environment variables (easier for deployment)
  const projectId   = cleanEnvVar(process.env.FIREBASE_PROJECT_ID);
  const clientEmail = cleanEnvVar(process.env.FIREBASE_CLIENT_EMAIL);
  let privateKey    = cleanEnvVar(process.env.FIREBASE_PRIVATE_KEY);
  if (privateKey) {
    privateKey = privateKey.replace(/\\n/g, '\n');
  }

  console.log('🔍 Firebase Diagnostic Info:', {
    FIREBASE_SERVICE_ACCOUNT_PATH: serviceAccountPath ? 'exists' : 'missing/empty',
    FIREBASE_PROJECT_ID: projectId ? 'exists' : 'missing/empty',
    FIREBASE_CLIENT_EMAIL: clientEmail ? 'exists' : 'missing/empty',
    FIREBASE_PRIVATE_KEY: privateKey ? `exists (length: ${privateKey.length})` : 'missing/empty'
  });

  if (projectId && clientEmail && privateKey) {
    try {
      initializeApp({
        credential: cert({ projectId, clientEmail, privateKey } as any),
      });
      console.log('✅ Firebase initialized via environment variables');
      return;
    } catch (e: any) {
      console.error('❌ Failed to initialize Firebase via env variables:', e.message);
      process.exit(1);
    }
  }

  console.error('❌ Firebase not configured! Set FIREBASE_SERVICE_ACCOUNT_PATH or (FIREBASE_PROJECT_ID + FIREBASE_CLIENT_EMAIL + FIREBASE_PRIVATE_KEY)');
  process.exit(1);
}

initFirebase();
const db = getFirestore();

// Firestore collection references
const employeesCol = db.collection('employees');
const leadsCol     = db.collection('leads');
const configCol    = db.collection('config');

const PORT = Number(process.env.PORT) || 3000;
const BCRYPT_ROUNDS = 10;
const JWT_SECRET = process.env.JWT_SECRET || randomUUID() + randomUUID();

// ─── AI Client ─────────────────────────────────────────────────────────────────

let aiClient: GoogleGenAI | null = null;
function getAiClient() {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error('GEMINI_API_KEY environment variable is required.');
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: { headers: { 'User-Agent': 'aistudio-build' } }
    });
  }
  return aiClient;
}

// ─── Interfaces ────────────────────────────────────────────────────────────────

interface Employee {
  id: string;
  loginId: string;
  passwordHash: string;
  name: string;
  active: boolean;
  leadCount: number;
}

interface Lead {
  id: string;
  name: string;
  phone: string;
  assignedTo: string | null;
  status: 'New' | 'Contacted' | 'Closed';
  notes?: string;
  followUpDate?: string;
  timeline?: { id: string; action: string; user: string; timestamp: string; notes?: string; }[];
  updatedAt?: string;
  createdAt: string;
}

interface AdminProfile {
  name: string;
  loginId: string;
  passwordHash: string;
}

// ─── Firestore Helpers ──────────────────────────────────────────────────────────

async function getEmployees(): Promise<Employee[]> {
  const snap = await employeesCol.get();
  return snap.docs.map(d => ({ id: d.id, ...d.data() } as Employee));
}

async function getLeads(): Promise<Lead[]> {
  const snap = await leadsCol.orderBy('createdAt', 'asc').get();
  return snap.docs.map(d => ({ id: d.id, ...d.data() } as Lead));
}

async function getAdminProfile(): Promise<AdminProfile> {
  const doc = await configCol.doc('adminProfile').get();
  if (!doc.exists) {
    // First-time: seed default admin
    const defaultAdmin: AdminProfile = {
      name: 'Admin',
      loginId: 'admin',
      passwordHash: bcrypt.hashSync('admin', BCRYPT_ROUNDS),
    };
    await configCol.doc('adminProfile').set(defaultAdmin);
    return defaultAdmin;
  }
  return doc.data() as AdminProfile;
}

/** leadCount = number of leads assigned to each employee (computed from Firestore) */
async function syncLeadCounts(emps: Employee[], allLeads: Lead[]): Promise<void> {
  const batch = db.batch();
  for (const emp of emps) {
    const count = allLeads.filter(l => l.assignedTo === emp.id).length;
    if (emp.leadCount !== count) {
      batch.update(employeesCol.doc(emp.id), { leadCount: count });
      emp.leadCount = count;
    }
  }
  await batch.commit();
}

function sanitizeEmployee(emp: Employee) {
  const { passwordHash, ...safe } = emp;
  return safe;
}

function sanitizeAdminProfile(profile: AdminProfile) {
  const { passwordHash, ...safe } = profile;
  return safe;
}

// ─── Input Limits ───────────────────────────────────────────────────────────────

const MAX_NAME_LEN = 120;
const MAX_LOGIN_ID_LEN = 60;
const MAX_PASSWORD_LEN = 128;
const MAX_NOTES_LEN = 5000;
const MAX_BULK_TEXT_LEN = 200_000;

function sanitizeStr(val: unknown, maxLen = 255): string {
  if (typeof val !== 'string') return '';
  return val.trim().substring(0, maxLen);
}

// ─── JWT Auth ───────────────────────────────────────────────────────────────────

interface AuthPayload {
  userId: string;
  role: 'admin' | 'employee';
}

function requireAuth(req: Request, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Authentication required' });
  }
  try {
    const payload = jwt.verify(authHeader.substring(7), JWT_SECRET) as AuthPayload;
    (req as any).auth = payload;
    next();
  } catch {
    return res.status(401).json({ error: 'Invalid or expired session. Please login again.' });
  }
}

function requireAdmin(req: Request, res: Response, next: NextFunction) {
  requireAuth(req, res, () => {
    if ((req as any).auth?.role !== 'admin') {
      return res.status(403).json({ error: 'Admin access required' });
    }
    next();
  });
}

// ─── Rate Limiters ──────────────────────────────────────────────────────────────

const loginRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, max: 20, standardHeaders: true, legacyHeaders: false,
  message: { error: 'Too many login attempts. Please wait 15 minutes and try again.' }
});

const apiRateLimiter = rateLimit({
  windowMs: 60 * 1000, max: 200, standardHeaders: true, legacyHeaders: false,
  message: { error: 'Too many requests. Please slow down.' }
});

// ─── Distribution Helper ─────────────────────────────────────────────────────────

/** Get employee with fewest leads assigned */
function pickAssignee(emps: Employee[]): Employee | null {
  if (emps.length === 0) return null;
  return [...emps].sort((a, b) => a.leadCount - b.leadCount)[0];
}

// ─── Server ─────────────────────────────────────────────────────────────────────

async function startServer() {
  const app = express();

  app.use(helmet({ contentSecurityPolicy: false }));
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ limit: '50mb', extended: true }));
  app.use('/api/', apiRateLimiter);

  // ── Login ─────────────────────────────────────────────────────────────────────

  app.post('/api/login', loginRateLimiter, async (req, res) => {
    try {
      const loginId  = sanitizeStr(req.body.loginId, MAX_LOGIN_ID_LEN);
      const password = sanitizeStr(req.body.password, MAX_PASSWORD_LEN);

      if (!loginId || !password) {
        return res.status(400).json({ error: 'Login ID and password are required' });
      }

      const adminProfile = await getAdminProfile();

      // Admin check
      if (loginId.toLowerCase() === adminProfile.loginId.toLowerCase()) {
        const valid = await bcrypt.compare(password, adminProfile.passwordHash);
        if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
        const token = jwt.sign({ userId: 'admin', role: 'admin' } as AuthPayload, JWT_SECRET, { expiresIn: '12h' });
        return res.json({ id: 'admin', role: 'admin', token });
      }

      // Employee check — query Firestore by loginId
      const empSnap = await employeesCol.where('loginId', '==', loginId).limit(1).get();
      if (empSnap.empty) return res.status(401).json({ error: 'Invalid credentials' });

      const empDoc = empSnap.docs[0];
      const emp = { id: empDoc.id, ...empDoc.data() } as Employee;

      const valid = await bcrypt.compare(password, emp.passwordHash);
      if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
      if (!emp.active) return res.status(403).json({ error: 'Account is inactive. Contact admin.' });

      const token = jwt.sign({ userId: emp.id, role: 'employee' } as AuthPayload, JWT_SECRET, { expiresIn: '12h' });
      return res.json({ id: emp.id, role: 'employee', token });

    } catch (err: any) {
      console.error('Login error:', err);
      return res.status(500).json({ error: 'Server error during login' });
    }
  });

  // ── State ─────────────────────────────────────────────────────────────────────

  app.get('/api/state', requireAuth, async (req, res) => {
    try {
      const [employees, leads, adminProfile] = await Promise.all([
        getEmployees(), getLeads(), getAdminProfile()
      ]);
      await syncLeadCounts(employees, leads);

      res.json({
        employees: employees.map(sanitizeEmployee),
        leads,
        adminProfile: sanitizeAdminProfile(adminProfile)
      });
    } catch (err: any) {
      console.error('State fetch error:', err);
      res.status(500).json({ error: 'Failed to fetch state' });
    }
  });

  // ── Employees ─────────────────────────────────────────────────────────────────

  app.post('/api/employees', requireAdmin, async (req, res) => {
    try {
      const name     = sanitizeStr(req.body.name, MAX_NAME_LEN);
      const loginId  = sanitizeStr(req.body.loginId, MAX_LOGIN_ID_LEN);
      const password = sanitizeStr(req.body.password, MAX_PASSWORD_LEN);

      if (!name || !loginId) return res.status(400).json({ error: 'Name and Login ID are required' });
      if (!password || password.length < 4) return res.status(400).json({ error: 'Password must be at least 4 characters' });

      // Check duplicate loginId in Firestore
      const [adminProfile, dupSnap] = await Promise.all([
        getAdminProfile(),
        employeesCol.where('loginId', '==', loginId).limit(1).get()
      ]);

      if (!dupSnap.empty || loginId.toLowerCase() === adminProfile.loginId.toLowerCase()) {
        return res.status(400).json({ error: 'Login ID already exists' });
      }

      const id = randomUUID();
      const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
      const emp: Employee = { id, loginId, passwordHash, name, active: true, leadCount: 0 };

      await employeesCol.doc(id).set(emp);
      res.json(sanitizeEmployee(emp));

    } catch (err: any) {
      console.error('Add employee error:', err);
      res.status(500).json({ error: 'Failed to create employee' });
    }
  });

  app.patch('/api/employees/:id', requireAdmin, async (req, res) => {
    try {
      const docRef = employeesCol.doc(req.params.id);
      const snap = await docRef.get();
      if (!snap.exists) return res.status(404).json({ error: 'Employee not found' });

      const emp = { id: snap.id, ...snap.data() } as Employee;
      const updates: Partial<Employee> = {};

      if (req.body.active !== undefined)  updates.active = Boolean(req.body.active);
      if (req.body.name !== undefined)    updates.name = sanitizeStr(req.body.name, MAX_NAME_LEN);

      if (req.body.loginId !== undefined) {
        const newLoginId = sanitizeStr(req.body.loginId, MAX_LOGIN_ID_LEN);
        const [adminProfile, dupSnap] = await Promise.all([
          getAdminProfile(),
          employeesCol.where('loginId', '==', newLoginId).limit(1).get()
        ]);
        const isDup = (!dupSnap.empty && dupSnap.docs[0].id !== req.params.id)
          || newLoginId.toLowerCase() === adminProfile.loginId.toLowerCase();
        if (isDup) return res.status(400).json({ error: 'Login ID already exists' });
        updates.loginId = newLoginId;
      }

      if (req.body.password !== undefined) {
        const newPwd = sanitizeStr(req.body.password, MAX_PASSWORD_LEN);
        if (newPwd.length < 4) return res.status(400).json({ error: 'Password must be at least 4 characters' });
        updates.passwordHash = await bcrypt.hash(newPwd, BCRYPT_ROUNDS);
      }

      await docRef.update(updates as any);
      const updated = { ...emp, ...updates };
      res.json(sanitizeEmployee(updated));

    } catch (err: any) {
      console.error('Update employee error:', err);
      res.status(500).json({ error: 'Failed to update employee' });
    }
  });

  app.delete('/api/employees/:id', requireAdmin, async (req, res) => {
    try {
      const empId = req.params.id;

      // Unassign leads for this employee
      const assignedSnap = await leadsCol.where('assignedTo', '==', empId).get();
      const batch = db.batch();
      assignedSnap.docs.forEach(d => batch.update(d.ref, { assignedTo: null }));
      batch.delete(employeesCol.doc(empId));
      await batch.commit();

      res.json({ success: true });
    } catch (err: any) {
      console.error('Delete employee error:', err);
      res.status(500).json({ error: 'Failed to delete employee' });
    }
  });

  // ── Admin Profile ─────────────────────────────────────────────────────────────

  app.patch('/api/admin/profile', requireAdmin, async (req, res) => {
    try {
      const adminProfile = await getAdminProfile();
      const updates: Partial<AdminProfile> = {};

      if (req.body.loginId !== undefined) {
        const newLoginId = sanitizeStr(req.body.loginId, MAX_LOGIN_ID_LEN);
        const dupSnap = await employeesCol.where('loginId', '==', newLoginId).limit(1).get();
        if (!dupSnap.empty) return res.status(400).json({ error: 'Login ID already exists as an employee' });
        updates.loginId = newLoginId;
      }
      if (req.body.name !== undefined) {
        updates.name = sanitizeStr(req.body.name, MAX_NAME_LEN);
      }
      if (req.body.password !== undefined) {
        const newPwd = sanitizeStr(req.body.password, MAX_PASSWORD_LEN);
        if (newPwd.length < 4) return res.status(400).json({ error: 'Password must be at least 4 characters' });
        updates.passwordHash = await bcrypt.hash(newPwd, BCRYPT_ROUNDS);
      }

      await configCol.doc('adminProfile').update(updates as any);
      const updated = { ...adminProfile, ...updates };
      res.json(sanitizeAdminProfile(updated));

    } catch (err: any) {
      console.error('Update admin profile error:', err);
      res.status(500).json({ error: 'Failed to update admin profile' });
    }
  });

  // ── Leads: AI Screenshot Scanner ─────────────────────────────────────────────

  app.post('/api/leads/parse-screenshots', requireAdmin, async (req, res) => {
    try {
      const { images } = req.body;
      if (!images || !Array.isArray(images) || images.length === 0) {
        return res.status(400).json({ error: 'At least one screenshot image is required.' });
      }
      if (images.length > 20) {
        return res.status(400).json({ error: 'Maximum 20 screenshots allowed per request.' });
      }

      const ai = getAiClient();
      const allExtracted: { name: string; phone: string }[] = [];

      for (const imageStr of images) {
        let mimeType = 'image/png', base64Data = imageStr;
        const match = imageStr.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,(.+)$/);
        if (match) { mimeType = match[1]; base64Data = match[2]; }

        const response = await ai.models.generateContent({
          model: 'gemini-3.5-flash',
          contents: { parts: [
            { inlineData: { mimeType, data: base64Data } },
            { text: `Extract all lead Names and Phone numbers from this screenshot. Return JSON.` }
          ]},
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                leads: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: {
                  name: { type: Type.STRING }, phone: { type: Type.STRING }
                }, required: ['name', 'phone'] }}
              }, required: ['leads']
            }
          }
        });

        if (response.text) {
          try {
            const parsed = JSON.parse(response.text.trim());
            if (parsed?.leads) allExtracted.push(...parsed.leads);
          } catch {}
        }
      }

      const validLeads = allExtracted.filter(l => l.name?.trim() || l.phone?.trim());
      const employees = await getEmployees();
      const batch = db.batch();
      const createdLeads: Lead[] = [];

      for (const leadData of validLeads) {
        const id = randomUUID();
        const now = new Date().toISOString();
        const assignee = pickAssignee(employees);

        const lead: Lead = {
          id, name: sanitizeStr(leadData.name, MAX_NAME_LEN) || 'Lead - Mobile',
          phone: sanitizeStr(leadData.phone, 20), assignedTo: assignee?.id || null,
          status: 'New', createdAt: now,
          timeline: [
            { id: randomUUID(), action: 'Lead created via Screenshot OCR scan', user: 'System', timestamp: now },
            ...(assignee ? [{ id: randomUUID(), action: `Auto-distributed to ${assignee.name}`, user: 'System', timestamp: now }] : [])
          ]
        };

        batch.set(leadsCol.doc(id), lead);
        if (assignee) assignee.leadCount++;
        createdLeads.push(lead);
      }

      await batch.commit();
      res.json({ success: true, extractedCount: validLeads.length, createdCount: createdLeads.length, leads: createdLeads });

    } catch (error: any) {
      console.error('Screenshot parse error:', error);
      res.status(500).json({ error: error.message || 'Failed to parse screenshots.' });
    }
  });

  // ── Leads: Bulk Text ──────────────────────────────────────────────────────────

  app.post('/api/leads/bulk', requireAdmin, async (req, res) => {
    try {
      const rawText = sanitizeStr(req.body.rawText, MAX_BULK_TEXT_LEN);
      if (!rawText) return res.status(400).json({ error: 'rawText is required' });

      const lines = rawText.split('\n').map(l => l.trim()).filter(l => l);
      const employees = await getEmployees();
      const batch = db.batch();
      let count = 0;

      for (const line of lines) {
        const parts = line.split(' ');
        let phone = '', name = '';
        if (parts.length > 1 && /\d/.test(parts[parts.length - 1])) {
          phone = parts[parts.length - 1].substring(0, 20);
          name  = parts.slice(0, -1).join(' ').substring(0, MAX_NAME_LEN);
        } else {
          name = line.substring(0, MAX_NAME_LEN);
        }

        const id  = randomUUID();
        const now = new Date().toISOString();
        const assignee = pickAssignee(employees);

        const lead: Lead = {
          id, name, phone, assignedTo: assignee?.id || null, status: 'New', createdAt: now,
          timeline: [
            { id: randomUUID(), action: 'Lead created via bulk raw text upload', user: 'Admin', timestamp: now },
            ...(assignee ? [{ id: randomUUID(), action: `Auto-distributed to ${assignee.name}`, user: 'System', timestamp: now }] : [])
          ]
        };
        batch.set(leadsCol.doc(id), lead);
        if (assignee) assignee.leadCount++;
        count++;
      }

      await batch.commit();
      res.json({ success: true, count });

    } catch (err: any) {
      console.error('Bulk leads error:', err);
      res.status(500).json({ error: 'Failed to add leads' });
    }
  });

  // ── Leads: Bulk JSON (CSV import) ─────────────────────────────────────────────

  app.post('/api/leads/bulk-json', requireAdmin, async (req, res) => {
    try {
      const { leads: importedLeads, actorName } = req.body;
      if (!importedLeads || !Array.isArray(importedLeads)) {
        return res.status(400).json({ error: 'Leads array is required' });
      }
      if (importedLeads.length > 5000) return res.status(400).json({ error: 'Maximum 5000 leads per import' });

      const actor = sanitizeStr(actorName, MAX_NAME_LEN) || 'Admin';
      const employees = await getEmployees();
      const batch = db.batch();
      let count = 0;

      for (const item of importedLeads) {
        if (!item.name && !item.phone) continue;
        const id  = randomUUID();
        const now = new Date().toISOString();
        const assignee = pickAssignee(employees);

        const lead: Lead = {
          id, name: sanitizeStr(item.name, MAX_NAME_LEN) || 'Lead - Mobile',
          phone: sanitizeStr(item.phone, 20), assignedTo: assignee?.id || null,
          status: item.status || 'New', notes: sanitizeStr(item.notes, MAX_NOTES_LEN),
          followUpDate: item.followUpDate || undefined, createdAt: now,
          timeline: [
            { id: randomUUID(), action: 'Lead imported via CSV file', user: actor, timestamp: now },
            ...(assignee ? [{ id: randomUUID(), action: `Auto-distributed to ${assignee.name}`, user: 'System', timestamp: now }] : [])
          ]
        };
        batch.set(leadsCol.doc(id), lead);
        if (assignee) assignee.leadCount++;
        count++;
      }

      await batch.commit();
      res.json({ success: true, count });

    } catch (err: any) {
      console.error('Bulk JSON import error:', err);
      res.status(500).json({ error: 'Failed to import leads' });
    }
  });

  // ── Lead: Update ──────────────────────────────────────────────────────────────

  app.patch('/api/leads/:id', requireAuth, async (req, res) => {
    try {
      const docRef = leadsCol.doc(req.params.id);
      const snap = await docRef.get();
      if (!snap.exists) return res.status(404).json({ error: 'Lead not found' });

      const lead = { id: snap.id, ...snap.data() } as Lead;
      const auth = (req as any).auth as AuthPayload;

      // Employees can only update their own leads
      if (auth.role === 'employee' && lead.assignedTo !== auth.userId) {
        return res.status(403).json({ error: 'You can only update leads assigned to you' });
      }

      const timeline = lead.timeline || [];
      const updates: Partial<Lead> = {};
      const actor = sanitizeStr(req.body.actorName, MAX_NAME_LEN) || 'System';
      const timestamp = new Date().toISOString();

      const allowedStatuses = ['New', 'Contacted', 'Closed'];
      if (req.body.status && req.body.status !== lead.status) {
        if (!allowedStatuses.includes(req.body.status)) return res.status(400).json({ error: 'Invalid status value' });
        timeline.push({ id: randomUUID(), action: `Status updated from "${lead.status}" to "${req.body.status}"`, user: actor, timestamp });
        updates.status = req.body.status;
      }

      if (req.body.assignedTo !== undefined && req.body.assignedTo !== lead.assignedTo) {
        if (auth.role !== 'admin') return res.status(403).json({ error: 'Only admin can reassign leads' });
        const employees = await getEmployees();
        const oldName = lead.assignedTo ? (employees.find(e => e.id === lead.assignedTo)?.name || 'Unknown') : 'Unassigned';
        const newName = req.body.assignedTo ? (employees.find(e => e.id === req.body.assignedTo)?.name || 'Unknown') : 'Unassigned';
        timeline.push({ id: randomUUID(), action: `Assignment changed from ${oldName} to ${newName}`, user: actor, timestamp });
        updates.assignedTo = req.body.assignedTo;
      }

      if (req.body.notes !== undefined && req.body.notes !== lead.notes) {
        const sanitizedNotes = sanitizeStr(req.body.notes, MAX_NOTES_LEN);
        timeline.push({ id: randomUUID(), action: 'Updated lead note details', user: actor, timestamp, notes: sanitizedNotes });
        updates.notes = sanitizedNotes;
      }

      if (req.body.followUpDate !== undefined && req.body.followUpDate !== lead.followUpDate) {
        const displayDate = req.body.followUpDate
          ? new Date(req.body.followUpDate).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })
          : 'Removed';
        timeline.push({ id: randomUUID(), action: `Set follow-up reminder for ${displayDate}`, user: actor, timestamp });
        updates.followUpDate = req.body.followUpDate;
      }

      updates.timeline = timeline;
      updates.updatedAt = timestamp;

      await docRef.update(updates as any);
      const updatedLead = { ...lead, ...updates };
      res.json(updatedLead);

    } catch (err: any) {
      console.error('Update lead error:', err);
      res.status(500).json({ error: 'Failed to update lead' });
    }
  });

  // ── Lead: Delete ──────────────────────────────────────────────────────────────

  app.delete('/api/leads/:id', requireAdmin, async (req, res) => {
    try {
      const snap = await leadsCol.doc(req.params.id).get();
      if (!snap.exists) return res.status(404).json({ error: 'Lead not found' });
      await leadsCol.doc(req.params.id).delete();
      res.json({ success: true });
    } catch (err: any) {
      console.error('Delete lead error:', err);
      res.status(500).json({ error: 'Failed to delete lead' });
    }
  });

  // ── Vite Integration ──────────────────────────────────────────────────────────

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, response) => {
      response.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
    console.log(`🔥 Firestore database connected`);
    console.log(`🔒 JWT authentication enabled`);
  });
}

startServer();
