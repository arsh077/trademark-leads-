import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { randomUUID } from 'crypto';
import { GoogleGenAI, Type } from '@google/genai';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';

const PORT = 3000;
const BCRYPT_ROUNDS = 10;

// JWT Secret — must be set in environment for production
const JWT_SECRET = process.env.JWT_SECRET || randomUUID() + randomUUID();

// ─── AI Client ─────────────────────────────────────────────────────────────────

let aiClient: GoogleGenAI | null = null;
function getAiClient() {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY environment variable is required.');
    }
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
  loginId?: string;
  passwordHash?: string;
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
  timeline?: {
    id: string;
    action: string;
    user: string;
    timestamp: string;
    notes?: string;
  }[];
  updatedAt?: string;
  createdAt: string;
}

interface AdminProfile {
  name: string;
  loginId: string;
  passwordHash: string;
}

// ─── In-memory Database ─────────────────────────────────────────────────────────

let employees: Employee[] = [];
let leads: Lead[] = [];

// Default admin password: "admin" — hashed securely
let adminProfile: AdminProfile = {
  name: 'Admin',
  loginId: 'admin',
  passwordHash: bcrypt.hashSync('admin', BCRYPT_ROUNDS)
};

// ─── Helpers ────────────────────────────────────────────────────────────────────

function syncLeadCounts() {
  employees.forEach(emp => {
    emp.leadCount = leads.filter(l => l.assignedTo === emp.id).length;
  });
}

/** Strip password hash before sending employee data to clients */
function sanitizeEmployee(emp: Employee) {
  const { passwordHash, ...safe } = emp;
  return safe;
}

/** Strip password hash from admin profile before sending */
function sanitizeAdminProfile(profile: AdminProfile) {
  const { passwordHash, ...safe } = profile;
  return safe;
}

// Input length limits
const MAX_NAME_LEN = 120;
const MAX_LOGIN_ID_LEN = 60;
const MAX_PASSWORD_LEN = 128;
const MAX_NOTES_LEN = 5000;
const MAX_BULK_TEXT_LEN = 200_000;

function sanitizeStr(val: unknown, maxLen = 255): string {
  if (typeof val !== 'string') return '';
  return val.trim().substring(0, maxLen);
}

// ─── JWT Middleware ─────────────────────────────────────────────────────────────

interface AuthPayload {
  userId: string;
  role: 'admin' | 'employee';
}

function requireAuth(req: Request, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Authentication required' });
  }
  const token = authHeader.substring(7);
  try {
    const payload = jwt.verify(token, JWT_SECRET) as AuthPayload;
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
  windowMs: 15 * 60 * 1000,   // 15 minutes
  max: 20,                     // max 20 attempts per IP per 15 min
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many login attempts. Please wait 15 minutes and try again.' }
});

const apiRateLimiter = rateLimit({
  windowMs: 60 * 1000,         // 1 minute
  max: 200,                    // 200 requests per IP per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests. Please slow down.' }
});

// ─── Server ─────────────────────────────────────────────────────────────────────

async function startServer() {
  const app = express();

  // Security headers (CSP, HSTS, X-Frame-Options, etc.)
  app.use(helmet({
    contentSecurityPolicy: false, // Disabled so Vite HMR works in dev
  }));

  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ limit: '50mb', extended: true }));
  app.use('/api/', apiRateLimiter);

  // ── Auth ──────────────────────────────────────────────────────────────────────

  app.post('/api/login', loginRateLimiter, async (req, res) => {
    const loginId = sanitizeStr(req.body.loginId, MAX_LOGIN_ID_LEN);
    const password = sanitizeStr(req.body.password, MAX_PASSWORD_LEN);

    if (!loginId || !password) {
      return res.status(400).json({ error: 'Login ID and password are required' });
    }

    // Admin check
    if (loginId.toLowerCase() === adminProfile.loginId.toLowerCase()) {
      const valid = await bcrypt.compare(password, adminProfile.passwordHash);
      if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
      const token = jwt.sign({ userId: 'admin', role: 'admin' } as AuthPayload, JWT_SECRET, { expiresIn: '12h' });
      return res.json({ id: 'admin', role: 'admin', token });
    }

    // Employee check
    const emp = employees.find(e => e.loginId?.toLowerCase() === loginId.toLowerCase());
    if (!emp || !emp.passwordHash) return res.status(401).json({ error: 'Invalid credentials' });
    const valid = await bcrypt.compare(password, emp.passwordHash);
    if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
    if (!emp.active) return res.status(403).json({ error: 'Account is inactive. Contact admin.' });

    const token = jwt.sign({ userId: emp.id, role: 'employee' } as AuthPayload, JWT_SECRET, { expiresIn: '12h' });
    return res.json({ id: emp.id, role: 'employee', token });
  });

  // ── State ─────────────────────────────────────────────────────────────────────

  app.get('/api/state', requireAuth, (req, res) => {
    syncLeadCounts();
    // Employees: never send password hash
    const safeEmployees = employees.map(sanitizeEmployee);
    res.json({
      employees: safeEmployees,
      leads,
      adminProfile: sanitizeAdminProfile(adminProfile)
    });
  });

  // ── Employees ─────────────────────────────────────────────────────────────────

  app.post('/api/employees', requireAdmin, async (req, res) => {
    const name    = sanitizeStr(req.body.name, MAX_NAME_LEN);
    const loginId = sanitizeStr(req.body.loginId, MAX_LOGIN_ID_LEN);
    const password = sanitizeStr(req.body.password, MAX_PASSWORD_LEN);

    if (!name || !loginId) {
      return res.status(400).json({ error: 'Name and Login ID are required' });
    }
    if (!password || password.length < 4) {
      return res.status(400).json({ error: 'Password must be at least 4 characters' });
    }

    const isDuplicate = employees.some(e => e.loginId?.toLowerCase() === loginId.toLowerCase())
      || loginId.toLowerCase() === adminProfile.loginId.toLowerCase();
    if (isDuplicate) return res.status(400).json({ error: 'Login ID already exists' });

    const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
    const emp: Employee = {
      id: randomUUID(),
      loginId,
      passwordHash,
      name,
      active: true,
      leadCount: 0
    };
    employees.push(emp);
    res.json(sanitizeEmployee(emp));
  });

  app.patch('/api/employees/:id', requireAdmin, async (req, res) => {
    const emp = employees.find(e => e.id === req.params.id);
    if (!emp) return res.status(404).json({ error: 'Employee not found' });

    if (req.body.active !== undefined) {
      emp.active = Boolean(req.body.active);
    }
    if (req.body.name !== undefined) {
      emp.name = sanitizeStr(req.body.name, MAX_NAME_LEN);
    }
    if (req.body.loginId !== undefined) {
      const newLoginId = sanitizeStr(req.body.loginId, MAX_LOGIN_ID_LEN);
      const isDuplicate = employees.some(e => e.loginId?.toLowerCase() === newLoginId.toLowerCase() && e.id !== req.params.id)
        || newLoginId.toLowerCase() === adminProfile.loginId.toLowerCase();
      if (isDuplicate) return res.status(400).json({ error: 'Login ID already exists' });
      emp.loginId = newLoginId;
    }
    if (req.body.password !== undefined) {
      const newPwd = sanitizeStr(req.body.password, MAX_PASSWORD_LEN);
      if (newPwd.length < 4) return res.status(400).json({ error: 'Password must be at least 4 characters' });
      emp.passwordHash = await bcrypt.hash(newPwd, BCRYPT_ROUNDS);
    }
    res.json(sanitizeEmployee(emp));
  });

  app.delete('/api/employees/:id', requireAdmin, (req, res) => {
    employees = employees.filter(e => e.id !== req.params.id);
    leads.forEach(l => {
      if (l.assignedTo === req.params.id) l.assignedTo = null;
    });
    res.json({ success: true });
  });

  // ── Admin Profile ─────────────────────────────────────────────────────────────

  app.patch('/api/admin/profile', requireAdmin, async (req, res) => {
    const { name, loginId, password } = req.body;

    if (loginId !== undefined) {
      const newLoginId = sanitizeStr(loginId, MAX_LOGIN_ID_LEN);
      const isDuplicate = employees.some(e => e.loginId?.toLowerCase() === newLoginId.toLowerCase());
      if (isDuplicate) return res.status(400).json({ error: 'Login ID already exists as an employee' });
      adminProfile.loginId = newLoginId;
    }
    if (name !== undefined) {
      adminProfile.name = sanitizeStr(name, MAX_NAME_LEN);
    }
    if (password !== undefined) {
      const newPwd = sanitizeStr(password, MAX_PASSWORD_LEN);
      if (newPwd.length < 4) return res.status(400).json({ error: 'Password must be at least 4 characters' });
      adminProfile.passwordHash = await bcrypt.hash(newPwd, BCRYPT_ROUNDS);
    }
    res.json(sanitizeAdminProfile(adminProfile));
  });

  // ── Leads ─────────────────────────────────────────────────────────────────────

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
      const allExtractedLeads: { name: string; phone: string }[] = [];

      for (const imageStr of images) {
        let mimeType = 'image/png';
        let base64Data = imageStr;
        const match = imageStr.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,(.+)$/);
        if (match) {
          mimeType = match[1];
          base64Data = match[2];
        }

        const imagePart = { inlineData: { mimeType, data: base64Data } };
        const textPart = {
          text: `You are an automated lead OCR scanner for Legal Success India. 
Analyze this screenshot image. Identify all potential customer contact/lead details. 
Specifically, extract any visible Names and Phone/Mobile numbers. 
Return a JSON object matching the requested schema. Ensure to clean up the phone numbers (remove formatting characters like spaces, hyphens, brackets, but preserve country codes if relevant).
If a contact name is missing but a phone number is visible, use a descriptive placeholder like "Lead - Mobile" or similar.
If no contacts are found, return an empty array.`
        };

        const response = await ai.models.generateContent({
          model: 'gemini-3.5-flash',
          contents: { parts: [imagePart, textPart] },
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                leads: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      name: { type: Type.STRING, description: 'Name of the lead/contact' },
                      phone: { type: Type.STRING, description: 'Clean phone number/mobile number' }
                    },
                    required: ['name', 'phone']
                  }
                }
              },
              required: ['leads']
            }
          }
        });

        const textResult = response.text;
        if (textResult) {
          try {
            const parsed = JSON.parse(textResult.trim());
            if (parsed && Array.isArray(parsed.leads)) {
              allExtractedLeads.push(...parsed.leads);
            }
          } catch (e) {
            console.error('Error parsing JSON response from Gemini:', e);
          }
        }
      }

      const validLeads = allExtractedLeads.filter(l => l.name?.trim() || l.phone?.trim());
      syncLeadCounts();

      const createdLeads: Lead[] = [];

      for (const leadData of validLeads) {
        const lead: Lead = {
          id: randomUUID(),
          name: sanitizeStr(leadData.name, MAX_NAME_LEN) || 'Lead - Mobile',
          phone: sanitizeStr(leadData.phone, 20),
          assignedTo: null,
          status: 'New',
          createdAt: new Date().toISOString(),
          timeline: [{
            id: randomUUID(),
            action: 'Lead created via Screenshot OCR scan',
            user: 'System',
            timestamp: new Date().toISOString()
          }]
        };

        if (employees.length > 0) {
          const sortedEmployees = [...employees].sort((a, b) => a.leadCount - b.leadCount);
          const assignee = sortedEmployees[0];
          lead.assignedTo = assignee.id;
          assignee.leadCount++;
          lead.timeline!.push({
            id: randomUUID(),
            action: `Auto-distributed to ${assignee.name} (Allocated offline)`,
            user: 'System',
            timestamp: new Date().toISOString()
          });
        }
        createdLeads.push(lead);
      }

      leads.push(...createdLeads);
      res.json({ success: true, extractedCount: validLeads.length, createdCount: createdLeads.length, leads: createdLeads });

    } catch (error: any) {
      console.error('Error in parse-screenshots endpoint:', error);
      res.status(500).json({ error: error.message || 'Failed to parse screenshots.' });
    }
  });

  app.post('/api/leads/bulk', requireAdmin, (req, res) => {
    const rawText = sanitizeStr(req.body.rawText, MAX_BULK_TEXT_LEN);
    if (!rawText) return res.status(400).json({ error: 'rawText is required' });

    const lines = rawText.split('\n').map(l => l.trim()).filter(l => l);
    syncLeadCounts();

    const newLeads: Lead[] = [];

    for (const line of lines) {
      const parts = line.split(' ');
      let phone = '';
      let name = '';

      if (parts.length > 1) {
        const lastPart = parts[parts.length - 1];
        if (/\d/.test(lastPart)) {
          phone = lastPart.substring(0, 20);
          name = parts.slice(0, -1).join(' ').substring(0, MAX_NAME_LEN);
        } else {
          name = line.substring(0, MAX_NAME_LEN);
        }
      } else {
        name = line.substring(0, MAX_NAME_LEN);
      }

      const lead: Lead = {
        id: randomUUID(),
        name,
        phone,
        assignedTo: null,
        status: 'New',
        createdAt: new Date().toISOString(),
        timeline: [{
          id: randomUUID(),
          action: 'Lead created via bulk raw text upload',
          user: 'Admin',
          timestamp: new Date().toISOString()
        }]
      };

      if (employees.length > 0) {
        const sortedEmployees = [...employees].sort((a, b) => a.leadCount - b.leadCount);
        const assignee = sortedEmployees[0];
        lead.assignedTo = assignee.id;
        assignee.leadCount++;
        lead.timeline!.push({
          id: randomUUID(),
          action: `Auto-distributed to ${assignee.name} (Allocated offline)`,
          user: 'System',
          timestamp: new Date().toISOString()
        });
      }

      newLeads.push(lead);
    }

    leads.push(...newLeads);
    res.json({ success: true, count: newLeads.length });
  });

  app.post('/api/leads/bulk-json', requireAdmin, (req, res) => {
    const { leads: importedLeads, actorName } = req.body;
    if (!importedLeads || !Array.isArray(importedLeads)) {
      return res.status(400).json({ error: 'Leads array is required' });
    }
    if (importedLeads.length > 5000) {
      return res.status(400).json({ error: 'Maximum 5000 leads per import' });
    }

    const newLeads: Lead[] = [];
    const actor = sanitizeStr(actorName, MAX_NAME_LEN) || 'Admin';
    syncLeadCounts();

    for (const item of importedLeads) {
      if (!item.name && !item.phone) continue;

      const lead: Lead = {
        id: randomUUID(),
        name: sanitizeStr(item.name, MAX_NAME_LEN) || 'Lead - Mobile',
        phone: sanitizeStr(item.phone, 20),
        assignedTo: null,
        status: item.status || 'New',
        notes: sanitizeStr(item.notes, MAX_NOTES_LEN),
        followUpDate: item.followUpDate || undefined,
        createdAt: new Date().toISOString(),
        timeline: [{
          id: randomUUID(),
          action: 'Lead imported via CSV file',
          user: actor,
          timestamp: new Date().toISOString()
        }]
      };

      if (employees.length > 0) {
        const sortedEmployees = [...employees].sort((a, b) => a.leadCount - b.leadCount);
        const assignee = sortedEmployees[0];
        lead.assignedTo = assignee.id;
        assignee.leadCount++;
        lead.timeline!.push({
          id: randomUUID(),
          action: `Auto-distributed to ${assignee.name} (Allocated offline)`,
          user: 'System',
          timestamp: new Date().toISOString()
        });
      }

      newLeads.push(lead);
    }

    leads.push(...newLeads);
    res.json({ success: true, count: newLeads.length });
  });

  app.patch('/api/leads/:id', requireAuth, (req, res) => {
    const lead = leads.find(l => l.id === req.params.id);
    if (!lead) return res.status(404).json({ error: 'Lead not found' });

    const auth = (req as any).auth as AuthPayload;

    // Employees can only update their own leads
    if (auth.role === 'employee' && lead.assignedTo !== auth.userId) {
      return res.status(403).json({ error: 'You can only update leads assigned to you' });
    }

    if (!lead.timeline) lead.timeline = [];

    const actor = sanitizeStr(req.body.actorName, MAX_NAME_LEN) || 'System';
    const timestamp = new Date().toISOString();

    const allowedStatuses = ['New', 'Contacted', 'Closed'];
    if (req.body.status && req.body.status !== lead.status) {
      if (!allowedStatuses.includes(req.body.status)) {
        return res.status(400).json({ error: 'Invalid status value' });
      }
      lead.timeline.push({
        id: randomUUID(),
        action: `Status updated from "${lead.status}" to "${req.body.status}"`,
        user: actor,
        timestamp
      });
      lead.status = req.body.status;
    }

    // Only admin can reassign leads
    if (req.body.assignedTo !== undefined && req.body.assignedTo !== lead.assignedTo) {
      if (auth.role !== 'admin') {
        return res.status(403).json({ error: 'Only admin can reassign leads' });
      }
      const oldEmpName = lead.assignedTo ? (employees.find(e => e.id === lead.assignedTo)?.name || 'Unknown') : 'Unassigned';
      const newEmpName = req.body.assignedTo ? (employees.find(e => e.id === req.body.assignedTo)?.name || 'Unknown') : 'Unassigned';
      lead.timeline.push({
        id: randomUUID(),
        action: `Assignment changed from ${oldEmpName} to ${newEmpName}`,
        user: actor,
        timestamp
      });
      lead.assignedTo = req.body.assignedTo;
    }

    if (req.body.notes !== undefined && req.body.notes !== lead.notes) {
      lead.timeline.push({
        id: randomUUID(),
        action: 'Updated lead note details',
        user: actor,
        timestamp,
        notes: sanitizeStr(req.body.notes, MAX_NOTES_LEN)
      });
      lead.notes = sanitizeStr(req.body.notes, MAX_NOTES_LEN);
    }

    if (req.body.followUpDate !== undefined && req.body.followUpDate !== lead.followUpDate) {
      const displayDate = req.body.followUpDate
        ? new Date(req.body.followUpDate).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })
        : 'Removed';
      lead.timeline.push({
        id: randomUUID(),
        action: `Set follow-up reminder for ${displayDate}`,
        user: actor,
        timestamp
      });
      lead.followUpDate = req.body.followUpDate;
    }

    lead.updatedAt = timestamp;
    syncLeadCounts();
    res.json(lead);
  });

  app.delete('/api/leads/:id', requireAdmin, (req, res) => {
    const leadIndex = leads.findIndex(l => l.id === req.params.id);
    if (leadIndex === -1) return res.status(404).json({ error: 'Lead not found' });
    leads.splice(leadIndex, 1);
    syncLeadCounts();
    res.json({ success: true });
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
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
    console.log(`🔒 JWT authentication enabled`);
  });
}

startServer();
