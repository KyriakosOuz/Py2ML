# User Authentication Plan — Py2ML Academy

## Overview
Add real user accounts with login/signup, role-based access control (Admin vs Student), and a polished user-friendly experience. Currently the app uses anonymous cookie-based guest sessions with zero authentication.

---

## Tech Choice: NextAuth.js v5 (Auth.js)

**Why**: Most mature auth library for Next.js, handles sessions/CSRF/security automatically, works great with Prisma + Vercel.

**Auth Methods**:
- **Google OAuth** — one-click signup/login (user-friendly)
- **Email + Password** — classic fallback with bcrypt hashing

**New packages**: `next-auth@beta`, `@auth/prisma-adapter`, `bcryptjs`

---

## Step-by-Step Plan

### Step 1 — Prisma Schema Changes

Add a proper `User` model and NextAuth required tables:

```
User          — id, name, email, emailVerified, image, passwordHash, role (ADMIN/STUDENT)
Account       — OAuth provider accounts (Google, etc.)
Session       — Server-side sessions for NextAuth
VerificationToken — Email verification tokens
```

Update all progress relations (`Submission`, `QuizAttempt`, `ProjectProgress`, `EarnedSkill`, `ActivityLog`) to link to `userId` instead of `sessionId`.

**Migration strategy**: Drop `GuestSession` model entirely (no real users exist yet, all progress is disposable guest data). Clean break.

### Step 2 — NextAuth Configuration

- Create `src/lib/auth.ts` with NextAuth config
- Credentials provider (email/password with bcrypt)
- Google OAuth provider
- Prisma adapter for database sessions
- Callbacks to include `role` and `id` in the session/JWT
- Create `src/app/api/auth/[...nextauth]/route.ts`

### Step 3 — Signup API Route

- `POST /api/auth/signup` — creates new user with hashed password
- Validates email format + password strength with Zod
- Default role: `STUDENT`
- Returns error if email already taken

### Step 4 — Middleware (Route Protection)

Create `src/middleware.ts`:
- **Public routes**: `/`, `/login`, `/signup`, `/api/auth/*`, `/api/run-code`
- **Protected routes**: Everything else requires login
- **Admin routes**: `/admin/*` and `/api/admin/*` require `role === ADMIN`
- Redirects unauthenticated users to `/login`

### Step 5 — Login & Signup Pages

**`/login` page**:
- Email + password form
- "Sign in with Google" button
- Link to signup page
- Error messages for invalid credentials
- Clean, centered card layout matching the app's dark theme

**`/signup` page**:
- Name, email, password, confirm password
- "Sign up with Google" button
- Link to login page
- Password strength indicator
- Auto-login after signup

### Step 6 — Update Layout & Header

**Header changes**:
- Show user name/avatar on the right side
- Dropdown menu: Profile, Settings, Logout
- Keep streak counter and progress ring

**Sidebar changes**:
- Hide "Admin" link for non-admin users
- Show user's role badge

### Step 7 — Update All API Routes

Replace `getOrCreateSession()` → `auth()` from NextAuth in every API route:
- `/api/curriculum`
- `/api/lessons/[lessonId]`
- `/api/dashboard`
- `/api/projects` and `/api/projects/[projectId]`
- `/api/submit-exercise`
- `/api/submit-quiz`
- `/api/admin/lessons` — add admin role check

All queries switch from `sessionId` to `userId`.

### Step 8 — Admin Seeding

Create a seed script to make you (the repo owner) an admin:
- Seed your email as an ADMIN user
- Password set via environment variable or a default you change on first login

### Step 9 — Environment Variables

New `.env` variables needed:
```
NEXTAUTH_SECRET=<random-string>
NEXTAUTH_URL=http://localhost:3000  (auto-detected on Vercel)
GOOGLE_CLIENT_ID=<from-google-console>    (optional)
GOOGLE_CLIENT_SECRET=<from-google-console> (optional)
```

Google OAuth is optional — the app works fine with just email/password.

---

## File Changes Summary

| Action | File |
|--------|------|
| **New** | `src/lib/auth.ts` — NextAuth config |
| **New** | `src/app/api/auth/[...nextauth]/route.ts` — Auth API |
| **New** | `src/app/api/auth/signup/route.ts` — Signup endpoint |
| **New** | `src/app/(auth)/login/page.tsx` — Login page |
| **New** | `src/app/(auth)/signup/page.tsx` — Signup page |
| **New** | `src/app/(auth)/layout.tsx` — Auth pages layout (no sidebar) |
| **New** | `src/middleware.ts` — Route protection |
| **New** | `src/components/auth/UserMenu.tsx` — Header user dropdown |
| **Modify** | `prisma/schema.prisma` — Add User, Account, Session models; update relations |
| **Modify** | `src/lib/session.ts` — Remove or repurpose (replaced by NextAuth) |
| **Modify** | `src/components/layout/Header.tsx` — Add UserMenu |
| **Modify** | `src/components/layout/Sidebar.tsx` — Role-based admin link |
| **Modify** | All 8 API routes — Switch from sessionId to userId |
| **Modify** | `.env.example` — Add auth env vars |
| **Modify** | `package.json` — Add auth dependencies |

---

## What You'll Need To Do

1. **After I push**: Pull, run `npx prisma migrate dev`, seed admin account
2. **Google OAuth (optional)**: Create a project in Google Cloud Console, get client ID/secret, add to `.env`
3. **Vercel**: Add `NEXTAUTH_SECRET` environment variable (I'll generate one for you)
4. **First login**: Use your email/password to log in as admin
