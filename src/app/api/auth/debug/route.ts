import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export const dynamic = 'force-dynamic';

export async function GET() {
  const checks: Record<string, string> = {};

  // Check env vars (existence only, not values)
  checks.AUTH_SECRET = process.env.AUTH_SECRET ? 'SET' : 'MISSING';
  checks.NEXTAUTH_SECRET = process.env.NEXTAUTH_SECRET ? 'SET' : 'MISSING';
  checks.GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID ? 'SET' : 'MISSING';
  checks.GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET ? 'SET' : 'MISSING';
  checks.DATABASE_URL = process.env.DATABASE_URL ? 'SET' : 'MISSING';
  checks.NEXTAUTH_URL = process.env.NEXTAUTH_URL || 'NOT SET (auto-detected)';
  checks.AUTH_URL = process.env.AUTH_URL || 'NOT SET (auto-detected)';
  checks.AUTH_TRUST_HOST = process.env.AUTH_TRUST_HOST || 'NOT SET (using trustHost:true in code)';

  // Check database connectivity
  try {
    const userCount = await prisma.user.count();
    checks.DATABASE = `CONNECTED (${userCount} users)`;
  } catch (e) {
    checks.DATABASE = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
  }

  // Check if Account table exists (needed for OAuth)
  try {
    const accountCount = await prisma.account.count();
    checks.ACCOUNT_TABLE = `OK (${accountCount} accounts)`;
  } catch (e) {
    checks.ACCOUNT_TABLE = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
  }

  return NextResponse.json(checks, { status: 200 });
}
