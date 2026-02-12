import { NextResponse } from 'next/server';
import bcrypt from 'bcryptjs';
import { prisma } from '@/lib/db';

export const dynamic = 'force-dynamic';

const ADMIN_EMAIL = 'kyriakos.ouzounis@gmail.com';
const ADMIN_PASSWORD = 'Py2ML-Admin-2024!';

export async function GET() {
  try {
    // Check if admin already has a password set
    const existing = await prisma.user.findUnique({
      where: { email: ADMIN_EMAIL },
      select: { id: true, role: true, passwordHash: true },
    });

    if (!existing) {
      // Create admin user from scratch
      const hash = await bcrypt.hash(ADMIN_PASSWORD, 12);
      const user = await prisma.user.create({
        data: {
          email: ADMIN_EMAIL,
          name: 'Kyriakos',
          role: 'ADMIN',
          passwordHash: hash,
        },
      });
      return NextResponse.json({
        status: 'created',
        message: `Admin user created: ${user.email} (role: ADMIN)`,
      });
    }

    // User exists (from Google OAuth) — upgrade to admin + set password
    const hash = await bcrypt.hash(ADMIN_PASSWORD, 12);
    await prisma.user.update({
      where: { email: ADMIN_EMAIL },
      data: {
        role: 'ADMIN',
        passwordHash: hash,
      },
    });

    return NextResponse.json({
      status: 'updated',
      message: `User ${ADMIN_EMAIL} upgraded to ADMIN with password set.`,
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 500 }
    );
  }
}
