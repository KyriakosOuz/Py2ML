import { cookies } from 'next/headers';
import { v4 as uuidv4 } from 'uuid';
import { prisma } from './db';

const SESSION_COOKIE = 'py2ml_session';

export async function getOrCreateSession(): Promise<string> {
  const cookieStore = cookies();
  const existing = cookieStore.get(SESSION_COOKIE);

  if (existing?.value) {
    const session = await prisma.guestSession.findUnique({
      where: { id: existing.value },
    });
    if (session) {
      await prisma.guestSession.update({
        where: { id: session.id },
        data: { lastActiveAt: new Date() },
      });
      return session.id;
    }
  }

  const id = uuidv4();
  await prisma.guestSession.create({ data: { id } });
  cookieStore.set(SESSION_COOKIE, id, {
    httpOnly: true,
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 365, // 1 year
    path: '/',
  });

  return id;
}

export async function getSessionId(): Promise<string | null> {
  const cookieStore = cookies();
  return cookieStore.get(SESSION_COOKIE)?.value ?? null;
}
