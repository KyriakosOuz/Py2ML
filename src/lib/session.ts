import { auth } from './auth';

/**
 * Get the authenticated user's ID from the session.
 * Returns null if not authenticated.
 */
export async function getUserId(): Promise<string | null> {
  const session = await auth();
  return session?.user?.id ?? null;
}

/**
 * Get the authenticated user's ID, or throw a 401-style error.
 * Use this in API routes that require authentication.
 */
export async function requireUserId(): Promise<string> {
  const userId = await getUserId();
  if (!userId) throw new Error('Unauthorized');
  return userId;
}

/**
 * Check if the current user has the ADMIN role.
 */
export async function requireAdmin(): Promise<string> {
  const session = await auth();
  if (!session?.user?.id) throw new Error('Unauthorized');
  if (session.user.role !== 'ADMIN') throw new Error('Forbidden');
  return session.user.id;
}
