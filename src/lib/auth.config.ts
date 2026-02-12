import type { NextAuthConfig } from 'next-auth';
import Google from 'next-auth/providers/google';

/**
 * Auth config that's safe for Edge Runtime (used by middleware).
 * Does NOT include Credentials provider (requires bcryptjs/Node.js).
 * Does NOT include PrismaAdapter (requires Node.js).
 *
 * Includes JWT/session callbacks so middleware can read user role.
 */
export default {
  trustHost: true,
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  pages: {
    signIn: '/login',
  },
  callbacks: {
    jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.role = user.role || 'STUDENT';
      }
      if (!token.id && token.sub) {
        token.id = token.sub;
      }
      return token;
    },
    session({ session, token }) {
      if (session.user) {
        session.user.id = (token.id || token.sub) as string;
        session.user.role = (token.role as string) || 'STUDENT';
      }
      return session;
    },
  },
} satisfies NextAuthConfig;
