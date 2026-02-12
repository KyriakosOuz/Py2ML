import type { NextAuthConfig } from 'next-auth';
import Google from 'next-auth/providers/google';

/**
 * Auth config that's safe for Edge Runtime (used by middleware).
 * Does NOT include Credentials provider (requires bcryptjs/Node.js).
 * Does NOT include PrismaAdapter (requires Node.js).
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
} satisfies NextAuthConfig;
