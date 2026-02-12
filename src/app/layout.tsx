'use client';

import { useState } from 'react';
import { usePathname } from 'next/navigation';
import { SessionProvider } from 'next-auth/react';
import './globals.css';
import Sidebar from '@/components/layout/Sidebar';
import Header from '@/components/layout/Header';
import MobileNav from '@/components/layout/MobileNav';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Auth pages get a clean layout without sidebar/header
  const isAuthPage = pathname === '/login' || pathname === '/signup';

  return (
    <html lang="en">
      <head>
        <title>Py2ML Academy — Python to Machine Learning</title>
        <meta name="description" content="Learn Python from beginner to AI/ML-ready with interactive exercises, quizzes, and projects" />
      </head>
      <body className="bg-background text-text-primary font-sans antialiased">
        <SessionProvider>
          {isAuthPage ? (
            <div className="min-h-screen flex items-center justify-center px-4">
              {children}
            </div>
          ) : (
            <>
              <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
              <div className="lg:ml-72 min-h-screen flex flex-col">
                <Header onMenuToggle={() => setSidebarOpen(!sidebarOpen)} />
                <main className="flex-1 pb-16 lg:pb-0">
                  {children}
                </main>
              </div>
              <MobileNav />
            </>
          )}
        </SessionProvider>
      </body>
    </html>
  );
}
