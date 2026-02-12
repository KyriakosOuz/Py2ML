'use client';

import '../globals.css';

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <title>Py2ML Academy</title>
      </head>
      <body className="bg-background text-text-primary font-sans antialiased">
        <div className="min-h-screen flex items-center justify-center px-4">
          {children}
        </div>
      </body>
    </html>
  );
}
