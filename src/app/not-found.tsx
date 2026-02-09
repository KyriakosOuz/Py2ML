'use client';

import Link from 'next/link';
import { Home } from 'lucide-react';
import Button from '@/components/ui/Button';

export default function NotFound() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center">
        <p className="font-mono text-6xl text-primary mb-4">404</p>
        <h1 className="font-serif text-2xl text-text-primary mb-2">Page Not Found</h1>
        <p className="text-text-secondary mb-6">The page you&apos;re looking for doesn&apos;t exist.</p>
        <Link href="/">
          <Button variant="secondary">
            <Home size={16} className="mr-2" />
            Back Home
          </Button>
        </Link>
      </div>
    </div>
  );
}
