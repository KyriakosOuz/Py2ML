'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, BookOpen, FolderKanban, Trophy } from 'lucide-react';

export default function MobileNav() {
  const pathname = usePathname();

  const items = [
    { href: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { href: '/curriculum', icon: BookOpen, label: 'Learn' },
    { href: '/projects', icon: FolderKanban, label: 'Projects' },
    { href: '/portfolio', icon: Trophy, label: 'Portfolio' },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-surface border-t border-border lg:hidden z-30">
      <div className="flex items-center justify-around py-2">
        {items.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`flex flex-col items-center gap-0.5 px-3 py-1 text-xs ${
              pathname === item.href ? 'text-primary' : 'text-text-muted'
            }`}
          >
            <item.icon size={20} />
            {item.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
