'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BookOpen, LayoutDashboard, FolderKanban, Trophy, Settings,
  ChevronDown, ChevronRight, CheckCircle2, Circle, X
} from 'lucide-react';
import type { CurriculumStage } from '@/types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const pathname = usePathname();
  const [curriculum, setCurriculum] = useState<CurriculumStage[]>([]);
  const [expandedStages, setExpandedStages] = useState<Set<string>>(new Set());
  const [expandedModules, setExpandedModules] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetch('/api/curriculum')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setCurriculum(data);
          // Auto-expand first stage
          if (data.length > 0) {
            setExpandedStages(new Set([data[0].id]));
            if (data[0].modules?.length > 0) {
              setExpandedModules(new Set([data[0].modules[0].id]));
            }
          }
        }
      })
      .catch(console.error);
  }, []);

  const toggleStage = (id: string) => {
    setExpandedStages((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleModule = (id: string) => {
    setExpandedModules((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const navItems = [
    { href: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { href: '/curriculum', icon: BookOpen, label: 'Curriculum' },
    { href: '/projects', icon: FolderKanban, label: 'Projects' },
    { href: '/portfolio', icon: Trophy, label: 'Portfolio' },
  ];

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={onClose} />
      )}

      <aside
        className={`fixed top-0 left-0 h-full w-72 bg-surface border-r border-border z-50 transform transition-transform duration-200 lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } flex flex-col overflow-hidden`}
      >
        {/* Logo */}
        <div className="p-4 border-b border-border flex items-center justify-between flex-shrink-0">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-background font-bold font-mono text-sm">Py</span>
            </div>
            <div>
              <h1 className="font-serif text-lg text-text-primary leading-none">Py2ML</h1>
              <p className="text-[10px] text-text-muted tracking-wider uppercase">Academy</p>
            </div>
          </Link>
          <button onClick={onClose} className="lg:hidden text-text-muted hover:text-text-primary">
            <X size={20} />
          </button>
        </div>

        {/* Nav links */}
        <nav className="p-3 space-y-1 flex-shrink-0">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                pathname === item.href
                  ? 'bg-primary/10 text-primary'
                  : 'text-text-secondary hover:bg-surface-light hover:text-text-primary'
              }`}
            >
              <item.icon size={18} />
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="border-t border-border mx-3" />

        {/* Curriculum tree */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <p className="text-xs text-text-muted uppercase tracking-wider px-3 mb-2">Lessons</p>
          {curriculum.map((stage) => (
            <div key={stage.id}>
              <button
                onClick={() => toggleStage(stage.id)}
                className="flex items-center gap-2 w-full px-3 py-1.5 text-sm text-text-secondary hover:text-text-primary rounded-lg hover:bg-surface-light transition-colors"
              >
                {expandedStages.has(stage.id) ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                <span className="font-medium truncate">{stage.title}</span>
              </button>
              {expandedStages.has(stage.id) && stage.modules?.map((mod) => (
                <div key={mod.id} className="ml-3">
                  <button
                    onClick={() => toggleModule(mod.id)}
                    className="flex items-center gap-2 w-full px-3 py-1 text-xs text-text-muted hover:text-text-secondary rounded transition-colors"
                  >
                    {expandedModules.has(mod.id) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    <span className="truncate">{mod.title}</span>
                  </button>
                  {expandedModules.has(mod.id) && mod.lessons?.map((lesson) => (
                    <Link
                      key={lesson.id}
                      href={`/lesson/${lesson.id}`}
                      className={`flex items-center gap-2 ml-5 px-3 py-1 text-xs rounded transition-colors ${
                        pathname === `/lesson/${lesson.id}`
                          ? 'text-primary bg-primary/10'
                          : 'text-text-muted hover:text-text-secondary hover:bg-surface-light'
                      }`}
                    >
                      {lesson.completed ? (
                        <CheckCircle2 size={12} className="text-success flex-shrink-0" />
                      ) : (
                        <Circle size={12} className="flex-shrink-0" />
                      )}
                      <span className="truncate">{lesson.title}</span>
                    </Link>
                  ))}
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Admin link */}
        <div className="p-3 border-t border-border flex-shrink-0">
          <Link
            href="/admin"
            className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
              pathname.startsWith('/admin')
                ? 'bg-primary/10 text-primary'
                : 'text-text-muted hover:bg-surface-light hover:text-text-secondary'
            }`}
          >
            <Settings size={18} />
            Admin
          </Link>
        </div>
      </aside>
    </>
  );
}
