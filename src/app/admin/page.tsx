'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Settings, BookOpen, Edit } from 'lucide-react';
import Badge from '@/components/ui/Badge';

interface AdminLesson {
  id: string;
  title: string;
  slug: string;
  order: number;
  stageName: string;
  moduleName: string;
  exerciseCount: number;
  quizCount: number;
}

export default function AdminPage() {
  const [lessons, setLessons] = useState<AdminLesson[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/admin/lessons')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) setLessons(data);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-surface rounded w-48" />
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-16 bg-surface rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="flex items-center gap-3 mb-8">
        <Settings size={28} className="text-primary" />
        <div>
          <h1 className="font-serif text-3xl text-text-primary">Admin</h1>
          <p className="text-text-secondary text-sm">Manage lessons, exercises, and quizzes</p>
        </div>
      </div>

      <div className="bg-surface border border-border rounded-xl overflow-hidden">
        <div className="px-6 py-3 border-b border-border bg-surface-light">
          <div className="grid grid-cols-12 gap-4 text-xs text-text-muted uppercase tracking-wider">
            <div className="col-span-5">Lesson</div>
            <div className="col-span-2">Stage</div>
            <div className="col-span-2">Module</div>
            <div className="col-span-1">Ex.</div>
            <div className="col-span-1">Quiz</div>
            <div className="col-span-1"></div>
          </div>
        </div>
        <div className="divide-y divide-border">
          {lessons.map((lesson) => (
            <div key={lesson.id} className="px-6 py-3 hover:bg-surface-light/50 transition-colors">
              <div className="grid grid-cols-12 gap-4 items-center">
                <div className="col-span-5 flex items-center gap-3">
                  <BookOpen size={14} className="text-text-muted flex-shrink-0" />
                  <span className="text-sm text-text-primary truncate">{lesson.title}</span>
                </div>
                <div className="col-span-2">
                  <Badge variant="outline" size="sm">{lesson.stageName}</Badge>
                </div>
                <div className="col-span-2">
                  <span className="text-xs text-text-muted truncate">{lesson.moduleName}</span>
                </div>
                <div className="col-span-1">
                  <span className="text-xs font-mono text-text-muted">{lesson.exerciseCount}</span>
                </div>
                <div className="col-span-1">
                  <span className="text-xs font-mono text-text-muted">{lesson.quizCount}</span>
                </div>
                <div className="col-span-1 flex justify-end">
                  <Link
                    href={`/admin/lessons/${lesson.id}`}
                    className="text-text-muted hover:text-primary transition-colors"
                  >
                    <Edit size={14} />
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
