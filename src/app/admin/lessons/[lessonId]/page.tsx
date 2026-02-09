'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft, Save } from 'lucide-react';
import Link from 'next/link';
import Button from '@/components/ui/Button';

interface LessonEditData {
  id: string;
  title: string;
  slug: string;
  content: string;
  commonMistakes: string;
  order: number;
}

export default function AdminLessonEditPage() {
  const params = useParams();
  const router = useRouter();
  const lessonId = params.lessonId as string;
  const [lesson, setLesson] = useState<LessonEditData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [commonMistakes, setCommonMistakes] = useState('');

  useEffect(() => {
    if (!lessonId) return;
    fetch(`/api/lessons/${lessonId}`)
      .then((res) => res.json())
      .then((data) => {
        if (!data.error) {
          setLesson(data);
          setTitle(data.title);
          setContent(data.content);
          setCommonMistakes(data.commonMistakes);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [lessonId]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await fetch('/api/admin/lessons', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: lessonId,
          title,
          content,
          commonMistakes,
        }),
      });
      router.push('/admin');
    } catch {
      // handle silently
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-surface rounded w-48" />
          <div className="h-64 bg-surface rounded-xl" />
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <p className="text-text-secondary">Lesson not found.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <Link href="/admin" className="text-text-muted hover:text-text-primary transition-colors">
            <ArrowLeft size={20} />
          </Link>
          <h1 className="font-serif text-2xl text-text-primary">Edit Lesson</h1>
        </div>
        <Button onClick={handleSave} loading={saving} size="sm">
          <Save size={14} className="mr-1" />
          Save
        </Button>
      </div>

      <div className="space-y-6">
        {/* Title */}
        <div>
          <label className="block text-sm text-text-secondary mb-2">Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full bg-surface border border-border rounded-lg px-4 py-2 text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
          />
        </div>

        {/* Content */}
        <div>
          <label className="block text-sm text-text-secondary mb-2">Content (Markdown)</label>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows={20}
            className="w-full bg-surface border border-border rounded-lg px-4 py-3 text-text-primary text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-y"
          />
        </div>

        {/* Common Mistakes */}
        <div>
          <label className="block text-sm text-text-secondary mb-2">Common Mistakes (Markdown)</label>
          <textarea
            value={commonMistakes}
            onChange={(e) => setCommonMistakes(e.target.value)}
            rows={10}
            className="w-full bg-surface border border-border rounded-lg px-4 py-3 text-text-primary text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-y"
          />
        </div>
      </div>
    </div>
  );
}
