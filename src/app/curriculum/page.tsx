'use client';

import { useState, useEffect } from 'react';
import CurriculumTree from '@/components/curriculum/CurriculumTree';
import type { CurriculumStage } from '@/types';

export default function CurriculumPage() {
  const [stages, setStages] = useState<CurriculumStage[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/curriculum')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) setStages(data);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="animate-pulse space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 bg-surface rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  const allLessons = stages.flatMap((s) => s.modules?.flatMap((m) => m.lessons) ?? []);
  const completedCount = allLessons.filter((l) => l.completed).length;

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="font-serif text-3xl text-text-primary mb-2">Curriculum</h1>
        <p className="text-text-secondary">
          {completedCount} of {allLessons.length} lessons completed
        </p>
      </div>
      <CurriculumTree stages={stages} />
    </div>
  );
}
