'use client';

import { ChevronDown, ChevronRight, CheckCircle2, Circle } from 'lucide-react';
import Link from 'next/link';
import ProgressBar from '@/components/ui/ProgressBar';
import type { CurriculumStage } from '@/types';

interface StageCardProps {
  stage: CurriculumStage;
  expanded: boolean;
  onToggle: () => void;
}

export default function StageCard({ stage, expanded, onToggle }: StageCardProps) {
  const allLessons = stage.modules?.flatMap((m) => m.lessons) ?? [];
  const completedCount = allLessons.filter((l) => l.completed).length;

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-6 hover:bg-surface-light transition-colors"
      >
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
            <span className="text-primary font-mono font-bold">{stage.order}</span>
          </div>
          <div className="text-left">
            <h3 className="font-serif text-lg text-text-primary">{stage.title}</h3>
            <p className="text-text-muted text-sm">{stage.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden sm:block w-32">
            <ProgressBar value={completedCount} max={allLessons.length} size="sm" />
          </div>
          <span className="text-text-muted text-sm font-mono">
            {completedCount}/{allLessons.length}
          </span>
          {expanded ? <ChevronDown size={20} className="text-text-muted" /> : <ChevronRight size={20} className="text-text-muted" />}
        </div>
      </button>

      {expanded && (
        <div className="border-t border-border">
          {stage.modules?.map((mod) => (
            <div key={mod.id} className="p-4 pl-8">
              <h4 className="text-text-secondary text-sm font-medium mb-3">{mod.title}</h4>
              <div className="space-y-1">
                {mod.lessons?.map((lesson) => (
                  <Link
                    key={lesson.id}
                    href={`/lesson/${lesson.id}`}
                    className="flex items-center gap-3 px-4 py-2.5 rounded-lg hover:bg-surface-light transition-colors group"
                  >
                    {lesson.completed ? (
                      <CheckCircle2 size={16} className="text-success flex-shrink-0" />
                    ) : (
                      <Circle size={16} className="text-text-muted group-hover:text-text-secondary flex-shrink-0" />
                    )}
                    <span className={`text-sm ${lesson.completed ? 'text-text-secondary' : 'text-text-primary'}`}>
                      {lesson.title}
                    </span>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
