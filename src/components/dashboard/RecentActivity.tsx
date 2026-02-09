'use client';

import { BookOpen, CheckCircle2, HelpCircle, FolderKanban, Trophy } from 'lucide-react';

interface Activity {
  type: string;
  metadata: Record<string, unknown>;
  createdAt: string;
}

interface RecentActivityProps {
  activities: Activity[];
}

const icons: Record<string, typeof BookOpen> = {
  LESSON_VIEW: BookOpen,
  EXERCISE_PASS: CheckCircle2,
  QUIZ_COMPLETE: HelpCircle,
  PROJECT_START: FolderKanban,
  PROJECT_COMPLETE: Trophy,
};

const labels: Record<string, string> = {
  LESSON_VIEW: 'Viewed lesson',
  EXERCISE_PASS: 'Passed exercise',
  QUIZ_COMPLETE: 'Completed quiz',
  PROJECT_START: 'Started project',
  PROJECT_COMPLETE: 'Completed project',
};

export default function RecentActivity({ activities }: RecentActivityProps) {
  if (activities.length === 0) {
    return (
      <div className="bg-surface border border-border rounded-xl p-6">
        <h3 className="font-serif text-lg text-text-primary mb-4">Recent Activity</h3>
        <p className="text-text-muted text-sm">No activity yet. Start a lesson to begin!</p>
      </div>
    );
  }

  return (
    <div className="bg-surface border border-border rounded-xl p-6">
      <h3 className="font-serif text-lg text-text-primary mb-4">Recent Activity</h3>
      <div className="space-y-3">
        {activities.map((a, i) => {
          const Icon = icons[a.type] || BookOpen;
          const label = labels[a.type] || a.type;
          const title = (a.metadata.lessonTitle || a.metadata.projectTitle || '') as string;
          const timeAgo = formatTimeAgo(a.createdAt);

          return (
            <div key={i} className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-surface-light flex items-center justify-center flex-shrink-0">
                <Icon size={14} className="text-text-muted" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-text-primary truncate">
                  {label}{title ? `: ${title}` : ''}
                </p>
                <p className="text-xs text-text-muted">{timeAgo}</p>
              </div>
              {a.type === 'QUIZ_COMPLETE' && a.metadata.score !== undefined && (
                <span className="text-xs font-mono text-primary">
                  {String(a.metadata.score)}/{String(a.metadata.total)}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatTimeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diff = now - then;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(dateStr).toLocaleDateString();
}
