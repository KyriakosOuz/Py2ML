'use client';

import { Flame } from 'lucide-react';

interface StreakTrackerProps {
  streak: number;
}

export default function StreakTracker({ streak }: StreakTrackerProps) {
  return (
    <div className="bg-surface border border-border rounded-xl p-6 flex items-center gap-4">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
        streak > 0 ? 'bg-primary/10' : 'bg-surface-light'
      }`}>
        <Flame size={24} className={streak > 0 ? 'text-primary' : 'text-text-muted'} />
      </div>
      <div>
        <p className="font-mono text-2xl text-text-primary">{streak}</p>
        <p className="text-text-muted text-sm">Day Streak</p>
      </div>
    </div>
  );
}
