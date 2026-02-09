'use client';

import { useState, useEffect } from 'react';
import { Menu, Flame } from 'lucide-react';

interface HeaderProps {
  onMenuToggle: () => void;
}

export default function Header({ onMenuToggle }: HeaderProps) {
  const [streak, setStreak] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    fetch('/api/dashboard')
      .then((res) => res.json())
      .then((data) => {
        if (data.streak !== undefined) setStreak(data.streak);
        if (data.totalLessons > 0) {
          setProgress(Math.round((data.completedLessons / data.totalLessons) * 100));
        }
      })
      .catch(console.error);
  }, []);

  return (
    <header className="h-14 bg-surface border-b border-border flex items-center justify-between px-4 lg:px-6">
      <button
        onClick={onMenuToggle}
        className="lg:hidden text-text-secondary hover:text-text-primary"
      >
        <Menu size={24} />
      </button>

      <div className="hidden lg:block" />

      <div className="flex items-center gap-4">
        {/* Streak */}
        <div className="flex items-center gap-1.5 text-sm">
          <Flame size={16} className={streak > 0 ? 'text-primary' : 'text-text-muted'} />
          <span className={streak > 0 ? 'text-primary font-mono font-medium' : 'text-text-muted font-mono'}>
            {streak}
          </span>
        </div>

        {/* Progress ring */}
        <div className="relative w-8 h-8">
          <svg className="w-8 h-8 -rotate-90" viewBox="0 0 36 36">
            <circle
              className="text-surface-light"
              strokeWidth="3"
              stroke="currentColor"
              fill="transparent"
              r="15.9155"
              cx="18"
              cy="18"
            />
            <circle
              className="text-primary"
              strokeWidth="3"
              strokeDasharray={`${progress}, 100`}
              strokeLinecap="round"
              stroke="currentColor"
              fill="transparent"
              r="15.9155"
              cx="18"
              cy="18"
            />
          </svg>
          <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono text-text-muted">
            {progress}%
          </span>
        </div>
      </div>
    </header>
  );
}
