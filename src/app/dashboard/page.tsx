'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import ProgressOverview from '@/components/dashboard/ProgressOverview';
import StageProgress from '@/components/dashboard/StageProgress';
import StreakTracker from '@/components/dashboard/StreakTracker';
import SkillTags from '@/components/dashboard/SkillTags';
import RecentActivity from '@/components/dashboard/RecentActivity';
import Button from '@/components/ui/Button';
import type { DashboardData } from '@/types';

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/dashboard')
      .then((res) => res.json())
      .then((d) => {
        if (!d.error) setData(d);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-8">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-surface rounded w-48" />
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-32 bg-surface rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <p className="text-text-secondary">Failed to load dashboard data.</p>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-serif text-3xl text-text-primary mb-1">Dashboard</h1>
          <p className="text-text-secondary">Track your Python learning journey</p>
        </div>
        {data.nextLesson && (
          <Link href={`/lesson/${data.nextLesson.id}`}>
            <Button size="sm">
              Continue: {data.nextLesson.title}
              <ArrowRight size={14} className="ml-1" />
            </Button>
          </Link>
        )}
      </div>

      <div className="space-y-6">
        <ProgressOverview
          totalLessons={data.totalLessons}
          completedLessons={data.completedLessons}
          totalExercises={data.totalExercises}
          passedExercises={data.passedExercises}
          quizAverage={data.quizAverage}
        />

        <div className="grid md:grid-cols-2 gap-6">
          <StageProgress stages={data.stageProgress} />
          <div className="space-y-6">
            <StreakTracker streak={data.streak} />
            <SkillTags skills={data.earnedSkills} />
          </div>
        </div>

        <RecentActivity activities={data.recentActivity} />
      </div>
    </div>
  );
}
