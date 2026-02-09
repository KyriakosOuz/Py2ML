'use client';

interface ProgressOverviewProps {
  totalLessons: number;
  completedLessons: number;
  totalExercises: number;
  passedExercises: number;
  quizAverage: number;
}

export default function ProgressOverview({
  totalLessons,
  completedLessons,
  totalExercises,
  passedExercises,
  quizAverage,
}: ProgressOverviewProps) {
  const lessonPercent = totalLessons > 0 ? Math.round((completedLessons / totalLessons) * 100) : 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {/* Overall progress ring */}
      <div className="col-span-2 md:col-span-1 bg-surface border border-border rounded-xl p-6 flex flex-col items-center justify-center">
        <div className="relative w-24 h-24 mb-3">
          <svg className="w-24 h-24 -rotate-90" viewBox="0 0 36 36">
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
              strokeDasharray={`${lessonPercent}, 100`}
              strokeLinecap="round"
              stroke="currentColor"
              fill="transparent"
              r="15.9155"
              cx="18"
              cy="18"
            />
          </svg>
          <span className="absolute inset-0 flex items-center justify-center text-2xl font-mono font-bold text-primary">
            {lessonPercent}%
          </span>
        </div>
        <p className="text-text-muted text-sm">Overall Progress</p>
      </div>

      {/* Stats cards */}
      <div className="bg-surface border border-border rounded-xl p-6">
        <p className="text-text-muted text-xs uppercase tracking-wider mb-2">Lessons</p>
        <p className="font-mono text-2xl text-text-primary">
          {completedLessons}<span className="text-text-muted text-sm">/{totalLessons}</span>
        </p>
      </div>

      <div className="bg-surface border border-border rounded-xl p-6">
        <p className="text-text-muted text-xs uppercase tracking-wider mb-2">Exercises</p>
        <p className="font-mono text-2xl text-text-primary">
          {passedExercises}<span className="text-text-muted text-sm">/{totalExercises}</span>
        </p>
      </div>

      <div className="bg-surface border border-border rounded-xl p-6">
        <p className="text-text-muted text-xs uppercase tracking-wider mb-2">Quiz Avg</p>
        <p className="font-mono text-2xl text-text-primary">
          {quizAverage}<span className="text-text-muted text-sm">%</span>
        </p>
      </div>
    </div>
  );
}
