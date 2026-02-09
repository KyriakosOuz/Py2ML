'use client';

import ProgressBar from '@/components/ui/ProgressBar';

interface StageProgressProps {
  stages: {
    stageId: string;
    stageTitle: string;
    total: number;
    completed: number;
  }[];
}

export default function StageProgress({ stages }: StageProgressProps) {
  return (
    <div className="bg-surface border border-border rounded-xl p-6">
      <h3 className="font-serif text-lg text-text-primary mb-4">Stage Progress</h3>
      <div className="space-y-4">
        {stages.map((stage) => (
          <div key={stage.stageId}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-text-primary">{stage.stageTitle}</span>
              <span className="text-xs font-mono text-text-muted">
                {stage.completed}/{stage.total}
              </span>
            </div>
            <ProgressBar
              value={stage.completed}
              max={stage.total}
              size="sm"
              color={stage.completed === stage.total ? 'success' : 'primary'}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
