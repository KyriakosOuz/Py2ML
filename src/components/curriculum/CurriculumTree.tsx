'use client';

import { useState } from 'react';
import StageCard from './StageCard';
import type { CurriculumStage } from '@/types';

interface CurriculumTreeProps {
  stages: CurriculumStage[];
}

export default function CurriculumTree({ stages }: CurriculumTreeProps) {
  const [expandedStages, setExpandedStages] = useState<Set<string>>(
    new Set(stages.length > 0 ? [stages[0].id] : [])
  );

  const toggleStage = (id: string) => {
    setExpandedStages((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-4">
      {stages.map((stage) => (
        <StageCard
          key={stage.id}
          stage={stage}
          expanded={expandedStages.has(stage.id)}
          onToggle={() => toggleStage(stage.id)}
        />
      ))}
    </div>
  );
}
