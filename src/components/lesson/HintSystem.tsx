'use client';

import { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronRight, Eye } from 'lucide-react';
import Button from '@/components/ui/Button';

interface HintSystemProps {
  hints: string[];
  solution?: string;
}

export default function HintSystem({ hints, solution }: HintSystemProps) {
  const [revealedCount, setRevealedCount] = useState(0);
  const [showSolution, setShowSolution] = useState(false);

  const revealNext = () => {
    if (revealedCount < hints.length) {
      setRevealedCount((c) => c + 1);
    }
  };

  return (
    <div className="space-y-2">
      {hints.slice(0, revealedCount).map((hint, i) => (
        <div
          key={i}
          className="flex items-start gap-2 bg-primary/5 border border-primary/20 rounded-lg px-3 py-2"
        >
          <Lightbulb size={14} className="text-primary mt-0.5 flex-shrink-0" />
          <p className="text-sm text-text-secondary">{hint}</p>
        </div>
      ))}

      <div className="flex items-center gap-2">
        {revealedCount < hints.length && (
          <Button variant="ghost" size="sm" onClick={revealNext}>
            <Lightbulb size={14} className="mr-1" />
            Show Hint {revealedCount + 1}
            {revealedCount < hints.length - 1 ? (
              <ChevronRight size={14} className="ml-1" />
            ) : (
              <ChevronDown size={14} className="ml-1" />
            )}
          </Button>
        )}
        {revealedCount >= hints.length && solution && !showSolution && (
          <Button variant="ghost" size="sm" onClick={() => setShowSolution(true)}>
            <Eye size={14} className="mr-1" />
            Show Solution
          </Button>
        )}
      </div>

      {showSolution && solution && (
        <div className="bg-surface-light border border-border rounded-lg p-3">
          <p className="text-xs text-text-muted mb-2 uppercase tracking-wider">Solution</p>
          <pre className="text-sm font-mono text-text-primary whitespace-pre-wrap">{solution}</pre>
        </div>
      )}
    </div>
  );
}
