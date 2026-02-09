'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight, AlertTriangle } from 'lucide-react';
import MarkdownRenderer from '@/components/ui/MarkdownRenderer';

interface LessonContentProps {
  content: string;
  commonMistakes: string;
}

export default function LessonContent({ content, commonMistakes }: LessonContentProps) {
  const [mistakesExpanded, setMistakesExpanded] = useState(false);

  return (
    <div>
      <MarkdownRenderer content={content} />

      {commonMistakes && (
        <div className="mt-8 border border-primary/20 rounded-xl overflow-hidden">
          <button
            onClick={() => setMistakesExpanded(!mistakesExpanded)}
            className="w-full flex items-center gap-3 px-4 py-3 bg-primary/5 hover:bg-primary/10 transition-colors"
          >
            <AlertTriangle size={16} className="text-primary" />
            <span className="text-sm font-medium text-primary">Common Mistakes</span>
            {mistakesExpanded ? (
              <ChevronDown size={16} className="text-primary ml-auto" />
            ) : (
              <ChevronRight size={16} className="text-primary ml-auto" />
            )}
          </button>
          {mistakesExpanded && (
            <div className="p-4">
              <MarkdownRenderer content={commonMistakes} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
