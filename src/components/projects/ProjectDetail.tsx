'use client';

import { useState } from 'react';
import { CheckCircle2, Circle, ChevronDown, ChevronRight, Target, Trophy, ListChecks } from 'lucide-react';
import CodeEditor from '@/components/lesson/CodeEditor';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';
import type { ProjectData } from '@/types';

interface ProjectDetailProps {
  project: ProjectData;
  onStatusChange: (status: string) => void;
}

export default function ProjectDetail({ project, onStatusChange }: ProjectDetailProps) {
  const [code, setCode] = useState('# Start coding your project here\n');
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([0]));
  const [output, setOutput] = useState('');
  const [running, setRunning] = useState(false);

  const toggleStep = (index: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const runCode = async () => {
    setRunning(true);
    try {
      const res = await fetch('/api/run-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      const data = await res.json();
      setOutput(data.stderr ? `${data.stdout}\n${data.stderr}`.trim() : data.stdout);
    } catch {
      setOutput('Error running code');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      {/* Left: Project info */}
      <div className="space-y-6">
        <div>
          <h1 className="font-serif text-2xl text-text-primary mb-2">{project.title}</h1>
          <p className="text-text-secondary">{project.brief}</p>
        </div>

        {/* Requirements */}
        <div className="bg-surface border border-border rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Target size={18} className="text-primary" />
            <h3 className="font-medium text-text-primary">Requirements</h3>
          </div>
          <ul className="space-y-2">
            {project.requirements.map((req, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                <CheckCircle2 size={14} className="text-text-muted mt-0.5 flex-shrink-0" />
                {req}
              </li>
            ))}
          </ul>
        </div>

        {/* Steps */}
        <div className="bg-surface border border-border rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <ListChecks size={18} className="text-primary" />
            <h3 className="font-medium text-text-primary">Steps</h3>
          </div>
          <div className="space-y-2">
            {project.steps.map((step, i) => (
              <div key={i} className="border border-border rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleStep(i)}
                  className="w-full flex items-center gap-3 p-3 hover:bg-surface-light transition-colors"
                >
                  <span className="text-xs font-mono text-text-muted w-6">{i + 1}.</span>
                  <span className="text-sm text-text-primary flex-1 text-left">{step.title}</span>
                  {expandedSteps.has(i) ? <ChevronDown size={14} className="text-text-muted" /> : <ChevronRight size={14} className="text-text-muted" />}
                </button>
                {expandedSteps.has(i) && (
                  <div className="px-3 pb-3 pl-12">
                    <p className="text-sm text-text-secondary">{step.description}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Stretch goals */}
        {project.stretchGoals.length > 0 && (
          <div className="bg-surface border border-border rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Trophy size={18} className="text-primary" />
              <h3 className="font-medium text-text-primary">Stretch Goals</h3>
            </div>
            <ul className="space-y-2">
              {project.stretchGoals.map((goal, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                  <Circle size={14} className="text-text-muted mt-0.5 flex-shrink-0" />
                  {goal}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Status actions */}
        <div className="flex items-center gap-3">
          {project.status === 'NOT_STARTED' && (
            <Button onClick={() => onStatusChange('IN_PROGRESS')}>Start Project</Button>
          )}
          {project.status === 'IN_PROGRESS' && (
            <Button onClick={() => onStatusChange('COMPLETED')} variant="primary">
              Mark Complete
            </Button>
          )}
          {project.status === 'COMPLETED' && (
            <Badge variant="success" size="md">Completed</Badge>
          )}
        </div>
      </div>

      {/* Right: Code editor */}
      <div className="space-y-4 lg:sticky lg:top-20 lg:self-start">
        <h3 className="font-medium text-text-primary">Code Workspace</h3>
        <CodeEditor value={code} onChange={setCode} height="400px" />
        <Button onClick={runCode} loading={running} variant="secondary" size="sm">
          Run Code
        </Button>
        {output && (
          <div className="bg-surface-light border border-border rounded-lg p-3 font-mono text-sm">
            <pre className="whitespace-pre-wrap text-text-secondary">{output}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
