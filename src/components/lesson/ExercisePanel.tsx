'use client';

import { useState, useCallback } from 'react';
import { Play, Send, CheckCircle2, XCircle, Terminal } from 'lucide-react';
import CodeEditor from './CodeEditor';
import HintSystem from './HintSystem';
import Button from '@/components/ui/Button';

interface Exercise {
  id: string;
  prompt: string;
  starterCode: string;
  expectedOutput: string;
  hints: string[];
  passed: boolean;
  order: number;
}

interface ExercisePanelProps {
  exercises: Exercise[];
  onExercisePass?: (exerciseId: string) => void;
}

export default function ExercisePanel({ exercises, onExercisePass }: ExercisePanelProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [codes, setCodes] = useState<Record<string, string>>(
    Object.fromEntries(exercises.map((ex) => [ex.id, ex.starterCode]))
  );
  const [output, setOutput] = useState('');
  const [running, setRunning] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<'pass' | 'fail' | null>(null);
  const [passedExercises, setPassedExercises] = useState<Set<string>>(
    new Set(exercises.filter((ex) => ex.passed).map((ex) => ex.id))
  );
  const [showConfetti, setShowConfetti] = useState(false);

  const activeExercise = exercises[activeIndex];
  const activeId = activeExercise?.id ?? '';
  const currentCode = codes[activeId] ?? activeExercise?.starterCode ?? '';

  const handleCodeChange = useCallback(
    (value: string) => {
      setCodes((prev) => ({ ...prev, [activeId]: value }));
    },
    [activeId]
  );

  if (!activeExercise) return null;

  const runCode = async () => {
    setRunning(true);
    setOutput('');
    setResult(null);
    try {
      const res = await fetch('/api/run-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: currentCode }),
      });
      const data = await res.json();
      setOutput(data.stderr ? `${data.stdout}\n${data.stderr}`.trim() : data.stdout);
    } catch {
      setOutput('Error: Failed to run code');
    } finally {
      setRunning(false);
    }
  };

  const submitCode = async () => {
    setSubmitting(true);
    setOutput('');
    setResult(null);
    try {
      const res = await fetch('/api/submit-exercise', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exerciseId: activeExercise.id, code: currentCode }),
      });
      const data = await res.json();

      if (data.passed) {
        setResult('pass');
        setOutput(data.stdout || 'All tests passed!');
        setPassedExercises((prev) => new Set([...prev, activeExercise.id]));
        setShowConfetti(true);
        setTimeout(() => setShowConfetti(false), 3000);
        onExercisePass?.(activeExercise.id);
      } else {
        setResult('fail');
        const lines = [];
        if (data.stdout) lines.push(`Output: ${data.stdout}`);
        if (data.expected) lines.push(`Expected: ${data.expected}`);
        if (data.stderr) lines.push(`Error: ${data.stderr}`);
        if (data.timedOut) lines.push('Execution timed out (5 second limit)');
        setOutput(lines.join('\n') || 'Output did not match expected result');
      }
    } catch {
      setOutput('Error: Failed to submit');
      setResult('fail');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden relative">
      {/* Confetti */}
      {showConfetti && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden z-10">
          {Array.from({ length: 30 }).map((_, i) => (
            <div
              key={i}
              className="confetti-piece absolute w-2 h-2 rounded-sm"
              style={{
                left: `${Math.random() * 100}%`,
                backgroundColor: ['#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ef4444'][i % 5],
                animationDelay: `${Math.random() * 0.5}s`,
                animationDuration: `${2 + Math.random() * 2}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* Exercise tabs */}
      <div className="flex border-b border-border">
        {exercises.map((ex, i) => (
          <button
            key={ex.id}
            onClick={() => {
              setActiveIndex(i);
              setOutput('');
              setResult(null);
            }}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-sm border-b-2 transition-colors ${
              i === activeIndex
                ? 'border-primary text-primary bg-primary/5'
                : 'border-transparent text-text-muted hover:text-text-secondary'
            }`}
          >
            {passedExercises.has(ex.id) ? (
              <CheckCircle2 size={14} className="text-success" />
            ) : null}
            Exercise {i + 1}
          </button>
        ))}
      </div>

      {/* Exercise content */}
      <div className="p-4 space-y-4">
        {/* Prompt */}
        <div className="bg-surface-light rounded-lg p-3">
          <p className="text-sm text-text-primary">{activeExercise.prompt}</p>
        </div>

        {/* Editor */}
        <CodeEditor value={currentCode} onChange={handleCodeChange} height="200px" />

        {/* Action buttons */}
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={runCode} loading={running} size="sm">
            <Play size={14} className="mr-1" />
            Run
          </Button>
          <Button onClick={submitCode} loading={submitting} size="sm">
            <Send size={14} className="mr-1" />
            Submit
          </Button>
        </div>

        {/* Output */}
        {output && (
          <div
            className={`rounded-lg p-3 font-mono text-sm ${
              result === 'pass'
                ? 'bg-success/10 border border-success/30'
                : result === 'fail'
                ? 'bg-error/10 border border-error/30'
                : 'bg-surface-light border border-border'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              {result === 'pass' ? (
                <CheckCircle2 size={14} className="text-success" />
              ) : result === 'fail' ? (
                <XCircle size={14} className="text-error" />
              ) : (
                <Terminal size={14} className="text-text-muted" />
              )}
              <span className={`text-xs ${result === 'pass' ? 'text-success' : result === 'fail' ? 'text-error' : 'text-text-muted'}`}>
                {result === 'pass' ? 'Passed!' : result === 'fail' ? 'Not quite right' : 'Output'}
              </span>
            </div>
            <pre className="whitespace-pre-wrap text-text-secondary">{output}</pre>
          </div>
        )}

        {/* Hints */}
        <HintSystem hints={activeExercise.hints} />
      </div>
    </div>
  );
}
