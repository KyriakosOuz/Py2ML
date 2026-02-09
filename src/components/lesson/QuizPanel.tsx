'use client';

import { useState } from 'react';
import { CheckCircle2, XCircle, HelpCircle } from 'lucide-react';
import Button from '@/components/ui/Button';

interface QuizQuestion {
  id: string;
  question: string;
  type: string;
  options: string[];
  correctAnswer: string;
  explanation: string;
}

interface QuizPanelProps {
  lessonId: string;
  questions: QuizQuestion[];
  previousAttempt?: { score: number; total: number; answers: Record<string, string> } | null;
}

export default function QuizPanel({ lessonId, questions, previousAttempt }: QuizPanelProps) {
  const [answers, setAnswers] = useState<Record<string, string>>(previousAttempt?.answers ?? {});
  const [submitted, setSubmitted] = useState(!!previousAttempt);
  const [submitting, setSubmitting] = useState(false);
  const [results, setResults] = useState<Record<string, { correct: boolean; correctAnswer: string; explanation: string }> | null>(null);
  const [score, setScore] = useState<{ score: number; total: number } | null>(
    previousAttempt ? { score: previousAttempt.score, total: previousAttempt.total } : null
  );

  const handleAnswer = (questionId: string, answer: string) => {
    if (submitted) return;
    setAnswers((prev) => ({ ...prev, [questionId]: answer }));
  };

  const submitQuiz = async () => {
    setSubmitting(true);
    try {
      const res = await fetch('/api/submit-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lessonId, answers }),
      });
      const data = await res.json();
      setSubmitted(true);
      setScore({ score: data.score, total: data.total });
      const resultMap: Record<string, { correct: boolean; correctAnswer: string; explanation: string }> = {};
      for (const r of data.results) {
        resultMap[r.questionId] = {
          correct: r.correct,
          correctAnswer: r.correctAnswer,
          explanation: r.explanation,
        };
      }
      setResults(resultMap);
    } catch {
      // handle error silently
    } finally {
      setSubmitting(false);
    }
  };

  const retake = () => {
    setAnswers({});
    setSubmitted(false);
    setResults(null);
    setScore(null);
  };

  return (
    <div className="bg-surface border border-border rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <HelpCircle size={20} className="text-primary" />
          <h3 className="font-serif text-lg text-text-primary">Knowledge Check</h3>
        </div>
        {score && (
          <div className={`font-mono text-sm px-3 py-1 rounded-full ${
            score.score === score.total ? 'bg-success/20 text-success' : 'bg-primary/20 text-primary'
          }`}>
            {score.score}/{score.total}
          </div>
        )}
      </div>

      <div className="space-y-6">
        {questions.map((q, qIndex) => {
          const result = results?.[q.id];
          const selected = answers[q.id];

          return (
            <div key={q.id} className="space-y-3">
              <p className="text-sm text-text-primary font-medium">
                {qIndex + 1}. {q.question}
              </p>
              <div className="space-y-2">
                {q.options.map((option) => {
                  const isSelected = selected === option;
                  const isCorrect = result?.correctAnswer === option;
                  let optionClass = 'border-border hover:border-primary/50 hover:bg-surface-light';

                  if (submitted && result) {
                    if (isCorrect) {
                      optionClass = 'border-success bg-success/10';
                    } else if (isSelected && !result.correct) {
                      optionClass = 'border-error bg-error/10';
                    } else {
                      optionClass = 'border-border opacity-60';
                    }
                  } else if (isSelected) {
                    optionClass = 'border-primary bg-primary/10';
                  }

                  return (
                    <button
                      key={option}
                      onClick={() => handleAnswer(q.id, option)}
                      disabled={submitted}
                      className={`w-full text-left px-4 py-2.5 rounded-lg border text-sm transition-colors flex items-center gap-3 ${optionClass}`}
                    >
                      <div className={`w-4 h-4 rounded-full border-2 flex-shrink-0 flex items-center justify-center ${
                        isSelected ? 'border-primary' : 'border-text-muted'
                      }`}>
                        {isSelected && <div className="w-2 h-2 rounded-full bg-primary" />}
                      </div>
                      <span className="text-text-primary">{option}</span>
                      {submitted && isCorrect && <CheckCircle2 size={14} className="text-success ml-auto" />}
                      {submitted && isSelected && !isCorrect && result && <XCircle size={14} className="text-error ml-auto" />}
                    </button>
                  );
                })}
              </div>
              {submitted && result && (
                <div className={`text-xs p-3 rounded-lg ${result.correct ? 'bg-success/5 text-success' : 'bg-surface-light text-text-secondary'}`}>
                  {result.explanation}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-6 flex gap-3">
        {!submitted ? (
          <Button
            onClick={submitQuiz}
            loading={submitting}
            disabled={Object.keys(answers).length < questions.length}
          >
            Submit Quiz
          </Button>
        ) : (
          <Button variant="secondary" onClick={retake}>
            Retake Quiz
          </Button>
        )}
      </div>
    </div>
  );
}
