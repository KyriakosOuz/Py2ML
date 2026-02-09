'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import LessonContent from '@/components/lesson/LessonContent';
import ExercisePanel from '@/components/lesson/ExercisePanel';
import QuizPanel from '@/components/lesson/QuizPanel';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';

interface LessonData {
  id: string;
  title: string;
  slug: string;
  content: string;
  commonMistakes: string;
  exercises: {
    id: string;
    prompt: string;
    starterCode: string;
    expectedOutput: string;
    hints: string[];
    passed: boolean;
    order: number;
  }[];
  quizQuestions: {
    id: string;
    question: string;
    type: string;
    options: string[];
    correctAnswer: string;
    explanation: string;
  }[];
  quizAttempt: { score: number; total: number; answers: Record<string, string> } | null;
  stage: { title: string; slug: string };
  module: { title: string; slug: string };
  prevLesson: { id: string; title: string } | null;
  nextLesson: { id: string; title: string } | null;
}

export default function LessonPage() {
  const params = useParams();
  const lessonId = params.lessonId as string;
  const [lesson, setLesson] = useState<LessonData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!lessonId) return;
    setLoading(true);
    fetch(`/api/lessons/${lessonId}`)
      .then((res) => res.json())
      .then((data) => {
        if (!data.error) setLesson(data);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [lessonId]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-surface rounded w-64 mb-4" />
          <div className="h-4 bg-surface rounded w-96 mb-8" />
          <div className="grid lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3 space-y-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-24 bg-surface rounded-xl" />
              ))}
            </div>
            <div className="lg:col-span-2">
              <div className="h-96 bg-surface rounded-xl" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-16 text-center">
        <h1 className="font-serif text-2xl text-text-primary mb-4">Lesson Not Found</h1>
        <p className="text-text-secondary mb-6">This lesson doesn&apos;t exist or may have been removed.</p>
        <Link href="/curriculum">
          <Button variant="secondary">Back to Curriculum</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <Badge variant="outline">{lesson.stage.title}</Badge>
          <span className="text-text-muted text-sm">/</span>
          <Badge variant="outline">{lesson.module.title}</Badge>
        </div>
        <h1 className="font-serif text-2xl md:text-3xl text-text-primary">{lesson.title}</h1>
      </div>

      {/* Split view */}
      <div className="grid lg:grid-cols-5 gap-6">
        {/* Left: Content */}
        <div className="lg:col-span-3 space-y-8">
          <LessonContent content={lesson.content} commonMistakes={lesson.commonMistakes} />

          {/* Quiz */}
          {lesson.quizQuestions.length > 0 && (
            <QuizPanel
              lessonId={lesson.id}
              questions={lesson.quizQuestions}
              previousAttempt={lesson.quizAttempt}
            />
          )}

          {/* Navigation */}
          <div className="flex items-center justify-between pt-4 border-t border-border">
            {lesson.prevLesson ? (
              <Link href={`/lesson/${lesson.prevLesson.id}`}>
                <Button variant="ghost" size="sm">
                  <ArrowLeft size={14} className="mr-1" />
                  {lesson.prevLesson.title}
                </Button>
              </Link>
            ) : <div />}
            {lesson.nextLesson ? (
              <Link href={`/lesson/${lesson.nextLesson.id}`}>
                <Button variant="ghost" size="sm">
                  {lesson.nextLesson.title}
                  <ArrowRight size={14} className="ml-1" />
                </Button>
              </Link>
            ) : <div />}
          </div>
        </div>

        {/* Right: Exercises */}
        <div className="lg:col-span-2">
          <div className="lg:sticky lg:top-20">
            {lesson.exercises.length > 0 && (
              <ExercisePanel exercises={lesson.exercises} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
