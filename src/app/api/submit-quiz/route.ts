import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { getOrCreateSession } from '@/lib/session';
import { submitQuizSchema } from '@/lib/validators';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const sessionId = await getOrCreateSession();
    const body = await request.json();
    const parsed = submitQuizSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Invalid input', details: parsed.error.flatten() },
        { status: 400 }
      );
    }

    const { lessonId, answers } = parsed.data;

    const questions = await prisma.quizQuestion.findMany({
      where: { lessonId },
      orderBy: { order: 'asc' },
    });

    if (questions.length === 0) {
      return NextResponse.json({ error: 'No quiz questions found' }, { status: 404 });
    }

    const results = questions.map((q) => {
      const selectedAnswer = answers[q.id] || '';
      const correct = selectedAnswer === q.correctAnswer;
      return {
        questionId: q.id,
        correct,
        selectedAnswer,
        correctAnswer: q.correctAnswer,
        explanation: q.explanation,
      };
    });

    const score = results.filter((r) => r.correct).length;
    const total = questions.length;

    // Save quiz attempt
    await prisma.quizAttempt.create({
      data: {
        sessionId,
        lessonId,
        answers: JSON.stringify(answers),
        score,
        total,
      },
    });

    // Log activity
    const lesson = await prisma.lesson.findUnique({ where: { id: lessonId } });
    await prisma.activityLog.create({
      data: {
        sessionId,
        type: 'QUIZ_COMPLETE',
        metadata: JSON.stringify({
          lessonId,
          lessonTitle: lesson?.title,
          score,
          total,
        }),
      },
    });

    return NextResponse.json({ score, total, results });
  } catch (error) {
    console.error('Submit quiz error:', error);
    return NextResponse.json({ error: 'Failed to submit quiz' }, { status: 500 });
  }
}
