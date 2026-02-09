import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { getOrCreateSession } from '@/lib/session';

export async function GET(
  _request: NextRequest,
  { params }: { params: { lessonId: string } }
) {
  try {
    const sessionId = await getOrCreateSession();
    const { lessonId } = params;

    const lesson = await prisma.lesson.findUnique({
      where: { id: lessonId },
      include: {
        exercises: {
          orderBy: { order: 'asc' },
        },
        quizQuestions: {
          orderBy: { order: 'asc' },
        },
        module: {
          include: {
            stage: true,
            lessons: {
              orderBy: { order: 'asc' },
              select: { id: true, title: true, slug: true, order: true },
            },
          },
        },
      },
    });

    if (!lesson) {
      return NextResponse.json({ error: 'Lesson not found' }, { status: 404 });
    }

    // Get submissions for this session's exercises
    const submissions = await prisma.submission.findMany({
      where: {
        sessionId,
        exerciseId: { in: lesson.exercises.map((e) => e.id) },
      },
      orderBy: { createdAt: 'desc' },
    });

    const passedExerciseIds = new Set(
      submissions.filter((s: { passed: boolean }) => s.passed).map((s: { exerciseId: string }) => s.exerciseId)
    );

    // Get quiz attempts
    const quizAttempt = await prisma.quizAttempt.findFirst({
      where: { sessionId, lessonId },
      orderBy: { createdAt: 'desc' },
    });

    // Log activity
    await prisma.activityLog.create({
      data: {
        sessionId,
        type: 'LESSON_VIEW',
        metadata: JSON.stringify({ lessonId, lessonTitle: lesson.title }),
      },
    });

    // Find next and previous lessons
    const allLessons = lesson.module.lessons;
    const currentIndex = allLessons.findIndex((l) => l.id === lessonId);
    const prevLesson = currentIndex > 0 ? allLessons[currentIndex - 1] : null;
    const nextLesson = currentIndex < allLessons.length - 1 ? allLessons[currentIndex + 1] : null;

    return NextResponse.json({
      ...lesson,
      exercises: lesson.exercises.map((ex) => ({
        ...ex,
        hints: JSON.parse(ex.hints),
        passed: passedExerciseIds.has(ex.id),
      })),
      quizQuestions: lesson.quizQuestions.map((q) => ({
        ...q,
        options: JSON.parse(q.options),
      })),
      quizAttempt: quizAttempt
        ? { score: quizAttempt.score, total: quizAttempt.total, answers: JSON.parse(quizAttempt.answers) }
        : null,
      stage: lesson.module.stage,
      prevLesson,
      nextLesson,
    });
  } catch (error) {
    console.error('Lesson API error:', error);
    return NextResponse.json({ error: 'Failed to load lesson' }, { status: 500 });
  }
}
