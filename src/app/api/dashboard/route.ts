import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { requireUserId } from '@/lib/session';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const userId = await requireUserId();

    // Get all lessons count
    const totalLessons = await prisma.lesson.count();

    // Get all exercises and passed exercises
    const totalExercises = await prisma.exercise.count();
    const passedSubmissions = await prisma.submission.findMany({
      where: { userId, passed: true },
      select: { exerciseId: true },
      distinct: ['exerciseId'],
    });
    const passedExercises = passedSubmissions.length;

    // Get quiz average
    const quizAttempts = await prisma.quizAttempt.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
    });
    // Get best attempt per lesson
    const bestAttempts = new Map<string, { score: number; total: number }>();
    for (const attempt of quizAttempts) {
      const existing = bestAttempts.get(attempt.lessonId);
      if (!existing || attempt.score / attempt.total > existing.score / existing.total) {
        bestAttempts.set(attempt.lessonId, { score: attempt.score, total: attempt.total });
      }
    }
    const quizAverage = bestAttempts.size > 0
      ? Array.from(bestAttempts.values()).reduce((sum, a) => sum + (a.score / a.total) * 100, 0) / bestAttempts.size
      : 0;

    // Get completed lessons (all exercises passed)
    const lessons = await prisma.lesson.findMany({
      include: { exercises: { select: { id: true } } },
    });
    const passedExerciseIds = new Set(passedSubmissions.map((s: { exerciseId: string }) => s.exerciseId));
    const completedLessonIds = lessons
      .filter((l) => l.exercises.length > 0 && l.exercises.every((e) => passedExerciseIds.has(e.id)))
      .map((l) => l.id);

    // Stage progress
    const stages = await prisma.stage.findMany({
      orderBy: { order: 'asc' },
      include: {
        modules: {
          include: {
            lessons: {
              select: { id: true },
            },
          },
        },
      },
    });

    const stageProgress = stages.map((stage) => {
      const stageLessonIds = stage.modules.flatMap((m) => m.lessons.map((l) => l.id));
      return {
        stageId: stage.id,
        stageTitle: stage.title,
        total: stageLessonIds.length,
        completed: stageLessonIds.filter((id) => completedLessonIds.includes(id)).length,
      };
    });

    // Earned skills
    const earnedSkills = await prisma.earnedSkill.findMany({
      where: { userId },
      include: { skillTag: true },
      orderBy: { earnedAt: 'desc' },
    });

    // Recent activity
    const recentActivity = await prisma.activityLog.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: 10,
    });

    // Calculate streak (consecutive days with activity)
    const allActivity = await prisma.activityLog.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      select: { createdAt: true },
    });

    let streak = 0;
    if (allActivity.length > 0) {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const activityDays = new Set(
        allActivity.map((a: { createdAt: Date }) => {
          const d = new Date(a.createdAt);
          d.setHours(0, 0, 0, 0);
          return d.getTime();
        })
      );

      let checkDate = today.getTime();
      while (activityDays.has(checkDate)) {
        streak++;
        checkDate -= 86400000; // subtract one day
      }
    }

    // Next lesson recommendation
    const allLessonsOrdered = await prisma.lesson.findMany({
      orderBy: [
        { module: { stage: { order: 'asc' } } },
        { module: { order: 'asc' } },
        { order: 'asc' },
      ],
      select: { id: true, title: true, slug: true },
    });
    const nextLesson = allLessonsOrdered.find((l) => !completedLessonIds.includes(l.id)) ?? null;

    return NextResponse.json({
      totalLessons,
      completedLessons: completedLessonIds.length,
      totalExercises,
      passedExercises,
      quizAverage: Math.round(quizAverage),
      stageProgress,
      earnedSkills: earnedSkills.map((es) => ({
        name: es.skillTag.name,
        slug: es.skillTag.slug,
        earnedAt: es.earnedAt.toISOString(),
      })),
      recentActivity: recentActivity.map((a) => ({
        type: a.type,
        metadata: JSON.parse(a.metadata),
        createdAt: a.createdAt.toISOString(),
      })),
      streak,
      nextLesson,
    });
  } catch (error) {
    console.error('Dashboard API error:', error);
    return NextResponse.json({ error: 'Failed to load dashboard' }, { status: 500 });
  }
}
