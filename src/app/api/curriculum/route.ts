import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { requireUserId } from '@/lib/session';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const userId = await requireUserId();

    const stages = await prisma.stage.findMany({
      orderBy: { order: 'asc' },
      include: {
        modules: {
          orderBy: { order: 'asc' },
          include: {
            lessons: {
              orderBy: { order: 'asc' },
              select: {
                id: true,
                title: true,
                slug: true,
                order: true,
                exercises: {
                  select: { id: true },
                },
              },
            },
          },
        },
      },
    });

    // Get all passed exercise IDs for this session
    const passedSubmissions = await prisma.submission.findMany({
      where: { userId, passed: true },
      select: { exerciseId: true },
      distinct: ['exerciseId'],
    });
    const passedExerciseIds = new Set(passedSubmissions.map((s: { exerciseId: string }) => s.exerciseId));

    const curriculum = stages.map((stage) => ({
      ...stage,
      modules: stage.modules.map((mod) => ({
        id: mod.id,
        title: mod.title,
        slug: mod.slug,
        description: mod.description,
        order: mod.order,
        lessons: mod.lessons.map((lesson) => {
          const totalExercises = lesson.exercises.length;
          const passedExercises = lesson.exercises.filter(
            (ex: { id: string }) => passedExerciseIds.has(ex.id)
          ).length;
          return {
            id: lesson.id,
            title: lesson.title,
            slug: lesson.slug,
            order: lesson.order,
            completed: totalExercises > 0 && passedExercises === totalExercises,
          };
        }),
      })),
    }));

    return NextResponse.json(curriculum);
  } catch (error) {
    console.error('Curriculum API error:', error);
    return NextResponse.json({ error: 'Failed to load curriculum' }, { status: 500 });
  }
}
