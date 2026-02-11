import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const lessons = await prisma.lesson.findMany({
      orderBy: [
        { module: { stage: { order: 'asc' } } },
        { module: { order: 'asc' } },
        { order: 'asc' },
      ],
      include: {
        module: {
          include: { stage: true },
        },
        exercises: { select: { id: true } },
        quizQuestions: { select: { id: true } },
      },
    });

    return NextResponse.json(
      lessons.map((l) => ({
        id: l.id,
        title: l.title,
        slug: l.slug,
        order: l.order,
        stageName: l.module.stage.title,
        moduleName: l.module.title,
        exerciseCount: l.exercises.length,
        quizCount: l.quizQuestions.length,
      }))
    );
  } catch (error) {
    console.error('Admin lessons error:', error);
    return NextResponse.json({ error: 'Failed to load lessons' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { title, slug, moduleId, content, commonMistakes, order } = body;

    const lesson = await prisma.lesson.create({
      data: { title, slug, moduleId, content, commonMistakes, order },
    });

    return NextResponse.json(lesson, { status: 201 });
  } catch (error) {
    console.error('Create lesson error:', error);
    return NextResponse.json({ error: 'Failed to create lesson' }, { status: 500 });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const { id, ...data } = body;

    const lesson = await prisma.lesson.update({
      where: { id },
      data,
    });

    return NextResponse.json(lesson);
  } catch (error) {
    console.error('Update lesson error:', error);
    return NextResponse.json({ error: 'Failed to update lesson' }, { status: 500 });
  }
}
