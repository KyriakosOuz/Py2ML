import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { getOrCreateSession } from '@/lib/session';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const sessionId = await getOrCreateSession();

    const projects = await prisma.project.findMany({
      orderBy: { order: 'asc' },
    });

    const progress = await prisma.projectProgress.findMany({
      where: { sessionId },
    });

    const progressMap = new Map(progress.map((p: { projectId: string; status: string }) => [p.projectId, p.status]));

    const result = projects.map((p) => ({
      id: p.id,
      title: p.title,
      slug: p.slug,
      stage: p.stage,
      brief: p.brief,
      requirements: JSON.parse(p.requirements),
      stretchGoals: JSON.parse(p.stretchGoals),
      steps: JSON.parse(p.steps),
      rubric: JSON.parse(p.rubric),
      solutionUrl: p.solutionUrl,
      order: p.order,
      status: progressMap.get(p.id) || 'NOT_STARTED',
    }));

    return NextResponse.json(result);
  } catch (error) {
    console.error('Projects API error:', error);
    return NextResponse.json({ error: 'Failed to load projects' }, { status: 500 });
  }
}
