import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { requireUserId } from '@/lib/session';

export const dynamic = 'force-dynamic';

export async function GET(
  _request: NextRequest,
  { params }: { params: { projectId: string } }
) {
  try {
    const userId = await requireUserId();
    const { projectId } = params;

    const project = await prisma.project.findUnique({
      where: { id: projectId },
    });

    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const progress = await prisma.projectProgress.findUnique({
      where: { userId_projectId: { userId, projectId } },
    });

    return NextResponse.json({
      ...project,
      requirements: JSON.parse(project.requirements),
      stretchGoals: JSON.parse(project.stretchGoals),
      steps: JSON.parse(project.steps),
      rubric: JSON.parse(project.rubric),
      status: progress?.status || 'NOT_STARTED',
      completedAt: progress?.completedAt?.toISOString() || null,
    });
  } catch (error) {
    console.error('Project detail API error:', error);
    return NextResponse.json({ error: 'Failed to load project' }, { status: 500 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { projectId: string } }
) {
  try {
    const userId = await requireUserId();
    const { projectId } = params;
    const { status } = await request.json();

    if (!['NOT_STARTED', 'IN_PROGRESS', 'COMPLETED'].includes(status)) {
      return NextResponse.json({ error: 'Invalid status' }, { status: 400 });
    }

    const progress = await prisma.projectProgress.upsert({
      where: { userId_projectId: { userId, projectId } },
      update: {
        status,
        completedAt: status === 'COMPLETED' ? new Date() : null,
      },
      create: {
        userId,
        projectId,
        status,
        completedAt: status === 'COMPLETED' ? new Date() : null,
      },
    });

    // Log activity
    if (status === 'IN_PROGRESS' || status === 'COMPLETED') {
      const project = await prisma.project.findUnique({ where: { id: projectId } });
      await prisma.activityLog.create({
        data: {
          userId,
          type: status === 'COMPLETED' ? 'PROJECT_COMPLETE' : 'PROJECT_START',
          metadata: JSON.stringify({ projectId, projectTitle: project?.title }),
        },
      });
    }

    return NextResponse.json(progress);
  } catch (error) {
    console.error('Project update error:', error);
    return NextResponse.json({ error: 'Failed to update project' }, { status: 500 });
  }
}
