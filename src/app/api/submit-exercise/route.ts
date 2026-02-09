import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { getOrCreateSession } from '@/lib/session';
import { runCode, runCodeWithTests, validateOutput } from '@/lib/code-runner';
import { submitExerciseSchema } from '@/lib/validators';

export async function POST(request: NextRequest) {
  try {
    const sessionId = await getOrCreateSession();
    const body = await request.json();
    const parsed = submitExerciseSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Invalid input', details: parsed.error.flatten() },
        { status: 400 }
      );
    }

    const { exerciseId, code } = parsed.data;

    const exercise = await prisma.exercise.findUnique({
      where: { id: exerciseId },
      include: { lesson: true },
    });

    if (!exercise) {
      return NextResponse.json({ error: 'Exercise not found' }, { status: 404 });
    }

    let result;
    let passed = false;

    if (exercise.testCode && exercise.testCode.trim()) {
      // Run with test code
      result = await runCodeWithTests(code, exercise.testCode);
      passed = result.exitCode === 0 && !result.timedOut;
    } else {
      // Run and compare output
      result = await runCode(code);
      passed = !result.timedOut && result.exitCode === 0 &&
        validateOutput(result.stdout, exercise.expectedOutput);
    }

    // Save submission
    await prisma.submission.create({
      data: {
        sessionId,
        exerciseId,
        code,
        passed,
        output: result.stdout + (result.stderr ? '\n' + result.stderr : ''),
      },
    });

    // Log activity if passed
    if (passed) {
      await prisma.activityLog.create({
        data: {
          sessionId,
          type: 'EXERCISE_PASS',
          metadata: JSON.stringify({
            exerciseId,
            lessonId: exercise.lessonId,
            lessonTitle: exercise.lesson.title,
          }),
        },
      });
    }

    return NextResponse.json({
      passed,
      stdout: result.stdout,
      stderr: result.stderr,
      expected: exercise.expectedOutput,
      timedOut: result.timedOut,
    });
  } catch (error) {
    console.error('Submit exercise error:', error);
    return NextResponse.json({ error: 'Failed to submit exercise' }, { status: 500 });
  }
}
