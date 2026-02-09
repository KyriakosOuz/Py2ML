import { NextRequest, NextResponse } from 'next/server';
import { runCode } from '@/lib/code-runner';
import { runCodeSchema } from '@/lib/validators';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const parsed = runCodeSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Invalid input', details: parsed.error.flatten() },
        { status: 400 }
      );
    }

    const { code } = parsed.data;
    const result = await runCode(code);

    if (result.timedOut) {
      return NextResponse.json({
        stdout: result.stdout,
        stderr: 'Execution timed out (5 second limit)',
        error: 'timeout',
      });
    }

    return NextResponse.json({
      stdout: result.stdout,
      stderr: result.stderr,
      error: result.exitCode !== 0 ? 'runtime_error' : null,
    });
  } catch (error) {
    console.error('Run code error:', error);
    return NextResponse.json({ error: 'Failed to run code' }, { status: 500 });
  }
}
