import { CODE_RUNNER } from './constants';
import type { CodeRunResult } from '@/types';

const PISTON_URL = 'https://emkc.org/api/v2/piston/execute';

const BLOCKED_IMPORTS = [
  'os', 'sys', 'subprocess', 'shutil', 'socket', 'http', 'urllib',
  'ftplib', 'smtplib', 'telnetlib', 'xmlrpc', 'ctypes', 'multiprocessing',
  'threading', 'signal', 'pathlib', 'glob', 'tempfile', 'pickle',
  'shelve', 'code', 'codeop', 'compile', 'compileall',
];

const ALLOWED_IMPORTS = [
  'math', 'random', 'string', 'collections', 'itertools', 'functools',
  'json', 're', 'datetime', 'typing', 'decimal', 'fractions',
  'statistics', 'operator', 'copy', 'pprint', 'textwrap',
];

function buildSafetyWrapper(): string {
  return `
import builtins as _builtins
_original_import = _builtins.__import__

_BLOCKED = {${BLOCKED_IMPORTS.map(m => `'${m}'`).join(', ')}}
_ALLOWED = {${ALLOWED_IMPORTS.map(m => `'${m}'`).join(', ')}}

def _safe_import(name, *args, **kwargs):
    top_level = name.split('.')[0]
    if top_level in _BLOCKED:
        raise ImportError(f"Import '{name}' is not allowed for security reasons")
    return _original_import(name, *args, **kwargs)

_builtins.__import__ = _safe_import

_blocked_builtins = {'exec', 'eval', 'compile', '__import__', 'open', 'input'}
for _name in _blocked_builtins:
    if _name != '__import__':
        if hasattr(_builtins, _name):
            delattr(_builtins, _name)

# --- User code starts below ---
`;
}

export async function runCode(code: string): Promise<CodeRunResult> {
  const wrappedCode = buildSafetyWrapper() + code;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), CODE_RUNNER.TIMEOUT_MS + 5000);

    const response = await fetch(PISTON_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        language: 'python',
        version: '3.10.0',
        files: [{ name: 'main.py', content: wrappedCode }],
        run_timeout: CODE_RUNNER.TIMEOUT_MS,
        compile_memory_limit: -1,
        run_memory_limit: 256000000, // 256MB
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      return {
        stdout: '',
        stderr: `Code execution service error (${response.status})`,
        exitCode: 1,
        timedOut: false,
      };
    }

    const data = await response.json();
    const run = data.run;

    if (!run) {
      return {
        stdout: '',
        stderr: 'No execution result returned',
        exitCode: 1,
        timedOut: false,
      };
    }

    const timedOut = run.signal === 'SIGKILL' || (run.stderr && run.stderr.includes('timed out'));
    const cleanStderr = cleanErrorOutput(run.stderr || '');

    return {
      stdout: (run.stdout || '').trimEnd(),
      stderr: cleanStderr.trimEnd(),
      exitCode: run.code ?? (timedOut ? 1 : 0),
      timedOut,
    };
  } catch (error: unknown) {
    if (error instanceof Error && error.name === 'AbortError') {
      return {
        stdout: '',
        stderr: 'Execution timed out',
        exitCode: 1,
        timedOut: true,
      };
    }
    return {
      stdout: '',
      stderr: error instanceof Error ? error.message : 'Unknown execution error',
      exitCode: 1,
      timedOut: false,
    };
  }
}

function cleanErrorOutput(stderr: string): string {
  const wrapperLines = buildSafetyWrapper().split('\n').length;
  return stderr.replace(/File ".*?main\.py", line (\d+)/g, (_match, lineNum) => {
    const adjusted = parseInt(lineNum) - wrapperLines;
    if (adjusted > 0) {
      return `File "<code>", line ${adjusted}`;
    }
    return `File "<code>", line ${lineNum}`;
  });
}

export async function runCodeWithTests(
  userCode: string,
  testCode: string
): Promise<CodeRunResult> {
  const combined = userCode + '\n\n# --- Tests ---\n' + testCode;
  return runCode(combined);
}

export function validateOutput(
  actual: string,
  expected: string
): boolean {
  return actual.trim() === expected.trim();
}
