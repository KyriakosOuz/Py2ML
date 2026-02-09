import { spawn } from 'child_process';
import { writeFileSync, unlinkSync, mkdtempSync, rmdirSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { CODE_RUNNER } from './constants';
import type { CodeRunResult } from '@/types';

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
  'numpy', 'pandas', 'matplotlib', 'sklearn', 'scipy',
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

# Redirect matplotlib to non-interactive backend
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

# --- User code starts below ---
`;
}

export async function runCode(code: string): Promise<CodeRunResult> {
  const dir = mkdtempSync(join(tmpdir(), 'py2ml-'));
  const filePath = join(dir, 'code.py');
  const wrappedCode = buildSafetyWrapper() + code;

  writeFileSync(filePath, wrappedCode, 'utf-8');

  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let settled = false;

    const proc = spawn('python3', [filePath], {
      timeout: CODE_RUNNER.TIMEOUT_MS,
      env: { ...process.env, HOME: '/tmp', PYTHONDONTWRITEBYTECODE: '1' },
      cwd: dir,
    });

    proc.stdout.on('data', (data: Buffer) => {
      if (stdout.length < CODE_RUNNER.MAX_BUFFER) {
        stdout += data.toString();
      }
    });

    proc.stderr.on('data', (data: Buffer) => {
      if (stderr.length < CODE_RUNNER.MAX_BUFFER) {
        stderr += data.toString();
      }
    });

    const timer = setTimeout(() => {
      timedOut = true;
      proc.kill('SIGKILL');
    }, CODE_RUNNER.TIMEOUT_MS);

    proc.on('close', (exitCode) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { unlinkSync(filePath); } catch {}
      try { rmdirSync(dir); } catch {}

      // Clean up stderr to remove safety wrapper line numbers
      const cleanStderr = cleanErrorOutput(stderr);

      resolve({
        stdout: stdout.trimEnd(),
        stderr: cleanStderr.trimEnd(),
        exitCode,
        timedOut,
      });
    });

    proc.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { unlinkSync(filePath); } catch {}
      try { rmdirSync(dir); } catch {}

      resolve({
        stdout: '',
        stderr: err.message,
        exitCode: 1,
        timedOut: false,
      });
    });
  });
}

function cleanErrorOutput(stderr: string): string {
  // Adjust line numbers in tracebacks to account for the safety wrapper
  const wrapperLines = buildSafetyWrapper().split('\n').length;
  return stderr.replace(/File ".*?code\.py", line (\d+)/g, (match, lineNum) => {
    const adjusted = parseInt(lineNum) - wrapperLines;
    if (adjusted > 0) {
      return `File "<code>", line ${adjusted}`;
    }
    return match;
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
