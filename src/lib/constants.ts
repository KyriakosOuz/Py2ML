export const STAGES = {
  PYTHON_FOUNDATIONS: 'python-foundations',
  PYTHON_FOR_DATA: 'python-for-data',
  ML_AI_TRACK: 'ml-ai-track',
} as const;

export const PROJECT_STAGES = {
  BEGINNER: 'BEGINNER',
  DATA: 'DATA',
  ML: 'ML',
} as const;

export const QUIZ_TYPES = {
  MCQ: 'MCQ',
  TRUE_FALSE: 'TRUE_FALSE',
} as const;

export const ACTIVITY_TYPES = {
  LESSON_VIEW: 'LESSON_VIEW',
  EXERCISE_PASS: 'EXERCISE_PASS',
  QUIZ_COMPLETE: 'QUIZ_COMPLETE',
  PROJECT_START: 'PROJECT_START',
  PROJECT_COMPLETE: 'PROJECT_COMPLETE',
} as const;

export const PROJECT_STATUS = {
  NOT_STARTED: 'NOT_STARTED',
  IN_PROGRESS: 'IN_PROGRESS',
  COMPLETED: 'COMPLETED',
} as const;

export const CODE_RUNNER = {
  TIMEOUT_MS: 5000,
  MAX_BUFFER: 1024 * 1024, // 1MB
} as const;
