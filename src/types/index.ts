export interface CodeRunResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
}

export interface ExerciseResult {
  passed: boolean;
  output: string;
  expected?: string;
  error?: string;
}

export interface QuizResult {
  score: number;
  total: number;
  results: {
    questionId: string;
    correct: boolean;
    selectedAnswer: string;
    correctAnswer: string;
    explanation: string;
  }[];
}

export interface CurriculumStage {
  id: string;
  title: string;
  slug: string;
  description: string;
  order: number;
  modules: CurriculumModule[];
}

export interface CurriculumModule {
  id: string;
  title: string;
  slug: string;
  description: string;
  order: number;
  lessons: CurriculumLesson[];
}

export interface CurriculumLesson {
  id: string;
  title: string;
  slug: string;
  order: number;
  completed: boolean;
}

export interface DashboardData {
  totalLessons: number;
  completedLessons: number;
  totalExercises: number;
  passedExercises: number;
  quizAverage: number;
  stageProgress: {
    stageId: string;
    stageTitle: string;
    total: number;
    completed: number;
  }[];
  earnedSkills: { name: string; slug: string; earnedAt: string }[];
  recentActivity: {
    type: string;
    metadata: Record<string, unknown>;
    createdAt: string;
  }[];
  streak: number;
  nextLesson: { id: string; title: string; slug: string } | null;
}

export interface ProjectData {
  id: string;
  title: string;
  slug: string;
  stage: string;
  brief: string;
  requirements: string[];
  stretchGoals: string[];
  steps: { title: string; description: string }[];
  rubric: { criterion: string; description: string }[];
  solutionUrl: string | null;
  order: number;
  status: string;
}
