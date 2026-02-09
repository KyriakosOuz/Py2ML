import { z } from 'zod';

export const runCodeSchema = z.object({
  code: z.string().min(1).max(10000),
  exerciseId: z.string().optional(),
});

export const submitExerciseSchema = z.object({
  exerciseId: z.string().min(1),
  code: z.string().min(1).max(10000),
});

export const submitQuizSchema = z.object({
  lessonId: z.string().min(1),
  answers: z.record(z.string(), z.string()),
});

export const lessonUpdateSchema = z.object({
  title: z.string().min(1).optional(),
  content: z.string().optional(),
  commonMistakes: z.string().optional(),
  order: z.number().int().optional(),
});

export const projectProgressSchema = z.object({
  projectId: z.string().min(1),
  status: z.enum(['NOT_STARTED', 'IN_PROGRESS', 'COMPLETED']),
});
