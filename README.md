# Py2ML Academy

A full-stack Next.js application that teaches Python from beginner to AI/ML-ready, with interactive coding exercises, quizzes, projects, and progress tracking.

## Tech Stack

- **Framework:** Next.js 14 (App Router) + TypeScript
- **Styling:** Tailwind CSS (dark theme with amber accents)
- **Database:** Prisma ORM + SQLite
- **Code Editor:** Monaco Editor (@monaco-editor/react)
- **Icons:** Lucide React
- **Validation:** Zod

## Prerequisites

- Node.js 18+
- Python 3.8+ (for code execution)
- npm

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Set up environment
cp .env.example .env

# 3. Run database migration
npx prisma migrate dev

# 4. Seed the database (20 lessons, 60 exercises, 60 quizzes, 6 projects)
npx prisma db seed

# 5. Start the dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Curriculum

### Stage A: Python Foundations (8 lessons)
1. Hello Python: print, comments, REPL
2. Variables & Data Types
3. Strings Deep Dive
4. Lists & Tuples
5. Dictionaries & Sets
6. Control Flow
7. Loops
8. Functions

### Stage B: Python for Data (7 lessons)
9. Modules & Imports
10. File I/O & Exceptions
11. NumPy Basics
12. Pandas Series & DataFrames
13. Pandas Data Wrangling
14. Matplotlib Basics
15. Exploratory Data Analysis

### Stage C: ML/AI Track (5 lessons)
16. Intro to Machine Learning
17. scikit-learn Basics
18. Classification
19. Regression
20. Feature Engineering & Model Selection

### Projects
1. CLI Calculator (Beginner)
2. Text File Analyzer (Beginner)
3. CSV Cleaner + Summary Report (Data)
4. EDA Dashboard (Data)
5. Titanic Survival Classification (ML)
6. House Price Regression (ML)

## Features

- **Interactive Code Editor** — Monaco Editor with Python syntax highlighting
- **Code Execution** — Python code runs in a sandboxed subprocess with timeout protection
- **Exercise Validation** — Output comparison with expected results
- **Progressive Hints** — Reveal hints one at a time when stuck
- **Quiz System** — MCQ and True/False questions with explanations
- **Progress Tracking** — Dashboard with completion stats, streak tracking, and skill badges
- **Project System** — Scaffolded projects with requirements, steps, and rubrics
- **Admin Panel** — Edit lesson content directly from the browser
- **Guest Sessions** — Cookie-based sessions, no auth required

## Code Execution

Python code is executed using a subprocess runner with:
- 5-second timeout
- Blocked dangerous imports (os, sys, subprocess, etc.)
- Allowed safe imports (math, random, numpy, pandas, sklearn, etc.)
- Output capture (stdout + stderr)
- Line number adjustment in error messages

## Project Structure

```
src/
├── app/                    # Next.js pages (App Router)
│   ├── api/                # API routes
│   ├── dashboard/          # Progress dashboard
│   ├── curriculum/         # Curriculum tree view
│   ├── lesson/[lessonId]/  # Lesson viewer + exercises
│   ├── projects/           # Project list + detail
│   ├── portfolio/          # Completed projects
│   └── admin/              # Lesson editor
├── components/             # React components
│   ├── layout/             # Sidebar, Header, MobileNav
│   ├── lesson/             # CodeEditor, ExercisePanel, QuizPanel
│   ├── dashboard/          # Progress, Streak, Skills
│   ├── curriculum/         # CurriculumTree, StageCard
│   ├── projects/           # ProjectCard, ProjectDetail
│   └── ui/                 # Button, Card, Badge, ProgressBar
├── lib/                    # Utilities
│   ├── db.ts               # Prisma client singleton
│   ├── code-runner.ts      # Python sandbox
│   ├── session.ts          # Guest session management
│   ├── validators.ts       # Zod schemas
│   └── constants.ts        # App constants
└── types/                  # TypeScript types
```

## API Routes

| Endpoint | Method | Description |
|---|---|---|
| `/api/curriculum` | GET | Full curriculum tree with completion status |
| `/api/lessons/[id]` | GET | Lesson content, exercises, quiz |
| `/api/run-code` | POST | Execute Python code |
| `/api/submit-exercise` | POST | Validate exercise submission |
| `/api/submit-quiz` | POST | Grade quiz answers |
| `/api/dashboard` | GET | Progress stats for session |
| `/api/projects` | GET | All projects with status |
| `/api/projects/[id]` | GET/PUT | Project detail + status update |
| `/api/admin/lessons` | GET/POST/PUT | Lesson CRUD |
