async function seedModule7(prisma: any) {
  console.log('Seeding Module 7: Professional Python...');

  // ─── Skill Tags ────────────────────────────────────────────────────
  const newSkills = [
    { id: 'skill-026', name: 'Comprehensions', slug: 'comprehensions' },
    { id: 'skill-027', name: 'Generators', slug: 'generators' },
    { id: 'skill-028', name: 'Type Hints', slug: 'type-hints' },
    { id: 'skill-029', name: 'Dataclasses', slug: 'dataclasses' },
    { id: 'skill-030', name: 'API Integration', slug: 'api-integration' },
    { id: 'skill-031', name: 'Git', slug: 'git' },
    { id: 'skill-032', name: 'Project Structure', slug: 'project-structure' },
  ];
  for (const s of newSkills) {
    await prisma.skillTag.upsert({
      where: { id: s.id },
      update: {},
      create: s,
    });
  }
  console.log('Created skill tags for Module 7');

  // ─── Module ────────────────────────────────────────────────────────
  const mod = await prisma.module.create({
    data: {
      id: 'module-007',
      stageId: 'stage-004',
      title: 'Professional Python',
      slug: 'professional-python',
      description: 'Write clean, maintainable, production-grade Python with modern tooling and patterns used in industry.',
      order: 2,
    },
  });

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 24 — Comprehensions & Generators
  // ═══════════════════════════════════════════════════════════════════
  const lesson24 = await prisma.lesson.create({
    data: {
      id: 'lesson-024',
      moduleId: mod.id,
      title: 'Comprehensions & Generators',
      slug: 'comprehensions-generators',
      order: 1,
      content: `# Comprehensions & Generators

Python gives you elegant, concise ways to create collections and produce sequences of values lazily. Mastering comprehensions and generators is what separates a beginner from an intermediate Python developer.

## List Comprehensions

A list comprehension creates a new list by applying an expression to each item in an iterable, optionally filtering items with a condition.

\`\`\`python
# Traditional loop
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension — same result, one line
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
\`\`\`

### With Filtering

Add an \`if\` clause to include only items that pass a condition:

\`\`\`python
evens = [x for x in range(20) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
\`\`\`

### Nested Comprehensions

You can nest loops inside comprehensions, though readability should always come first:

\`\`\`python
# Flatten a 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
\`\`\`

## Dict Comprehensions

The same idea works for dictionaries — use \`{key: value for ...}\`:

\`\`\`python
words = ["hello", "world", "python"]
word_lengths = {w: len(w) for w in words}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}
\`\`\`

You can also invert a dictionary:

\`\`\`python
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}
\`\`\`

## Set Comprehensions

Use curly braces without key-value pairs to build a set:

\`\`\`python
nums = [1, 2, 2, 3, 3, 3, 4]
unique_squares = {x ** 2 for x in nums}
print(unique_squares)  # {16, 1, 4, 9}
\`\`\`

## Generator Expressions

A generator expression looks just like a list comprehension but uses parentheses instead of brackets. The critical difference: it produces values **lazily** — one at a time, on demand — instead of building the entire list in memory.

\`\`\`python
# List comprehension — creates entire list in memory
sum_list = sum([x ** 2 for x in range(1000000)])

# Generator expression — produces values one at a time
sum_gen = sum(x ** 2 for x in range(1000000))
\`\`\`

Both produce the same result, but the generator version uses almost no extra memory.

## Generator Functions with \`yield\`

For more complex logic, use a generator function with the \`yield\` keyword. Each time the function hits \`yield\`, it pauses and produces a value. When \`next()\` is called again, it resumes from where it stopped.

\`\`\`python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)
# 5, 4, 3, 2, 1
\`\`\`

### Fibonacci Generator

A classic example showing generators' power:

\`\`\`python
def fibonacci(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

print(list(fibonacci(10)))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
\`\`\`

## When to Use What

| Use Case | Tool |
|----------|------|
| Transform & filter a list | List comprehension |
| Build a dictionary from data | Dict comprehension |
| Remove duplicates with transformation | Set comprehension |
| Process large data without memory issues | Generator expression / function |
| Create infinite sequences | Generator function |

**Rule of thumb:** If the comprehension fits on one line and is easy to read, use it. If it takes more than 2 nested loops or complex conditions, stick with a regular for loop for clarity.`,
      commonMistakes: `## Common Mistakes

### 1. Over-Complicated Comprehensions
\`\`\`python
# Too complex — hard to read
result = [f(x) for x in data if g(x) for y in h(x) if p(y)]

# Better: use a regular loop
result = []
for x in data:
    if g(x):
        for y in h(x):
            if p(y):
                result.append(f(x))
\`\`\`

### 2. Consuming a Generator Twice
Generators are **one-shot** — once exhausted, they produce no more values:
\`\`\`python
gen = (x for x in range(5))
print(list(gen))  # [0, 1, 2, 3, 4]
print(list(gen))  # [] — empty! Already consumed.
\`\`\`

### 3. Modifying a List While Iterating
\`\`\`python
# WRONG — never modify a list while looping over it
nums = [1, 2, 3, 4, 5]
for n in nums:
    if n % 2 == 0:
        nums.remove(n)  # Skips elements!

# RIGHT — use a comprehension to create a new list
nums = [n for n in nums if n % 2 != 0]
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-070',
        lessonId: lesson24.id,
        prompt: 'Use a list comprehension to create a list of squares of even numbers from 1 to 20 (inclusive). Print the result.',
        starterCode: '# Use a list comprehension with a condition\nsquares = # your code here\nprint(squares)\n',
        expectedOutput: '[4, 16, 36, 64, 100, 144, 196, 256, 324, 400]',
        testCode: '',
        hints: JSON.stringify([
          'Start with [x ** 2 for x in range(1, 21)]',
          'Add a condition: if x % 2 == 0',
          'Full answer: [x ** 2 for x in range(1, 21) if x % 2 == 0]',
        ]),
        order: 1,
      },
      {
        id: 'exercise-071',
        lessonId: lesson24.id,
        prompt: 'Use a dict comprehension to create a dictionary mapping each word in `words = ["hello", "world", "python", "code"]` to its length. Print the result.',
        starterCode: 'words = ["hello", "world", "python", "code"]\nword_lengths = # your dict comprehension here\nprint(word_lengths)\n',
        expectedOutput: "{'hello': 5, 'world': 5, 'python': 6, 'code': 4}",
        testCode: '',
        hints: JSON.stringify([
          'Dict comprehension syntax: {key: value for item in iterable}',
          'The key is the word, the value is len(word)',
          'Answer: {w: len(w) for w in words}',
        ]),
        order: 2,
      },
      {
        id: 'exercise-072',
        lessonId: lesson24.id,
        prompt: 'Write a generator function `fibonacci(n)` that yields the first n Fibonacci numbers (starting 0, 1, 1, 2...). Print `list(fibonacci(10))`.',
        starterCode: 'def fibonacci(n):\n    # Use yield to produce Fibonacci numbers\n    pass\n\nprint(list(fibonacci(10)))\n',
        expectedOutput: '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]',
        testCode: '',
        hints: JSON.stringify([
          'Start with a, b = 0, 1 and a counter',
          'In a while loop: yield a, then update a, b = b, a + b',
          'Loop while count < n, incrementing count each iteration',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-070',
        lessonId: lesson24.id,
        question: 'What is the main advantage of generators over lists?',
        type: 'MCQ',
        options: JSON.stringify(['They are faster', 'They use less memory by yielding values lazily', 'They support more operations', 'They are easier to write']),
        correctAnswer: 'They use less memory by yielding values lazily',
        explanation: 'Generators produce values one at a time (lazily) instead of storing the entire sequence in memory. This makes them ideal for large or infinite sequences.',
        order: 1,
      },
      {
        id: 'quiz-071',
        lessonId: lesson24.id,
        question: 'A list comprehension can replace any for loop.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation: 'List comprehensions are great for simple transformations and filters, but complex logic with multiple side effects, exception handling, or state management is better expressed with regular loops.',
        order: 2,
      },
      {
        id: 'quiz-072',
        lessonId: lesson24.id,
        question: 'What keyword is used to produce values from a generator function?',
        type: 'MCQ',
        options: JSON.stringify(['return', 'yield', 'generate', 'next']),
        correctAnswer: 'yield',
        explanation: 'The yield keyword pauses the function and produces a value. When next() is called, execution resumes from where it left off.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 24: Comprehensions & Generators');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 25 — Type Hints & Dataclasses
  // ═══════════════════════════════════════════════════════════════════
  const lesson25 = await prisma.lesson.create({
    data: {
      id: 'lesson-025',
      moduleId: mod.id,
      title: 'Type Hints & Dataclasses',
      slug: 'type-hints-dataclasses',
      order: 2,
      content: `# Type Hints & Dataclasses

Modern Python uses **type hints** to document what types your functions expect and return, and **dataclasses** to eliminate boilerplate when creating classes that primarily store data. These two features together make your code significantly more readable and maintainable.

## Type Hints Basics

Type hints (introduced in Python 3.5) let you annotate variables and function signatures with expected types:

\`\`\`python
def greet(name: str) -> str:
    return f"Hello, {name}!"

age: int = 25
price: float = 19.99
active: bool = True
\`\`\`

**Important:** Type hints are **not enforced at runtime**. Python won't raise an error if you pass the wrong type — they're documentation for developers and tools like mypy.

## Common Type Annotations

\`\`\`python
from typing import Optional, Union

# Basic types
x: int = 10
y: float = 3.14
s: str = "hello"
flag: bool = True

# Collections (Python 3.9+)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95, "Bob": 87}
coords: tuple[float, float] = (1.0, 2.0)
unique: set[int] = {1, 2, 3}

# Optional — can be the type or None
middle_name: Optional[str] = None

# Union — can be one of several types
value: Union[int, str] = 42
\`\`\`

## Function Annotations

\`\`\`python
def calculate_average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

def find_user(user_id: int) -> Optional[dict]:
    # Returns a dict or None if not found
    users = {1: {"name": "Alice"}, 2: {"name": "Bob"}}
    return users.get(user_id)
\`\`\`

## Dataclasses

The \`@dataclass\` decorator (Python 3.7+) auto-generates \`__init__\`, \`__repr__\`, \`__eq__\`, and more:

\`\`\`python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(3.0, 4.0)
p2 = Point(3.0, 4.0)
print(p1)        # Point(x=3.0, y=4.0)
print(p1 == p2)  # True — __eq__ compares all fields
\`\`\`

Without \`@dataclass\`, you'd need to write \`__init__\`, \`__repr__\`, and \`__eq__\` yourself — that's a lot of boilerplate.

### Default Values

\`\`\`python
@dataclass
class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

cfg = Config()
print(cfg)  # Config(host='localhost', port=8080, debug=False)
\`\`\`

### Mutable Defaults with \`field()\`

Never use mutable defaults directly — use \`field(default_factory=...)\`:

\`\`\`python
from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    grades: list[int] = field(default_factory=list)

s1 = Student("Alice")
s1.grades.append(95)
s2 = Student("Bob")
print(s2.grades)  # [] — each instance gets its own list
\`\`\`

### Frozen Dataclasses

Use \`frozen=True\` to make instances immutable:

\`\`\`python
@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

c = Color(255, 128, 0)
# c.r = 100  # FrozenInstanceError!
\`\`\`

### Methods & Properties in Dataclasses

Dataclasses can have methods just like regular classes:

\`\`\`python
from dataclasses import dataclass
import math

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)
\`\`\`

## When to Use Dataclasses vs Regular Classes

| Use dataclass when... | Use regular class when... |
|-----------------------|---------------------------|
| Primary purpose is storing data | Complex initialization logic |
| You want auto-generated \`__init__\`, \`__repr__\`, \`__eq__\` | You need full control over construction |
| Lots of fields | Mostly methods, few fields |
| Config objects, DTOs, records | Stateful services, managers |`,
      commonMistakes: `## Common Mistakes

### 1. Thinking Type Hints Enforce Types
\`\`\`python
def add(a: int, b: int) -> int:
    return a + b

# This runs fine — no error at runtime!
print(add("hello", " world"))  # "hello world"
\`\`\`
Type hints are for documentation and static analysis tools (mypy), not runtime enforcement.

### 2. Mutable Default in Dataclass
\`\`\`python
# WRONG — shared list across all instances
@dataclass
class Bad:
    items: list = []  # TypeError!

# RIGHT — use default_factory
@dataclass
class Good:
    items: list = field(default_factory=list)
\`\`\`

### 3. Forgetting \`from __future__ import annotations\`
For forward references and using newer syntax in older Python versions:
\`\`\`python
from __future__ import annotations  # Allows string-based annotations

class Node:
    def __init__(self, next: Node | None = None):  # Works!
        self.next = next
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-073',
        lessonId: lesson25.id,
        prompt: 'Create a dataclass `Point` with `x: float` and `y: float`. Add a method `distance_to(other: Point) -> float` that returns the Euclidean distance. Create p1=Point(0,0), p2=Point(3,4), print distance rounded to 1 decimal.',
        starterCode: 'from dataclasses import dataclass\nimport math\n\n# Define the Point dataclass here\n\np1 = Point(0, 0)\np2 = Point(3, 4)\nprint(round(p1.distance_to(p2), 1))\n',
        expectedOutput: '5.0',
        testCode: '',
        hints: JSON.stringify([
          'Use @dataclass decorator above the class definition',
          'distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)',
          'Full class: @dataclass\\nclass Point:\\n    x: float\\n    y: float\\n    def distance_to(self, other) -> float:\\n        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)',
        ]),
        order: 1,
      },
      {
        id: 'exercise-074',
        lessonId: lesson25.id,
        prompt: 'Create a frozen dataclass `Color` with `r: int`, `g: int`, `b: int`. Add a method `hex()` that returns the hex color string like "#FF8000". Create Color(255, 128, 0) and print its hex().',
        starterCode: 'from dataclasses import dataclass\n\n# Define the frozen Color dataclass here\n\nc = Color(255, 128, 0)\nprint(c.hex())\n',
        expectedOutput: '#FF8000',
        testCode: '',
        hints: JSON.stringify([
          'Use @dataclass(frozen=True) to make it immutable',
          'Format hex with f"#{self.r:02X}{self.g:02X}{self.b:02X}"',
          ':02X formats an integer as uppercase hex with at least 2 digits',
        ]),
        order: 2,
      },
      {
        id: 'exercise-075',
        lessonId: lesson25.id,
        prompt: 'Create a dataclass `Student` with `name: str` and `grades: list[int]` (default empty list using field). Add a property `average` that returns the mean of grades. Create a student, add grades [90, 85, 92], print average.',
        starterCode: 'from dataclasses import dataclass, field\n\n# Define the Student dataclass here\n\ns = Student("Alice")\ns.grades.extend([90, 85, 92])\nprint(s.average)\n',
        expectedOutput: '89.0',
        testCode: '',
        hints: JSON.stringify([
          'Use grades: list[int] = field(default_factory=list)',
          'Add @property decorator for the average method',
          'average returns sum(self.grades) / len(self.grades)',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-073',
        lessonId: lesson25.id,
        question: 'What does `frozen=True` do in a dataclass?',
        type: 'MCQ',
        options: JSON.stringify(['Prevents adding new methods', 'Makes instances immutable', 'Freezes the class from being inherited', 'Disables __init__']),
        correctAnswer: 'Makes instances immutable',
        explanation: 'frozen=True makes all fields read-only after creation. Attempting to modify a field raises a FrozenInstanceError.',
        order: 1,
      },
      {
        id: 'quiz-074',
        lessonId: lesson25.id,
        question: 'Type hints in Python enforce types at runtime by default.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation: 'Type hints are purely informational — Python does not check or enforce them at runtime. They are used by IDEs and static analysis tools like mypy.',
        order: 2,
      },
      {
        id: 'quiz-075',
        lessonId: lesson25.id,
        question: 'What is `field(default_factory=list)` used for in dataclasses?',
        type: 'MCQ',
        options: JSON.stringify(['To create a list field shared by all instances', 'To avoid mutable default argument bugs', 'To validate list inputs', 'To convert inputs to lists']),
        correctAnswer: 'To avoid mutable default argument bugs',
        explanation: 'Without default_factory, a mutable default (like a list) would be shared across all instances. default_factory=list creates a new list for each instance.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 25: Type Hints & Dataclasses');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 26 — Project Structure & Git Basics
  // ═══════════════════════════════════════════════════════════════════
  const lesson26 = await prisma.lesson.create({
    data: {
      id: 'lesson-026',
      moduleId: mod.id,
      title: 'Project Structure & Git Basics',
      slug: 'project-structure-git',
      order: 3,
      content: `# Project Structure & Git Basics

Every professional Python developer needs to know how to organize a project properly and use version control. This lesson covers the standard Python project layout and the essential Git workflow.

## Python Project Layout

A well-structured Python project looks like this:

\`\`\`
my-project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── main.py
│       ├── models.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   └── test_utils.py
├── .gitignore
├── .env
├── README.md
├── requirements.txt
└── pyproject.toml
\`\`\`

### Key Files Explained

**\`__init__.py\`** — Makes a directory a Python package. Can be empty or contain package-level imports:
\`\`\`python
# src/my_project/__init__.py
from .utils import helper_function
from .models import User
\`\`\`

**\`requirements.txt\`** — Lists all project dependencies:
\`\`\`
requests==2.31.0
pandas>=2.0.0
numpy~=1.24.0
python-dotenv
\`\`\`

**\`.gitignore\`** — Files Git should never track:
\`\`\`
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
\`\`\`

**\`pyproject.toml\`** — Modern Python project configuration (replaces setup.py):
\`\`\`toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["requests", "pandas"]
\`\`\`

## Imports: Absolute vs Relative

\`\`\`python
# Absolute import (preferred)
from my_project.utils import helper_function

# Relative import (within the same package)
from .utils import helper_function
from ..models import User  # Go up one level
\`\`\`

## Virtual Environments

Always isolate project dependencies with a virtual environment:

\`\`\`bash
# Create a virtual environment
python -m venv .venv

# Activate it
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Save current dependencies
pip freeze > requirements.txt

# Deactivate when done
deactivate
\`\`\`

## Git Basics

Git is the version control system used by virtually every software team. Here's the essential workflow:

### Initial Setup
\`\`\`bash
git init                    # Initialize a new repo
git clone <url>             # Clone an existing repo
\`\`\`

### The Core Workflow
\`\`\`bash
git status                  # See what's changed
git add <file>              # Stage specific files
git add .                   # Stage all changes
git commit -m "message"     # Commit staged changes
git push origin main        # Push to remote
git pull origin main        # Pull latest changes
\`\`\`

### Branching
\`\`\`bash
git branch feature-login    # Create a branch
git checkout feature-login  # Switch to it
# or combined:
git checkout -b feature-login

# After making changes:
git push -u origin feature-login

# Merge back to main:
git checkout main
git merge feature-login
\`\`\`

### Commit Messages

Good commit messages follow this pattern:
\`\`\`
feat: Add user authentication
fix: Handle empty input in calculator
docs: Update README with setup instructions
refactor: Extract validation into helper function
\`\`\`

## Environment Variables

Never hardcode secrets. Use \`.env\` files with \`python-dotenv\`:

\`\`\`python
# .env file (NEVER commit this!)
API_KEY=sk-abc123
DATABASE_URL=postgresql://localhost/mydb

# In your code
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Committing Secrets
\`\`\`bash
# NEVER commit .env files or API keys
# Always add .env to .gitignore BEFORE your first commit
echo ".env" >> .gitignore
\`\`\`

### 2. Not Using Virtual Environments
Installing packages globally leads to version conflicts between projects. Always use \`python -m venv\`.

### 3. Circular Imports
\`\`\`python
# models.py
from utils import format_name  # utils imports from models too!

# Fix: restructure code, use local imports, or create a third module
def process():
    from utils import format_name  # Local import breaks the cycle
\`\`\`

### 4. Committing \`__pycache__\`
Add \`__pycache__/\` to your \`.gitignore\` right away. These are compiled bytecode files that should never be in version control.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-076',
        lessonId: lesson26.id,
        prompt: 'Simulate a simple module system: create a dict `utils` with a key "greet" mapped to a lambda that takes `name` and returns `f"Hello, {name}!"`. Call `utils["greet"]("World")` and print the result.',
        starterCode: '# Simulate a module with a dictionary\nutils = # your code here\nresult = utils["greet"]("World")\nprint(result)\n',
        expectedOutput: 'Hello, World!',
        testCode: '',
        hints: JSON.stringify([
          'A lambda is defined as: lambda param: expression',
          'utils = {"greet": lambda name: f"Hello, {name}!"}',
          'The lambda takes name and returns the f-string greeting',
        ]),
        order: 1,
      },
      {
        id: 'exercise-077',
        lessonId: lesson26.id,
        prompt: 'Write a function `parse_requirements(text)` that takes a requirements.txt content string (one package per line, some with ==version) and returns a sorted list of package names without versions. Test with "requests==2.28.0\\nnumpy==1.23.1\\npandas\\nflask==2.2.0".',
        starterCode: 'def parse_requirements(text):\n    # Split lines, extract package names, sort\n    pass\n\nreqs = "requests==2.28.0\\nnumpy==1.23.1\\npandas\\nflask==2.2.0"\nprint(parse_requirements(reqs))\n',
        expectedOutput: "['flask', 'numpy', 'pandas', 'requests']",
        testCode: '',
        hints: JSON.stringify([
          'Split the text by newlines with text.split("\\n")',
          'For each line, split by "==" and take the first part: line.split("==")[0]',
          'Return sorted(package_names)',
        ]),
        order: 2,
      },
      {
        id: 'exercise-078',
        lessonId: lesson26.id,
        prompt: 'Write a function `create_gitignore(items)` that takes a list of patterns and returns a string with a header comment "# Auto-generated .gitignore" followed by each pattern on its own line. Test with ["*.pyc", "__pycache__/", ".env", "*.db"].',
        starterCode: 'def create_gitignore(items):\n    # Build the .gitignore content string\n    pass\n\npatterns = ["*.pyc", "__pycache__/", ".env", "*.db"]\nprint(create_gitignore(patterns))\n',
        expectedOutput: '# Auto-generated .gitignore\n*.pyc\n__pycache__/\n.env\n*.db',
        testCode: '',
        hints: JSON.stringify([
          'Start with the header: "# Auto-generated .gitignore"',
          'Join the items with newlines: "\\n".join(items)',
          'Combine: header + "\\n" + "\\n".join(items)',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-076',
        lessonId: lesson26.id,
        question: 'What file makes a directory a Python package?',
        type: 'MCQ',
        options: JSON.stringify(['__main__.py', '__init__.py', 'setup.py', 'package.json']),
        correctAnswer: '__init__.py',
        explanation: '__init__.py marks a directory as a Python package, allowing its modules to be imported. It can be empty or contain package initialization code.',
        order: 1,
      },
      {
        id: 'quiz-077',
        lessonId: lesson26.id,
        question: 'You should commit your .env file to Git to share environment variables.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation: 'Never commit .env files — they contain secrets like API keys and database passwords. Add .env to .gitignore and share a .env.example template instead.',
        order: 2,
      },
      {
        id: 'quiz-078',
        lessonId: lesson26.id,
        question: 'What command creates a virtual environment in Python?',
        type: 'MCQ',
        options: JSON.stringify(['pip install venv', 'python -m venv myenv', 'virtualenv --create myenv', 'python --venv myenv']),
        correctAnswer: 'python -m venv myenv',
        explanation: 'python -m venv myenv creates a virtual environment in the myenv directory using Python\'s built-in venv module.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 26: Project Structure & Git Basics');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 27 — Working with APIs
  // ═══════════════════════════════════════════════════════════════════
  const lesson27 = await prisma.lesson.create({
    data: {
      id: 'lesson-027',
      moduleId: mod.id,
      title: 'Working with APIs — requests & JSON',
      slug: 'apis-requests-json',
      order: 4,
      content: `# Working with APIs — requests & JSON

Almost every modern application communicates with external services through APIs (Application Programming Interfaces). As a Python developer, knowing how to make HTTP requests and process JSON responses is an essential skill used daily in data science, web development, and AI.

## HTTP Basics

HTTP (HyperText Transfer Protocol) is how clients (your code) talk to servers (APIs):

| Method | Purpose | Example |
|--------|---------|---------|
| GET | Retrieve data | Fetch a list of users |
| POST | Send data / create | Create a new user |
| PUT | Update data | Update user profile |
| DELETE | Remove data | Delete a user |

### Status Codes

| Code | Meaning |
|------|---------|
| 200 | OK — success |
| 201 | Created — resource created |
| 400 | Bad Request — invalid input |
| 401 | Unauthorized — need authentication |
| 403 | Forbidden — no permission |
| 404 | Not Found |
| 429 | Too Many Requests — rate limited |
| 500 | Internal Server Error |

## The \`requests\` Library

\`requests\` is Python's most popular HTTP library:

\`\`\`python
import requests

# GET request
response = requests.get("https://api.example.com/users")
print(response.status_code)  # 200
print(response.json())       # Parse JSON response

# POST request
data = {"name": "Alice", "email": "alice@example.com"}
response = requests.post("https://api.example.com/users", json=data)
\`\`\`

### Query Parameters

\`\`\`python
# Instead of building URLs manually:
# https://api.example.com/search?q=python&page=1

params = {"q": "python", "page": 1}
response = requests.get("https://api.example.com/search", params=params)
print(response.url)  # Full URL with query string
\`\`\`

### Headers & Authentication

\`\`\`python
# API Key in headers
headers = {
    "Authorization": "Bearer your-api-key-here",
    "Content-Type": "application/json"
}
response = requests.get("https://api.example.com/data", headers=headers)
\`\`\`

### Error Handling

\`\`\`python
import requests

try:
    response = requests.get("https://api.example.com/users", timeout=10)
    response.raise_for_status()  # Raises HTTPError for 4xx/5xx
    data = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
\`\`\`

## Working with JSON

JSON (JavaScript Object Notation) is the standard data format for APIs. Python's \`json\` module converts between JSON strings and Python objects:

\`\`\`python
import json

# Python dict → JSON string
data = {"name": "Alice", "scores": [95, 87, 92]}
json_string = json.dumps(data, indent=2)
print(json_string)

# JSON string → Python dict
parsed = json.loads(json_string)
print(parsed["name"])  # "Alice"
\`\`\`

### Processing API Responses

A common pattern for working with API data:

\`\`\`python
import requests

response = requests.get("https://api.example.com/users")
users = response.json()

# Extract specific fields
for user in users:
    name = user.get("name", "Unknown")
    email = user.get("email", "N/A")
    print(f"Name: {name}, Email: {email}")
\`\`\`

## Building an API Client Class

For production code, wrap API calls in a reusable class:

\`\`\`python
import requests

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def get(self, endpoint: str, params: dict = None):
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def post(self, endpoint: str, data: dict = None):
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
\`\`\`

## Rate Limiting

Many APIs limit how many requests you can make. Handle this gracefully:

\`\`\`python
import time
import requests

def fetch_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:  # Rate limited
            wait = int(response.headers.get("Retry-After", 2 ** attempt))
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise Exception("Max retries exceeded")
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Not Checking Status Codes
\`\`\`python
# WRONG — assumes success
response = requests.get(url)
data = response.json()  # Crashes if response is an error page

# RIGHT — check first
response = requests.get(url)
response.raise_for_status()  # or check response.status_code
data = response.json()
\`\`\`

### 2. Hardcoding API Keys
\`\`\`python
# WRONG — secret in source code
headers = {"Authorization": "Bearer sk-abc123secret"}

# RIGHT — use environment variables
import os
api_key = os.getenv("API_KEY")
headers = {"Authorization": f"Bearer {api_key}"}
\`\`\`

### 3. No Timeout
\`\`\`python
# WRONG — can hang forever
response = requests.get(url)

# RIGHT — always set a timeout
response = requests.get(url, timeout=10)
\`\`\`

### 4. Not Using \`.get()\` for Dictionary Access
\`\`\`python
# WRONG — crashes if key missing
name = data["name"]

# RIGHT — safe access with default
name = data.get("name", "Unknown")
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-079',
        lessonId: lesson27.id,
        prompt: 'Simulate an API response. Create a dict with status 200 and data containing a list of users [{name: "Alice", age: 30}, {name: "Bob", age: 25}]. Extract and print each user as "Name: {name}, Age: {age}".',
        starterCode: 'response = {\n    "status": 200,\n    "data": {\n        "users": [\n            {"name": "Alice", "age": 30},\n            {"name": "Bob", "age": 25}\n        ]\n    }\n}\n\n# Extract and print each user\n',
        expectedOutput: 'Name: Alice, Age: 30\nName: Bob, Age: 25',
        testCode: '',
        hints: JSON.stringify([
          'Access the users list: response["data"]["users"]',
          'Loop through users and use f-string formatting',
          'print(f"Name: {user[\\"name\\"]}, Age: {user[\\"age\\"]}")',
        ]),
        order: 1,
      },
      {
        id: 'exercise-080',
        lessonId: lesson27.id,
        prompt: 'Write a function `build_url(base, params)` that takes a base URL and a dict of query params and returns the full URL. Test with base="https://api.example.com/search" and params={"q": "python", "page": "1", "limit": "10"}.',
        starterCode: 'def build_url(base, params):\n    # Build the URL with query string\n    pass\n\nurl = build_url("https://api.example.com/search", {"q": "python", "page": "1", "limit": "10"})\nprint(url)\n',
        expectedOutput: 'https://api.example.com/search?q=python&page=1&limit=10',
        testCode: '',
        hints: JSON.stringify([
          'Join params with "&": "&".join(f"{k}={v}" for k, v in params.items())',
          'Combine: f"{base}?{query_string}"',
          'Full: return base + "?" + "&".join(f"{k}={v}" for k, v in params.items())',
        ]),
        order: 2,
      },
      {
        id: 'exercise-081',
        lessonId: lesson27.id,
        prompt: 'Create a class `APIClient` with `base_url` and `headers` (dict). Add a `build_request(endpoint, method="GET")` method returning a dict with "url", "method", and "headers" keys. Test and print the result.',
        starterCode: 'class APIClient:\n    def __init__(self, base_url, headers):\n        pass\n\n    def build_request(self, endpoint, method="GET"):\n        pass\n\nclient = APIClient("https://api.example.com", {"Authorization": "Bearer token123"})\nprint(client.build_request("/users"))\n',
        expectedOutput: "{'url': 'https://api.example.com/users', 'method': 'GET', 'headers': {'Authorization': 'Bearer token123'}}",
        testCode: '',
        hints: JSON.stringify([
          'Store base_url and headers as self attributes in __init__',
          'build_request returns {"url": self.base_url + endpoint, "method": method, "headers": self.headers}',
          'Make sure the dict keys are in the correct order: url, method, headers',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-079',
        lessonId: lesson27.id,
        question: 'What HTTP status code means "Not Found"?',
        type: 'MCQ',
        options: JSON.stringify(['200', '301', '404', '500']),
        correctAnswer: '404',
        explanation: '404 means the requested resource was not found on the server. 200 is OK, 301 is redirect, and 500 is internal server error.',
        order: 1,
      },
      {
        id: 'quiz-080',
        lessonId: lesson27.id,
        question: 'The requests.get() method returns a Response object.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation: 'requests.get() returns a Response object with properties like .status_code, .text, .json(), .headers, and more.',
        order: 2,
      },
      {
        id: 'quiz-081',
        lessonId: lesson27.id,
        question: 'What is the safest way to include API keys in your code?',
        type: 'MCQ',
        options: JSON.stringify(['Hardcode them in the script', 'Store them in environment variables', 'Put them in comments', 'Include them in the URL']),
        correctAnswer: 'Store them in environment variables',
        explanation: 'Environment variables keep secrets out of your source code. Use .env files locally and environment configuration in production.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 27: Working with APIs');

  // ═══════════════════════════════════════════════════════════════════
  //  PROJECTS
  // ═══════════════════════════════════════════════════════════════════
  await prisma.project.create({
    data: {
      id: 'project-007',
      title: 'OOP Contact Book',
      slug: 'oop-contact-book',
      stage: 'ADVANCED',
      order: 7,
      brief: 'Build a command-line contact book application using OOP principles — classes, inheritance, and data persistence.',
      requirements: JSON.stringify([
        'Create a Contact class with name, phone, email, and category fields',
        'Create a ContactBook class that manages a list of contacts',
        'Implement add, remove, search (by name), and list all operations',
        'Use inheritance to create SpecialContact subclasses (e.g., WorkContact with company field)',
        'Save/load contacts to/from a JSON file for persistence',
      ]),
      stretchGoals: JSON.stringify([
        'Add contact groups/categories with filtering',
        'Implement fuzzy search that finds partial name matches',
        'Add an export to CSV feature',
      ]),
      steps: JSON.stringify([
        { title: 'Design the Contact class', description: 'Define the Contact class with __init__, __str__, __repr__, and a to_dict() method for serialization.' },
        { title: 'Build the ContactBook manager', description: 'Create the ContactBook class with add_contact(), remove_contact(), search(), and list_all() methods.' },
        { title: 'Add inheritance', description: 'Create WorkContact and PersonalContact subclasses with additional fields specific to each type.' },
        { title: 'Implement persistence', description: 'Add save_to_file() and load_from_file() methods using JSON serialization.' },
        { title: 'Build the CLI interface', description: 'Create a main loop with a menu that lets users interact with the contact book through the terminal.' },
      ]),
      rubric: JSON.stringify([
        { criterion: 'OOP Design', description: 'Proper use of classes, encapsulation, and inheritance with clear responsibilities.' },
        { criterion: 'Functionality', description: 'All CRUD operations work correctly with proper error handling.' },
        { criterion: 'Data Persistence', description: 'Contacts are saved to and loaded from a JSON file correctly.' },
        { criterion: 'Code Quality', description: 'Clean code with docstrings, type hints, and meaningful variable names.' },
      ]),
      solutionUrl: null,
    },
  });

  await prisma.project.create({
    data: {
      id: 'project-008',
      title: 'REST API Data Fetcher',
      slug: 'api-data-fetcher',
      stage: 'ADVANCED',
      order: 8,
      brief: 'Build a Python class that fetches data from a public REST API, processes JSON responses, and displays formatted results.',
      requirements: JSON.stringify([
        'Create an APIClient class with configurable base URL and headers',
        'Implement GET requests with query parameter support',
        'Parse and process JSON responses into clean Python objects',
        'Add error handling for network failures, timeouts, and HTTP errors',
        'Display results in a formatted, readable way in the terminal',
      ]),
      stretchGoals: JSON.stringify([
        'Add caching to avoid repeated API calls for the same data',
        'Support multiple API endpoints (e.g., users, posts, comments)',
        'Export fetched data to CSV or JSON files',
      ]),
      steps: JSON.stringify([
        { title: 'Set up the APIClient class', description: 'Create the class with __init__ accepting base_url, optional headers, and a requests.Session for connection reuse.' },
        { title: 'Implement the GET method', description: 'Build a get() method that constructs URLs, passes params, handles timeouts, and returns parsed JSON.' },
        { title: 'Add error handling', description: 'Handle ConnectionError, Timeout, HTTPError and provide meaningful error messages.' },
        { title: 'Process and format data', description: 'Create methods to transform raw API data into formatted output (tables, summaries, etc.).' },
        { title: 'Build the main interface', description: 'Create a script that demonstrates fetching from a public API like JSONPlaceholder or OpenWeather.' },
      ]),
      rubric: JSON.stringify([
        { criterion: 'API Integration', description: 'Correct use of requests library with proper URL construction and parameter handling.' },
        { criterion: 'Error Handling', description: 'Graceful handling of network errors, timeouts, and invalid responses.' },
        { criterion: 'Code Architecture', description: 'Clean OOP design with reusable APIClient class and separation of concerns.' },
        { criterion: 'Output Quality', description: 'Data is presented in a clear, formatted, and user-friendly way.' },
      ]),
      solutionUrl: null,
    },
  });
  console.log('Seeded Stage 4 Projects');

  console.log('✅ Module 7 (Professional Python) seeding complete!');
}

module.exports = { seedModule7 };
