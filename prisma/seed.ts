const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  console.log('Seeding Py2ML Academy database (Part 1)...');

  // ─── Clear existing data ───────────────────────────────────────────
  await prisma.earnedSkill.deleteMany();
  await prisma.activityLog.deleteMany();
  await prisma.projectProgress.deleteMany();
  await prisma.quizAttempt.deleteMany();
  await prisma.submission.deleteMany();
  await prisma.quizQuestion.deleteMany();
  await prisma.exercise.deleteMany();
  await prisma.lesson.deleteMany();
  await prisma.module.deleteMany();
  await prisma.stage.deleteMany();
  await prisma.project.deleteMany();
  await prisma.skillTag.deleteMany();
  await prisma.guestSession.deleteMany();

  // ═══════════════════════════════════════════════════════════════════
  //  SKILL TAGS
  // ═══════════════════════════════════════════════════════════════════
  const skillTags = await Promise.all([
    prisma.skillTag.create({ data: { id: 'skill-001', name: 'Python Basics', slug: 'python-basics' } }),
    prisma.skillTag.create({ data: { id: 'skill-002', name: 'Variables', slug: 'variables' } }),
    prisma.skillTag.create({ data: { id: 'skill-003', name: 'Data Types', slug: 'data-types' } }),
    prisma.skillTag.create({ data: { id: 'skill-004', name: 'Strings', slug: 'strings' } }),
    prisma.skillTag.create({ data: { id: 'skill-005', name: 'Lists', slug: 'lists' } }),
    prisma.skillTag.create({ data: { id: 'skill-006', name: 'Tuples', slug: 'tuples' } }),
    prisma.skillTag.create({ data: { id: 'skill-007', name: 'Dictionaries', slug: 'dictionaries' } }),
    prisma.skillTag.create({ data: { id: 'skill-008', name: 'Sets', slug: 'sets' } }),
    prisma.skillTag.create({ data: { id: 'skill-009', name: 'Control Flow', slug: 'control-flow' } }),
    prisma.skillTag.create({ data: { id: 'skill-010', name: 'Loops', slug: 'loops' } }),
    prisma.skillTag.create({ data: { id: 'skill-011', name: 'Functions', slug: 'functions' } }),
    prisma.skillTag.create({ data: { id: 'skill-012', name: 'Modules & Imports', slug: 'modules-imports' } }),
    prisma.skillTag.create({ data: { id: 'skill-013', name: 'File I/O', slug: 'file-io' } }),
    prisma.skillTag.create({ data: { id: 'skill-014', name: 'Exception Handling', slug: 'exception-handling' } }),
    prisma.skillTag.create({ data: { id: 'skill-015', name: 'f-Strings', slug: 'f-strings' } }),
    prisma.skillTag.create({ data: { id: 'skill-016', name: 'String Methods', slug: 'string-methods' } }),
    prisma.skillTag.create({ data: { id: 'skill-017', name: 'Lambda Functions', slug: 'lambda-functions' } }),
    prisma.skillTag.create({ data: { id: 'skill-018', name: 'Scope & Closures', slug: 'scope-closures' } }),
    prisma.skillTag.create({ data: { id: 'skill-019', name: 'Iteration Patterns', slug: 'iteration-patterns' } }),
    prisma.skillTag.create({ data: { id: 'skill-020', name: 'Virtual Environments', slug: 'virtual-environments' } }),
  ]);
  console.log(`Created ${skillTags.length} skill tags`);

  // ═══════════════════════════════════════════════════════════════════
  //  STAGES
  // ═══════════════════════════════════════════════════════════════════
  const stageA = await prisma.stage.create({
    data: {
      id: 'stage-001',
      title: 'Python Foundations',
      slug: 'python-foundations',
      description: 'Master the core building blocks of Python — from your very first print statement to writing reusable functions. This stage gives you the vocabulary and mental models you need before tackling data science or machine learning.',
      order: 1,
    },
  });

  const stageB = await prisma.stage.create({
    data: {
      id: 'stage-002',
      title: 'Python for Data',
      slug: 'python-for-data',
      description: 'Level up your Python skills with intermediate topics that every data professional needs: working with files, handling errors gracefully, using third-party packages, and wrangling data with NumPy and Pandas.',
      order: 2,
    },
  });

  const stageC = await prisma.stage.create({
    data: {
      id: 'stage-003',
      title: 'ML / AI Track',
      slug: 'ml-ai-track',
      description: 'Apply everything you have learned to build real machine-learning models. From linear regression to neural networks, this stage bridges the gap between Python programmer and ML practitioner.',
      order: 3,
    },
  });

  console.log('Created 3 stages');

  // ═══════════════════════════════════════════════════════════════════
  //  MODULES
  // ═══════════════════════════════════════════════════════════════════
  const moduleGettingStarted = await prisma.module.create({
    data: {
      id: 'module-001',
      stageId: stageA.id,
      title: 'Getting Started',
      slug: 'getting-started',
      description: 'Your first steps with Python: printing output, understanding values and variables, and mastering strings.',
      order: 1,
    },
  });

  const moduleDataStructures = await prisma.module.create({
    data: {
      id: 'module-002',
      stageId: stageA.id,
      title: 'Data Structures & Control Flow',
      slug: 'data-structures-control-flow',
      description: 'Learn how to store collections of data and control the execution path of your programs with conditionals, loops, and functions.',
      order: 2,
    },
  });

  const moduleIntermediatePython = await prisma.module.create({
    data: {
      id: 'module-003',
      stageId: stageB.id,
      title: 'Intermediate Python',
      slug: 'intermediate-python',
      description: 'Expand your toolkit with modules, file handling, error management, and an introduction to the data-science ecosystem.',
      order: 1,
    },
  });

  console.log('Created 3 modules');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 1 — Hello Python
  // ═══════════════════════════════════════════════════════════════════
  const lesson1 = await prisma.lesson.create({
    data: {
      id: 'lesson-001',
      moduleId: moduleGettingStarted.id,
      title: 'Hello Python: print, Comments & the REPL',
      slug: 'hello-python',
      order: 1,
      content: `# Hello Python: print, Comments & the REPL

Welcome to your very first Python lesson! By the end of this page you will know how to display text on the screen, leave notes in your code, and experiment interactively with the Python REPL.

---

## 1. The \`print()\` Function

The most fundamental thing a program can do is produce **output**. In Python the built-in \`print()\` function writes text to the console:

\`\`\`python
print("Hello, world!")
\`\`\`

Output:

\`\`\`
Hello, world!
\`\`\`

You can print numbers, expressions, and even multiple values separated by commas:

\`\`\`python
print(42)
print(2 + 3)
print("The answer is", 7 * 6)
\`\`\`

Output:

\`\`\`
42
5
The answer is 42
\`\`\`

When you pass multiple arguments to \`print()\`, Python inserts a **space** between each one by default. You can change the separator with the \`sep\` parameter and the line ending with \`end\`:

\`\`\`python
print("A", "B", "C", sep="-")
print("Hello", end=" ")
print("World")
\`\`\`

Output:

\`\`\`
A-B-C
Hello World
\`\`\`

---

## 2. Comments

Comments let you leave human-readable notes that Python ignores entirely. A single-line comment starts with \`#\`:

\`\`\`python
# This line is a comment
print("Not a comment")  # This part is a comment too
\`\`\`

Python does **not** have a dedicated multi-line comment syntax like \`/* ... */\` in other languages. However, developers sometimes use a triple-quoted string as a block comment because Python evaluates and discards it:

\`\`\`python
"""
This block is technically a string expression,
but it is commonly used as a multi-line comment.
"""
\`\`\`

**Best practice:** Use \`#\` for short notes and docstrings (\`\"\"\"...\"\"\"\`) only inside functions and classes to document their purpose.

---

## 3. The Python REPL

REPL stands for **Read-Eval-Print Loop**. When you type \`python\` (or \`python3\`) in your terminal without a filename you enter an interactive session:

\`\`\`
$ python3
>>> 2 + 2
4
>>> print("hi")
hi
>>> exit()
\`\`\`

The \`>>>\` prompt means Python is waiting for input. Each expression you type is immediately evaluated and the result is displayed. This is an incredibly useful scratchpad for trying ideas quickly.

**Key differences between REPL and script mode:**

| Feature | REPL | Script (.py file) |
|---|---|---|
| Shows expression results automatically | Yes | No |
| Requires \`print()\` for output | No (but you can) | Yes |
| Saves your work | No | Yes |

---

## 4. Putting It All Together

Create a file called \`hello.py\` and add:

\`\`\`python
# My very first Python script
print("Welcome to Py2ML Academy!")
print("Python version tip: use Python 3.10+")
print(1 + 1)  # quick math
\`\`\`

Run it:

\`\`\`bash
python3 hello.py
\`\`\`

You should see:

\`\`\`
Welcome to Py2ML Academy!
Python version tip: use Python 3.10+
2
\`\`\`

Congratulations — you have written and executed your first Python program!`,
      commonMistakes: `## Common Mistakes

### 1. Forgetting parentheses around \`print\`

\`\`\`python
# Wrong (Python 2 style)
print "hello"

# Right (Python 3)
print("hello")
\`\`\`

In Python 3 \`print\` is a **function**, so parentheses are mandatory.

### 2. Mismatched quotes

\`\`\`python
# Wrong
print("hello')

# Right
print("hello")
print('hello')
\`\`\`

Opening and closing quotes must be the same type.

### 3. Using a comment symbol inside a string

\`\`\`python
# This prints exactly what you expect
print("# This is not a comment")
\`\`\`

Text inside quotes is a string literal, not a comment.`,
    },
  });

  // Exercises for Lesson 1
  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-001',
        lessonId: lesson1.id,
        prompt: 'Use the `print()` function to display the text **Hello, Py2ML!** on the screen (exactly as shown, with the comma and exclamation mark).',
        starterCode: '# Write your print statement below\n',
        expectedOutput: 'Hello, Py2ML!',
        testCode: '',
        hints: JSON.stringify([
          'The print() function takes a string argument in quotes: print("your text here")',
          'Make sure your string exactly matches: Hello, Py2ML! — with a comma after Hello and an exclamation mark at the end.',
        ]),
        order: 1,
      },
      {
        id: 'exercise-002',
        lessonId: lesson1.id,
        prompt: 'Print three values — your name (as a string), your age (as a number), and your favorite language (as a string) — all in a **single** `print()` call so they appear on one line separated by spaces.',
        starterCode: '# Replace the placeholders with your own values\nname = "Alice"\nage = 25\nlanguage = "Python"\nprint(name, age, language)\n',
        expectedOutput: 'Alice 25 Python',
        testCode: '',
        hints: JSON.stringify([
          'print() accepts multiple arguments separated by commas and joins them with spaces by default.',
          'You do not need to convert the integer to a string — print() handles that automatically.',
        ]),
        order: 2,
      },
      {
        id: 'exercise-003',
        lessonId: lesson1.id,
        prompt: 'Print the numbers 1, 2, and 3 separated by dashes (`-`) instead of spaces. Use the `sep` parameter of `print()`.',
        starterCode: '# Use the sep parameter\nprint(1, 2, 3)\n',
        expectedOutput: '1-2-3',
        testCode: '',
        hints: JSON.stringify([
          'The sep parameter changes the separator: print(a, b, c, sep="...")',
          'Change the call to: print(1, 2, 3, sep="-")',
        ]),
        order: 3,
      },
    ],
  });

  // Quiz for Lesson 1
  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-001',
        lessonId: lesson1.id,
        question: 'What does REPL stand for?',
        type: 'MCQ',
        options: JSON.stringify([
          'Read-Eval-Print Loop',
          'Run-Execute-Process Loop',
          'Read-Enter-Print List',
          'Real-time Evaluation Programming Language',
        ]),
        correctAnswer: 'Read-Eval-Print Loop',
        explanation: 'REPL stands for Read-Eval-Print Loop. It reads your input, evaluates the expression, prints the result, and loops back to wait for more input.',
        order: 1,
      },
      {
        id: 'quiz-002',
        lessonId: lesson1.id,
        question: 'In Python 3, `print` is a function and requires parentheses.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation: 'In Python 2, print was a statement and parentheses were optional. In Python 3, print() is a built-in function and parentheses are required.',
        order: 2,
      },
      {
        id: 'quiz-003',
        lessonId: lesson1.id,
        question: 'What is the output of `print("X", "Y", "Z", sep="")`?',
        type: 'MCQ',
        options: JSON.stringify(['X Y Z', 'XYZ', 'X,Y,Z', 'X-Y-Z']),
        correctAnswer: 'XYZ',
        explanation: 'Setting sep="" (an empty string) removes all separators between the arguments, so the three characters are printed right next to each other: XYZ.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 1: Hello Python');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 2 — Variables & Data Types
  // ═══════════════════════════════════════════════════════════════════
  const lesson2 = await prisma.lesson.create({
    data: {
      id: 'lesson-002',
      moduleId: moduleGettingStarted.id,
      title: 'Variables & Data Types: int, float, str, bool',
      slug: 'variables-and-data-types',
      order: 2,
      content: `# Variables & Data Types

Every program needs to remember data. In Python you store data in **variables** — named containers that can hold values of different **types**.

---

## 1. Creating Variables

Python uses **dynamic typing**: you do not declare a type — the interpreter figures it out from the value you assign:

\`\`\`python
name = "Alice"        # str  (string)
age = 30              # int  (integer)
height = 5.7          # float (decimal number)
is_student = True     # bool (boolean)
\`\`\`

The \`=\` sign is the **assignment operator**. It stores the value on the right into the variable on the left.

### Naming Rules

- Must start with a letter or underscore (\`_\`)
- Can contain letters, digits, and underscores
- Case-sensitive (\`Age\` and \`age\` are different variables)
- Cannot be a Python keyword (\`if\`, \`for\`, \`class\`, etc.)

Convention: use **snake_case** for variable names (\`user_name\`, not \`userName\`).

---

## 2. Core Data Types

| Type | Example | Description |
|------|---------|-------------|
| \`int\` | \`42\`, \`-7\`, \`0\` | Whole numbers (unlimited precision) |
| \`float\` | \`3.14\`, \`-0.5\`, \`1.0\` | Decimal (floating-point) numbers |
| \`str\` | \`"hello"\`, \`'world'\` | Text enclosed in quotes |
| \`bool\` | \`True\`, \`False\` | Logical values (capitalized!) |

### Checking the Type

Use the built-in \`type()\` function:

\`\`\`python
x = 10
print(type(x))  # <class 'int'>

y = 3.14
print(type(y))  # <class 'float'>
\`\`\`

---

## 3. Type Conversion (Casting)

You can convert between types with \`int()\`, \`float()\`, \`str()\`, and \`bool()\`:

\`\`\`python
# String to integer
age_str = "25"
age_num = int(age_str)
print(age_num + 5)  # 30

# Integer to float
x = float(7)
print(x)  # 7.0

# Number to string
score = 100
message = "Your score: " + str(score)
print(message)  # Your score: 100
\`\`\`

### Truthiness

In Python every value has a boolean interpretation:

\`\`\`python
print(bool(0))       # False
print(bool(42))      # True
print(bool(""))      # False  (empty string)
print(bool("hello")) # True   (non-empty string)
print(bool(None))    # False
\`\`\`

---

## 4. Multiple Assignment

Python supports assigning several variables in one line:

\`\`\`python
a, b, c = 1, 2, 3
print(a, b, c)  # 1 2 3

# Swap two variables without a temp
x, y = 10, 20
x, y = y, x
print(x, y)  # 20 10
\`\`\`

---

## 5. Constants (Convention Only)

Python has no true constant keyword. By convention, use **ALL_CAPS** to signal that a value should not change:

\`\`\`python
PI = 3.14159
MAX_RETRIES = 5
\`\`\`

---

## Quick Summary

- Variables store values and are created with \`=\`.
- Python is dynamically typed — the type comes from the value.
- Four fundamental types: \`int\`, \`float\`, \`str\`, \`bool\`.
- Use \`type()\` to inspect and casting functions to convert.`,
      commonMistakes: `## Common Mistakes

### 1. Starting a variable name with a digit

\`\`\`python
# Wrong
2name = "Alice"

# Right
name2 = "Alice"
\`\`\`

### 2. Confusing \`=\` (assignment) with \`==\` (comparison)

\`\`\`python
x = 5    # assigns 5 to x
x == 5   # evaluates to True (comparison)
\`\`\`

### 3. Concatenating a string with a number without casting

\`\`\`python
# Wrong — raises TypeError
print("Age: " + 25)

# Right
print("Age: " + str(25))
# Or even better:
print("Age:", 25)
\`\`\`

### 4. Forgetting that \`bool\` values are capitalized

\`\`\`python
# Wrong
flag = true   # NameError

# Right
flag = True
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-004',
        lessonId: lesson2.id,
        prompt: 'Create three variables: `name` set to `"Py2ML"`, `version` set to `3`, and `is_free` set to `True`. Then print each variable on its own line.',
        starterCode: '# Create three variables and print them\n',
        expectedOutput: 'Py2ML\n3\nTrue',
        testCode: '',
        hints: JSON.stringify([
          'Assign each variable on its own line: name = "Py2ML", version = 3, is_free = True',
          'Use three separate print() calls — one for each variable: print(name), print(version), print(is_free)',
        ]),
        order: 1,
      },
      {
        id: 'exercise-005',
        lessonId: lesson2.id,
        prompt: 'Given the string `"42"`, convert it to an integer, add `8` to it, and print the result.',
        starterCode: 'value = "42"\n# Convert to int, add 8, and print the result\n',
        expectedOutput: '50',
        testCode: '',
        hints: JSON.stringify([
          'Use int() to convert the string to an integer: num = int(value)',
          'Then compute num + 8 and print the result: print(num + 8)',
        ]),
        order: 2,
      },
      {
        id: 'exercise-006',
        lessonId: lesson2.id,
        prompt: 'Use `type()` inside `print()` to display the type of the value `3.14`.',
        starterCode: '# Print the type of 3.14\n',
        expectedOutput: "<class 'float'>",
        testCode: '',
        hints: JSON.stringify([
          'You can nest function calls: print(type(3.14))',
          'The output will be: <class \'float\'>',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-004',
        lessonId: lesson2.id,
        question: 'What is the result of `type(3.0)`?',
        type: 'MCQ',
        options: JSON.stringify(["<class 'int'>", "<class 'float'>", "<class 'str'>", "<class 'number'>"]),
        correctAnswer: "<class 'float'>",
        explanation: '3.0 contains a decimal point, which makes it a float in Python, even though its fractional part is zero.',
        order: 1,
      },
      {
        id: 'quiz-005',
        lessonId: lesson2.id,
        question: 'Python variable names are case-sensitive.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation: 'In Python, `age`, `Age`, and `AGE` are three completely different variables because Python is case-sensitive.',
        order: 2,
      },
      {
        id: 'quiz-006',
        lessonId: lesson2.id,
        question: 'What does `bool("")` return?',
        type: 'MCQ',
        options: JSON.stringify(['True', 'False', '""', 'None']),
        correctAnswer: 'False',
        explanation: 'An empty string is considered "falsy" in Python. bool("") returns False, while any non-empty string returns True.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 2: Variables & Data Types');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 3 — Strings Deep Dive
  // ═══════════════════════════════════════════════════════════════════
  const lesson3 = await prisma.lesson.create({
    data: {
      id: 'lesson-003',
      moduleId: moduleGettingStarted.id,
      title: 'Strings Deep Dive: Slicing, Methods & f-Strings',
      slug: 'strings-deep-dive',
      order: 3,
      content: `# Strings Deep Dive

Strings are one of the most frequently used types in Python. In this lesson you will learn how to slice them, transform them with built-in methods, and build dynamic text with f-strings.

---

## 1. String Basics Recap

A string is a sequence of characters enclosed in single or double quotes:

\`\`\`python
greeting = "Hello"
name = 'Alice'
\`\`\`

Triple quotes allow multi-line strings:

\`\`\`python
poem = """Roses are red,
Violets are blue,
Python is awesome,
And so are you."""
\`\`\`

---

## 2. Indexing

Every character in a string has an **index** starting at 0:

\`\`\`python
word = "Python"
print(word[0])   # P
print(word[5])   # n
print(word[-1])  # n  (last character)
print(word[-2])  # o  (second to last)
\`\`\`

Negative indices count from the end: \`-1\` is the last character, \`-2\` is second-to-last, etc.

---

## 3. Slicing

Slicing extracts a **substring** using the syntax \`string[start:stop:step]\`:

\`\`\`python
s = "Hello, World!"
print(s[0:5])    # Hello       (chars 0-4)
print(s[7:12])   # World       (chars 7-11)
print(s[:5])     # Hello       (from beginning)
print(s[7:])     # World!      (to end)
print(s[::2])    # Hlo ol!     (every 2nd char)
print(s[::-1])   # !dlroW ,olleH  (reversed)
\`\`\`

The \`stop\` index is **exclusive** — the character at that position is not included.

---

## 4. Useful String Methods

Strings are **immutable** — methods return a new string rather than modifying the original.

\`\`\`python
msg = "  Hello, Python!  "

print(msg.strip())       # "Hello, Python!"   (remove whitespace)
print(msg.lower())       # "  hello, python!  "
print(msg.upper())       # "  HELLO, PYTHON!  "
print(msg.replace("Python", "World"))  # "  Hello, World!  "
print("hello".capitalize())  # "Hello"
print("hello world".title())  # "Hello World"
\`\`\`

### Searching

\`\`\`python
s = "banana"
print(s.find("nan"))    # 2  (index of first match)
print(s.count("a"))     # 3  (number of occurrences)
print(s.startswith("ba"))  # True
print(s.endswith("na"))    # True
\`\`\`

### Splitting and Joining

\`\`\`python
csv = "red,green,blue"
colors = csv.split(",")
print(colors)  # ['red', 'green', 'blue']

joined = " | ".join(colors)
print(joined)  # red | green | blue
\`\`\`

---

## 5. f-Strings (Formatted String Literals)

Introduced in Python 3.6, f-strings are the modern way to embed expressions in strings:

\`\`\`python
name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.")
# My name is Alice and I am 30 years old.

# You can put any expression inside the braces
print(f"Next year I will be {age + 1}.")
# Next year I will be 31.

# Format numbers
pi = 3.14159
print(f"Pi is approximately {pi:.2f}")
# Pi is approximately 3.14
\`\`\`

---

## 6. Escape Characters

Special characters are represented with a backslash:

| Escape | Meaning |
|--------|---------|
| \`\\n\` | Newline |
| \`\\t\` | Tab |
| \`\\\\\` | Literal backslash |
| \`\\'\` | Single quote inside single-quoted string |
| \`\\"\` | Double quote inside double-quoted string |

\`\`\`python
print("Line 1\\nLine 2")
# Line 1
# Line 2
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Off-by-one errors in slicing

\`\`\`python
s = "Python"
# Trying to get "Pyth" (first 4 chars)
print(s[0:4])  # Correct: "Pyth"
print(s[0:3])  # Wrong: only "Pyt" (3 chars)
\`\`\`

Remember: the stop index is exclusive.

### 2. Trying to modify a string in place

\`\`\`python
s = "hello"
# s[0] = "H"  # TypeError! Strings are immutable.

# Instead, create a new string:
s = "H" + s[1:]
\`\`\`

### 3. Forgetting the \`f\` prefix on f-strings

\`\`\`python
name = "Alice"
print("{name}")   # Prints: {name}  (literal braces)
print(f"{name}")  # Prints: Alice
\`\`\`

### 4. Confusing \`find()\` returning -1 with an error

\`\`\`python
s = "hello"
idx = s.find("xyz")
print(idx)  # -1 (not found, but no error)
# Use 'in' to check membership first:
if "xyz" in s:
    print(s.find("xyz"))
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-007',
        lessonId: lesson3.id,
        prompt: 'Given the string `"Py2ML Academy"`, use slicing to extract and print just `"Py2ML"` (the first 5 characters).',
        starterCode: 'text = "Py2ML Academy"\n# Use slicing to get the first 5 characters and print them\n',
        expectedOutput: 'Py2ML',
        testCode: '',
        hints: JSON.stringify([
          'Use text[start:stop] where start is 0 and stop is 5.',
          'print(text[0:5]) or equivalently print(text[:5])',
        ]),
        order: 1,
      },
      {
        id: 'exercise-008',
        lessonId: lesson3.id,
        prompt: 'Use an f-string to print `I have 3 apples and 5 oranges.` using the variables `apples = 3` and `oranges = 5`.',
        starterCode: 'apples = 3\noranges = 5\n# Use an f-string to print the sentence\n',
        expectedOutput: 'I have 3 apples and 5 oranges.',
        testCode: '',
        hints: JSON.stringify([
          'An f-string starts with the letter f before the quote: f"..."',
          'print(f"I have {apples} apples and {oranges} oranges.")',
        ]),
        order: 2,
      },
      {
        id: 'exercise-009',
        lessonId: lesson3.id,
        prompt: 'Given `sentence = "hello world"`, use string methods to print it in title case (each word capitalized).',
        starterCode: 'sentence = "hello world"\n# Print in title case\n',
        expectedOutput: 'Hello World',
        testCode: '',
        hints: JSON.stringify([
          'Python strings have a .title() method that capitalizes each word.',
          'print(sentence.title())',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-007',
        lessonId: lesson3.id,
        question: 'What does `"Python"[::-1]` produce?',
        type: 'MCQ',
        options: JSON.stringify(['Python', 'nohtyP', 'nothy', 'Error']),
        correctAnswer: 'nohtyP',
        explanation: 'The slice [::-1] reverses the string. "Python" reversed character by character is "nohtyP".',
        order: 1,
      },
      {
        id: 'quiz-008',
        lessonId: lesson3.id,
        question: 'Strings in Python are immutable.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation: 'Strings cannot be changed in place. Operations like .upper() or .replace() return a new string rather than modifying the original.',
        order: 2,
      },
      {
        id: 'quiz-009',
        lessonId: lesson3.id,
        question: 'Which method splits a string into a list of substrings?',
        type: 'MCQ',
        options: JSON.stringify(['.split()', '.slice()', '.divide()', '.partition()']),
        correctAnswer: '.split()',
        explanation: 'The .split() method divides a string by a delimiter (spaces by default) and returns a list of substrings.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 3: Strings Deep Dive');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 4 — Lists & Tuples
  // ═══════════════════════════════════════════════════════════════════
  const lesson4 = await prisma.lesson.create({
    data: {
      id: 'lesson-004',
      moduleId: moduleDataStructures.id,
      title: 'Lists & Tuples: Indexing, Mutability & Methods',
      slug: 'lists-and-tuples',
      order: 1,
      content: `# Lists & Tuples

Python gives you two primary sequence types for ordered collections: **lists** (mutable) and **tuples** (immutable). Understanding when to use each is a key skill.

---

## 1. Lists

A list is an **ordered, mutable** collection created with square brackets:

\`\`\`python
fruits = ["apple", "banana", "cherry"]
numbers = [10, 20, 30, 40, 50]
mixed = [1, "two", 3.0, True]
\`\`\`

### Indexing & Slicing

Lists support the same indexing and slicing as strings:

\`\`\`python
print(fruits[0])     # apple
print(fruits[-1])    # cherry
print(numbers[1:4])  # [20, 30, 40]
\`\`\`

### Modifying Lists

Because lists are mutable, you can change elements in place:

\`\`\`python
fruits[1] = "blueberry"
print(fruits)  # ['apple', 'blueberry', 'cherry']
\`\`\`

### Common List Methods

\`\`\`python
nums = [3, 1, 4, 1, 5]

nums.append(9)        # Add to end:  [3, 1, 4, 1, 5, 9]
nums.insert(0, 0)     # Insert at index 0: [0, 3, 1, 4, 1, 5, 9]
nums.remove(1)        # Remove first occurrence of 1: [0, 3, 4, 1, 5, 9]
popped = nums.pop()   # Remove & return last item (9): [0, 3, 4, 1, 5]
nums.sort()           # Sort in place: [0, 1, 3, 4, 5]
nums.reverse()        # Reverse in place: [5, 4, 3, 1, 0]
print(len(nums))      # 5
\`\`\`

### List Comprehensions (Preview)

A concise way to build lists:

\`\`\`python
squares = [x ** 2 for x in range(6)]
print(squares)  # [0, 1, 4, 9, 16, 25]
\`\`\`

---

## 2. Tuples

A tuple is an **ordered, immutable** collection created with parentheses (or just commas):

\`\`\`python
point = (3, 4)
rgb = (255, 128, 0)
single = (42,)  # Note the trailing comma for single-element tuples
\`\`\`

### Why Use Tuples?

- **Immutability** makes them safer for data that should not change (coordinates, RGB values, database rows).
- They are **hashable** and can be used as dictionary keys.
- Slightly faster than lists for iteration.

### Tuple Unpacking

\`\`\`python
point = (10, 20)
x, y = point
print(x)  # 10
print(y)  # 20

# Swap trick uses tuple packing/unpacking
a, b = 1, 2
a, b = b, a
print(a, b)  # 2 1
\`\`\`

---

## 3. Converting Between Lists and Tuples

\`\`\`python
my_list = [1, 2, 3]
my_tuple = tuple(my_list)
print(my_tuple)  # (1, 2, 3)

back_to_list = list(my_tuple)
print(back_to_list)  # [1, 2, 3]
\`\`\`

---

## 4. Useful Built-in Functions

These work with both lists and tuples:

\`\`\`python
data = [10, 5, 8, 3, 12]
print(len(data))   # 5
print(min(data))   # 3
print(max(data))   # 12
print(sum(data))   # 38
print(sorted(data))  # [3, 5, 8, 10, 12]  (returns new list)
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Forgetting the trailing comma in a single-element tuple

\`\`\`python
not_a_tuple = (42)    # This is just an int!
is_a_tuple = (42,)    # This is a tuple
print(type(not_a_tuple))  # <class 'int'>
print(type(is_a_tuple))   # <class 'tuple'>
\`\`\`

### 2. Trying to modify a tuple

\`\`\`python
t = (1, 2, 3)
# t[0] = 10  # TypeError! Tuples are immutable.
\`\`\`

### 3. Confusing \`sort()\` and \`sorted()\`

\`\`\`python
nums = [3, 1, 2]
# sort() modifies in place and returns None
result = nums.sort()
print(result)  # None  (not the sorted list!)

# sorted() returns a new sorted list
nums = [3, 1, 2]
result = sorted(nums)
print(result)  # [1, 2, 3]
\`\`\`

### 4. Index out of range

\`\`\`python
a = [10, 20, 30]
# print(a[3])  # IndexError! Valid indices are 0, 1, 2
print(a[2])    # 30 (last valid index)
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-010',
        lessonId: lesson4.id,
        prompt: 'Create a list called `colors` with the values `"red"`, `"green"`, `"blue"`. Append `"yellow"` to the list, then print the list.',
        starterCode: '# Create the list, append "yellow", then print\n',
        expectedOutput: "['red', 'green', 'blue', 'yellow']",
        testCode: '',
        hints: JSON.stringify([
          'Create the list with: colors = ["red", "green", "blue"]',
          'Use colors.append("yellow") then print(colors)',
        ]),
        order: 1,
      },
      {
        id: 'exercise-011',
        lessonId: lesson4.id,
        prompt: 'Given `numbers = [5, 2, 8, 1, 9]`, print the minimum value, maximum value, and the sum on separate lines.',
        starterCode: 'numbers = [5, 2, 8, 1, 9]\n# Print min, max, and sum on separate lines\n',
        expectedOutput: '1\n9\n25',
        testCode: '',
        hints: JSON.stringify([
          'Use the built-in functions min(), max(), and sum().',
          'print(min(numbers))\nprint(max(numbers))\nprint(sum(numbers))',
        ]),
        order: 2,
      },
      {
        id: 'exercise-012',
        lessonId: lesson4.id,
        prompt: 'Create a tuple `coordinates = (10, 20)` and use tuple unpacking to assign the values to `x` and `y`. Print `x` and `y` on separate lines.',
        starterCode: '# Create the tuple and unpack it\n',
        expectedOutput: '10\n20',
        testCode: '',
        hints: JSON.stringify([
          'Tuple unpacking: x, y = (10, 20)',
          'Then print(x) and print(y) on separate lines.',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-010',
        lessonId: lesson4.id,
        question: 'Which of the following creates a valid single-element tuple?',
        type: 'MCQ',
        options: JSON.stringify(['(42)', '(42,)', '[42]', '{42}']),
        correctAnswer: '(42,)',
        explanation: 'A trailing comma is required for single-element tuples. Without it, (42) is just the integer 42 in parentheses.',
        order: 1,
      },
      {
        id: 'quiz-011',
        lessonId: lesson4.id,
        question: 'Lists in Python are mutable, meaning their elements can be changed after creation.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation: 'Lists are mutable — you can add, remove, and change elements in place. Tuples, by contrast, are immutable.',
        order: 2,
      },
      {
        id: 'quiz-012',
        lessonId: lesson4.id,
        question: 'What does `[1, 2, 3].pop()` return?',
        type: 'MCQ',
        options: JSON.stringify(['1', '3', '[1, 2]', 'None']),
        correctAnswer: '3',
        explanation: 'The pop() method with no argument removes and returns the last element of the list, which is 3.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 4: Lists & Tuples');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 5 — Dictionaries & Sets
  // ═══════════════════════════════════════════════════════════════════
  const lesson5 = await prisma.lesson.create({
    data: {
      id: 'lesson-005',
      moduleId: moduleDataStructures.id,
      title: 'Dictionaries & Sets: Key-Value Pairs & Set Operations',
      slug: 'dictionaries-and-sets',
      order: 2,
      content: `# Dictionaries & Sets

While lists and tuples store ordered sequences, **dictionaries** store key-value mappings and **sets** store unique unordered elements. These two data structures are workhorses in real-world Python.

---

## 1. Dictionaries

A dictionary (\`dict\`) maps **keys** to **values**:

\`\`\`python
student = {
    "name": "Alice",
    "age": 25,
    "grade": "A"
}
\`\`\`

### Accessing Values

\`\`\`python
print(student["name"])     # Alice
print(student.get("age"))  # 25
print(student.get("gpa", 0.0))  # 0.0 (default if key missing)
\`\`\`

Using \`[]\` raises a \`KeyError\` if the key does not exist; \`.get()\` returns \`None\` (or a default value) instead.

### Adding & Updating

\`\`\`python
student["email"] = "alice@example.com"  # add new key
student["age"] = 26                     # update existing key
print(student)
\`\`\`

### Removing Items

\`\`\`python
del student["grade"]          # remove by key
email = student.pop("email")  # remove and return value
\`\`\`

### Iterating

\`\`\`python
person = {"name": "Bob", "age": 30, "city": "NYC"}

# Keys
for key in person:
    print(key)

# Values
for value in person.values():
    print(value)

# Key-Value pairs
for key, value in person.items():
    print(f"{key}: {value}")
\`\`\`

### Dictionary Comprehensions

\`\`\`python
squares = {x: x ** 2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
\`\`\`

---

## 2. Sets

A **set** is an **unordered** collection of **unique** elements:

\`\`\`python
fruits = {"apple", "banana", "cherry"}
numbers = {1, 2, 3, 2, 1}  # duplicates removed
print(numbers)  # {1, 2, 3}
\`\`\`

### Creating Sets

\`\`\`python
empty_set = set()         # NOT {} — that creates an empty dict!
from_list = set([1, 2, 2, 3])
print(from_list)  # {1, 2, 3}
\`\`\`

### Set Operations

\`\`\`python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)   # Union: {1, 2, 3, 4, 5, 6}
print(a & b)   # Intersection: {3, 4}
print(a - b)   # Difference: {1, 2}
print(a ^ b)   # Symmetric difference: {1, 2, 5, 6}
\`\`\`

### Membership Testing

Sets provide **O(1)** membership testing, making them much faster than lists for \`in\` checks:

\`\`\`python
valid_codes = {100, 200, 301, 404, 500}
print(404 in valid_codes)  # True
print(999 in valid_codes)  # False
\`\`\`

### Set Methods

\`\`\`python
s = {1, 2, 3}
s.add(4)          # {1, 2, 3, 4}
s.discard(2)      # {1, 3, 4}  — no error if missing
s.remove(3)       # {1, 4}    — KeyError if missing
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Using \`{}\` to create an empty set

\`\`\`python
# Wrong — creates an empty dict
empty = {}
print(type(empty))  # <class 'dict'>

# Right
empty = set()
print(type(empty))  # <class 'set'>
\`\`\`

### 2. Using a mutable type as a dictionary key

\`\`\`python
# Wrong — lists are not hashable
# d = {[1, 2]: "value"}  # TypeError

# Right — use a tuple instead
d = {(1, 2): "value"}
\`\`\`

### 3. Assuming sets are ordered

\`\`\`python
# Sets do not guarantee order
s = {3, 1, 2}
# Iteration order may vary — do not rely on it
\`\`\`

### 4. Forgetting .get() and getting a KeyError

\`\`\`python
d = {"a": 1}
# print(d["b"])       # KeyError!
print(d.get("b", 0))  # 0 (safe default)
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-013',
        lessonId: lesson5.id,
        prompt: 'Create a dictionary called `person` with keys `"name"` (value `"Alice"`), `"age"` (value `30`), and `"city"` (value `"NYC"`). Print the value associated with the `"name"` key.',
        starterCode: '# Create the dictionary and print the name\n',
        expectedOutput: 'Alice',
        testCode: '',
        hints: JSON.stringify([
          'Use curly braces with key: value pairs: person = {"name": "Alice", "age": 30, "city": "NYC"}',
          'Access the value with person["name"] and print it.',
        ]),
        order: 1,
      },
      {
        id: 'exercise-014',
        lessonId: lesson5.id,
        prompt: 'Given two sets `a = {1, 2, 3, 4}` and `b = {3, 4, 5, 6}`, print their intersection (the elements that appear in both sets).',
        starterCode: 'a = {1, 2, 3, 4}\nb = {3, 4, 5, 6}\n# Print the intersection\n',
        expectedOutput: '{3, 4}',
        testCode: '',
        hints: JSON.stringify([
          'Use the & operator or the .intersection() method.',
          'print(a & b) will output {3, 4}',
        ]),
        order: 2,
      },
      {
        id: 'exercise-015',
        lessonId: lesson5.id,
        prompt: 'Create a dictionary comprehension that maps each number from 1 to 5 to its cube. Print the resulting dictionary.',
        starterCode: '# Create a dict comprehension {n: n**3 for n in range(...)}\n',
        expectedOutput: '{1: 1, 2: 8, 3: 27, 4: 64, 5: 125}',
        testCode: '',
        hints: JSON.stringify([
          'Use a dictionary comprehension: {n: n**3 for n in range(1, 6)}',
          'print({n: n**3 for n in range(1, 6)})',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-013',
        lessonId: lesson5.id,
        question: 'What does `{}` create in Python?',
        type: 'MCQ',
        options: JSON.stringify(['An empty set', 'An empty dictionary', 'An empty list', 'A syntax error']),
        correctAnswer: 'An empty dictionary',
        explanation: 'Curly braces {} create an empty dictionary, not a set. To create an empty set, use set().',
        order: 1,
      },
      {
        id: 'quiz-014',
        lessonId: lesson5.id,
        question: 'Sets in Python can contain duplicate elements.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation: 'Sets automatically remove duplicates. Each element in a set must be unique.',
        order: 2,
      },
      {
        id: 'quiz-015',
        lessonId: lesson5.id,
        question: 'What is the time complexity of checking membership (`in`) for a set?',
        type: 'MCQ',
        options: JSON.stringify(['O(1) average', 'O(n)', 'O(log n)', 'O(n^2)']),
        correctAnswer: 'O(1) average',
        explanation: 'Sets use a hash table internally, so membership testing is O(1) on average, compared to O(n) for lists.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 5: Dictionaries & Sets');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 6 — Control Flow
  // ═══════════════════════════════════════════════════════════════════
  const lesson6 = await prisma.lesson.create({
    data: {
      id: 'lesson-006',
      moduleId: moduleDataStructures.id,
      title: 'Control Flow: if / elif / else & Ternary Expressions',
      slug: 'control-flow',
      order: 3,
      content: `# Control Flow: if / elif / else

Programs often need to make decisions. Python's **conditional statements** let you execute different blocks of code depending on whether conditions are true or false.

---

## 1. The \`if\` Statement

\`\`\`python
age = 18

if age >= 18:
    print("You are an adult.")
\`\`\`

Key syntax rules:
- The condition ends with a **colon** (\`:\`)
- The body is **indented** (4 spaces by convention)
- No parentheses are required around the condition (unlike C/Java)

---

## 2. \`if / else\`

\`\`\`python
temperature = 35

if temperature > 30:
    print("It's hot outside!")
else:
    print("The weather is nice.")
\`\`\`

Output: \`It's hot outside!\`

---

## 3. \`if / elif / else\`

When you have multiple branches, use \`elif\` (short for "else if"):

\`\`\`python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Your grade: {grade}")  # Your grade: B
\`\`\`

Python evaluates conditions **top to bottom** and runs the **first** block whose condition is true. If none match, the \`else\` block runs.

---

## 4. Comparison & Logical Operators

### Comparison Operators

| Operator | Meaning |
|----------|---------|
| \`==\` | Equal to |
| \`!=\` | Not equal to |
| \`<\` | Less than |
| \`>\` | Greater than |
| \`<=\` | Less than or equal |
| \`>=\` | Greater than or equal |

### Logical Operators

\`\`\`python
age = 25
has_id = True

if age >= 18 and has_id:
    print("Entry allowed")

if age < 13 or age > 65:
    print("Discounted ticket")

if not has_id:
    print("ID required")
\`\`\`

---

## 5. The Ternary (Conditional) Expression

Python supports a one-line conditional often called the **ternary operator**:

\`\`\`python
age = 20
status = "adult" if age >= 18 else "minor"
print(status)  # adult
\`\`\`

This is equivalent to:

\`\`\`python
if age >= 18:
    status = "adult"
else:
    status = "minor"
\`\`\`

---

## 6. Truthiness in Conditions

You do not need to write \`== True\` or \`!= 0\`. Python treats certain values as "falsy":

\`\`\`python
# Falsy values: False, 0, 0.0, "", [], {}, set(), None
items = []
if items:
    print("List has items")
else:
    print("List is empty")  # This runs
\`\`\`

---

## 7. Nested Conditions

\`\`\`python
user_role = "admin"
is_active = True

if is_active:
    if user_role == "admin":
        print("Welcome, admin!")
    else:
        print("Welcome, user!")
else:
    print("Account is inactive.")
\`\`\`

While nesting works, flat is usually better than nested. Prefer \`elif\` or combine conditions with \`and\`/\`or\` when possible.`,
      commonMistakes: `## Common Mistakes

### 1. Using \`=\` instead of \`==\` in a condition

\`\`\`python
x = 5
# Wrong — this is assignment, not comparison
# if x = 5:  # SyntaxError

# Right
if x == 5:
    print("x is five")
\`\`\`

### 2. Forgetting the colon

\`\`\`python
# Wrong
# if x > 0
#     print("positive")

# Right
if x > 0:
    print("positive")
\`\`\`

### 3. Inconsistent indentation

\`\`\`python
# Wrong — mixed spaces and tabs or wrong indent level
if True:
    print("line 1")
      print("line 2")  # IndentationError

# Right
if True:
    print("line 1")
    print("line 2")
\`\`\`

### 4. Writing \`if x == True\` instead of \`if x\`

\`\`\`python
flag = True
# Unnecessary
if flag == True:
    pass

# Pythonic
if flag:
    pass
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-016',
        lessonId: lesson6.id,
        prompt: 'Write a program that checks the variable `number = 7`. If `number` is positive, print `"Positive"`. If it is negative, print `"Negative"`. If it is zero, print `"Zero"`.',
        starterCode: 'number = 7\n# Write if/elif/else to check positive, negative, or zero\n',
        expectedOutput: 'Positive',
        testCode: '',
        hints: JSON.stringify([
          'Use if number > 0, elif number < 0, else for the three cases.',
          'if number > 0:\n    print("Positive")\nelif number < 0:\n    print("Negative")\nelse:\n    print("Zero")',
        ]),
        order: 1,
      },
      {
        id: 'exercise-017',
        lessonId: lesson6.id,
        prompt: 'Use a ternary expression to set `label` to `"even"` if `n` is even, or `"odd"` if `n` is odd. Then print `label`. Use `n = 4`.',
        starterCode: 'n = 4\n# Use a ternary expression\n',
        expectedOutput: 'even',
        testCode: '',
        hints: JSON.stringify([
          'The ternary syntax is: value_if_true if condition else value_if_false',
          'label = "even" if n % 2 == 0 else "odd"\nprint(label)',
        ]),
        order: 2,
      },
      {
        id: 'exercise-018',
        lessonId: lesson6.id,
        prompt: 'Given `score = 72`, print the letter grade using this scale: A (>=90), B (>=80), C (>=70), D (>=60), F (below 60).',
        starterCode: 'score = 72\n# Determine and print the letter grade\n',
        expectedOutput: 'C',
        testCode: '',
        hints: JSON.stringify([
          'Use if/elif/else with the thresholds 90, 80, 70, 60.',
          'if score >= 90:\n    print("A")\nelif score >= 80:\n    print("B")\nelif score >= 70:\n    print("C")\nelif score >= 60:\n    print("D")\nelse:\n    print("F")',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-016',
        lessonId: lesson6.id,
        question: 'What is the output of: `print("yes" if 0 else "no")`?',
        type: 'MCQ',
        options: JSON.stringify(['yes', 'no', '0', 'Error']),
        correctAnswer: 'no',
        explanation: '0 is a falsy value in Python. Since the condition is falsy, the else branch is selected and "no" is printed.',
        order: 1,
      },
      {
        id: 'quiz-017',
        lessonId: lesson6.id,
        question: 'In Python, `elif` is short for "else if".',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation: 'elif is Python\'s syntax for "else if". It allows chaining multiple conditions without deep nesting.',
        order: 2,
      },
      {
        id: 'quiz-018',
        lessonId: lesson6.id,
        question: 'Which of the following is a falsy value in Python?',
        type: 'MCQ',
        options: JSON.stringify(['"False"', '1', '[]', '[0]']),
        correctAnswer: '[]',
        explanation: 'An empty list [] is falsy. The string "False" is truthy (non-empty string), 1 is truthy, and [0] is truthy (non-empty list).',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 6: Control Flow');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 7 — Loops
  // ═══════════════════════════════════════════════════════════════════
  const lesson7 = await prisma.lesson.create({
    data: {
      id: 'lesson-007',
      moduleId: moduleDataStructures.id,
      title: 'Loops: for, while, range, enumerate, break & continue',
      slug: 'loops',
      order: 4,
      content: `# Loops

Loops let you repeat a block of code multiple times. Python has two loop types: \`for\` (iterate over a sequence) and \`while\` (repeat while a condition is true).

---

## 1. The \`for\` Loop

\`\`\`python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
\`\`\`

Output:

\`\`\`
apple
banana
cherry
\`\`\`

The variable \`fruit\` takes each value from the list in order.

---

## 2. \`range()\`

\`range()\` generates a sequence of numbers — perfect for counted loops:

\`\`\`python
# range(stop) — 0 to stop-1
for i in range(5):
    print(i, end=" ")
# 0 1 2 3 4

print()  # newline

# range(start, stop)
for i in range(2, 6):
    print(i, end=" ")
# 2 3 4 5

print()

# range(start, stop, step)
for i in range(0, 10, 2):
    print(i, end=" ")
# 0 2 4 6 8
\`\`\`

---

## 3. \`enumerate()\`

When you need both the **index** and the **value**, use \`enumerate()\`:

\`\`\`python
colors = ["red", "green", "blue"]
for index, color in enumerate(colors):
    print(f"{index}: {color}")
\`\`\`

Output:

\`\`\`
0: red
1: green
2: blue
\`\`\`

You can set a custom start index:

\`\`\`python
for i, color in enumerate(colors, start=1):
    print(f"{i}. {color}")
\`\`\`

Output:

\`\`\`
1. red
2. green
3. blue
\`\`\`

---

## 4. The \`while\` Loop

A \`while\` loop repeats as long as its condition is true:

\`\`\`python
count = 0
while count < 5:
    print(count, end=" ")
    count += 1
# 0 1 2 3 4
\`\`\`

**Warning:** If you forget to update the condition variable, you create an **infinite loop**.

---

## 5. \`break\` and \`continue\`

- **\`break\`** — exits the loop immediately
- **\`continue\`** — skips the rest of the current iteration

\`\`\`python
# break example — stop at first even number
for n in [1, 3, 4, 7, 8]:
    if n % 2 == 0:
        print(f"First even: {n}")
        break
# First even: 4

# continue example — skip odd numbers
for n in range(6):
    if n % 2 != 0:
        continue
    print(n, end=" ")
# 0 2 4
\`\`\`

---

## 6. Nested Loops

\`\`\`python
for i in range(3):
    for j in range(3):
        print(f"({i},{j})", end=" ")
    print()
\`\`\`

Output:

\`\`\`
(0,0) (0,1) (0,2)
(1,0) (1,1) (1,2)
(2,0) (2,1) (2,2)
\`\`\`

---

## 7. Loop \`else\` Clause

Python uniquely supports an \`else\` on loops that runs only if the loop completed **without** a \`break\`:

\`\`\`python
for n in [1, 3, 5]:
    if n % 2 == 0:
        print("Found even!")
        break
else:
    print("No even numbers found.")
# No even numbers found.
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Infinite loops

\`\`\`python
# Danger! count never changes
count = 0
while count < 5:
    print(count)
    # Forgot: count += 1
\`\`\`

Always ensure the loop condition will eventually become false.

### 2. Modifying a list while iterating over it

\`\`\`python
nums = [1, 2, 3, 4, 5]
# Wrong — skips elements
for n in nums:
    if n % 2 == 0:
        nums.remove(n)

# Right — iterate over a copy
for n in nums[:]:
    if n % 2 == 0:
        nums.remove(n)
\`\`\`

### 3. Off-by-one with range

\`\`\`python
# Want to print 1 to 5
for i in range(5):    # prints 0-4
    print(i)

for i in range(1, 6): # prints 1-5 (correct)
    print(i)
\`\`\`

### 4. Forgetting that range() is exclusive at the stop

\`\`\`python
list(range(1, 5))  # [1, 2, 3, 4] — not [1, 2, 3, 4, 5]
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-019',
        lessonId: lesson7.id,
        prompt: 'Use a `for` loop and `range()` to print the numbers 1 through 5, each on its own line.',
        starterCode: '# Print numbers 1 to 5 using a for loop\n',
        expectedOutput: '1\n2\n3\n4\n5',
        testCode: '',
        hints: JSON.stringify([
          'range(1, 6) generates 1, 2, 3, 4, 5.',
          'for i in range(1, 6):\n    print(i)',
        ]),
        order: 1,
      },
      {
        id: 'exercise-020',
        lessonId: lesson7.id,
        prompt: 'Given `animals = ["cat", "dog", "fish"]`, use `enumerate()` (starting at 1) to print each animal with its number, in the format `1. cat`, `2. dog`, `3. fish`.',
        starterCode: 'animals = ["cat", "dog", "fish"]\n# Use enumerate with start=1\n',
        expectedOutput: '1. cat\n2. dog\n3. fish',
        testCode: '',
        hints: JSON.stringify([
          'enumerate(animals, start=1) gives you pairs of (index, value) starting at 1.',
          'for i, animal in enumerate(animals, start=1):\n    print(f"{i}. {animal}")',
        ]),
        order: 2,
      },
      {
        id: 'exercise-021',
        lessonId: lesson7.id,
        prompt: 'Use a `while` loop to print the numbers 10, 8, 6, 4, 2 (counting down by 2), each on its own line.',
        starterCode: '# Use a while loop to count down from 10 to 2 by 2\n',
        expectedOutput: '10\n8\n6\n4\n2',
        testCode: '',
        hints: JSON.stringify([
          'Start with n = 10 and keep looping while n >= 2, decrementing by 2 each time.',
          'n = 10\nwhile n >= 2:\n    print(n)\n    n -= 2',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-019',
        lessonId: lesson7.id,
        question: 'What does `range(3)` produce?',
        type: 'MCQ',
        options: JSON.stringify(['[1, 2, 3]', '[0, 1, 2]', '[0, 1, 2, 3]', '[3]']),
        correctAnswer: '[0, 1, 2]',
        explanation: 'range(3) generates the sequence 0, 1, 2. It starts at 0 by default and stops before 3.',
        order: 1,
      },
      {
        id: 'quiz-020',
        lessonId: lesson7.id,
        question: 'The `break` statement exits the current loop immediately.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation: 'break immediately terminates the innermost enclosing loop and execution continues with the statement after the loop.',
        order: 2,
      },
      {
        id: 'quiz-021',
        lessonId: lesson7.id,
        question: 'What does `continue` do inside a loop?',
        type: 'MCQ',
        options: JSON.stringify([
          'Exits the loop entirely',
          'Skips the rest of the current iteration and goes to the next',
          'Restarts the loop from the beginning',
          'Pauses the loop for one second',
        ]),
        correctAnswer: 'Skips the rest of the current iteration and goes to the next',
        explanation: 'continue skips any remaining code in the current loop iteration and proceeds directly to the next iteration.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 7: Loops');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 8 — Functions
  // ═══════════════════════════════════════════════════════════════════
  const lesson8 = await prisma.lesson.create({
    data: {
      id: 'lesson-008',
      moduleId: moduleDataStructures.id,
      title: 'Functions: def, Arguments, Return, Scope & Lambda',
      slug: 'functions',
      order: 5,
      content: `# Functions

Functions let you package reusable blocks of logic. Instead of copying and pasting the same code, you define a function once and **call** it whenever you need it.

---

## 1. Defining a Function

\`\`\`python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Hello, Alice!
greet("Bob")    # Hello, Bob!
\`\`\`

- \`def\` introduces the function definition.
- \`name\` is a **parameter** (placeholder).
- \`"Alice"\` is an **argument** (actual value passed in).

---

## 2. Return Values

Functions can compute and **return** a value:

\`\`\`python
def add(a, b):
    return a + b

result = add(3, 4)
print(result)  # 7
\`\`\`

If a function has no \`return\` statement (or just \`return\` with no value), it returns \`None\`.

### Returning Multiple Values

Python functions can return multiple values via tuple packing:

\`\`\`python
def min_max(numbers):
    return min(numbers), max(numbers)

lo, hi = min_max([5, 2, 8, 1, 9])
print(lo, hi)  # 1 9
\`\`\`

---

## 3. Default Arguments

\`\`\`python
def power(base, exponent=2):
    return base ** exponent

print(power(3))     # 9  (exponent defaults to 2)
print(power(3, 3))  # 27
\`\`\`

**Important:** Default arguments are evaluated **once** at function definition time. Never use a mutable default like \`def f(items=[]):\`.

---

## 4. Keyword Arguments

You can pass arguments by name for clarity:

\`\`\`python
def describe(name, age, city):
    print(f"{name}, {age}, from {city}")

describe(name="Alice", city="NYC", age=30)
# Alice, 30, from NYC
\`\`\`

---

## 5. *args and **kwargs

\`\`\`python
# *args collects extra positional arguments into a tuple
def total(*numbers):
    return sum(numbers)

print(total(1, 2, 3))    # 6
print(total(10, 20))     # 30

# **kwargs collects extra keyword arguments into a dict
def show_info(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

show_info(name="Alice", age=30)
# name: Alice
# age: 30
\`\`\`

---

## 6. Scope

Variables defined **inside** a function are local to that function:

\`\`\`python
x = "global"

def f():
    x = "local"
    print(x)  # local

f()
print(x)  # global
\`\`\`

The **LEGB rule** describes Python's scope resolution order:
- **L**ocal — inside the current function
- **E**nclosing — in any enclosing function (closures)
- **G**lobal — at module level
- **B**uilt-in — Python's built-in names

---

## 7. Lambda Functions

A **lambda** is an anonymous (unnamed) function written as a single expression:

\`\`\`python
square = lambda x: x ** 2
print(square(5))  # 25

# Often used with map, filter, sorted
numbers = [3, 1, 4, 1, 5]
sorted_nums = sorted(numbers, key=lambda n: -n)
print(sorted_nums)  # [5, 4, 3, 1, 1]
\`\`\`

Lambdas are best for small throwaway functions. For anything complex, define a regular function with \`def\`.

---

## 8. Docstrings

Document what your function does with a docstring:

\`\`\`python
def area(radius):
    """Calculate the area of a circle given its radius."""
    import math
    return math.pi * radius ** 2

print(area.__doc__)
# Calculate the area of a circle given its radius.
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Mutable default arguments

\`\`\`python
# Wrong — the list is shared across calls!
def append_item(item, items=[]):
    items.append(item)
    return items

print(append_item(1))  # [1]
print(append_item(2))  # [1, 2]  (not [2]!)

# Right
def append_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
\`\`\`

### 2. Forgetting to return a value

\`\`\`python
def add(a, b):
    a + b  # This computes but doesn't return!

result = add(1, 2)
print(result)  # None
\`\`\`

### 3. Confusing print() and return

\`\`\`python
def add(a, b):
    print(a + b)  # Outputs but returns None

result = add(1, 2)  # prints 3
print(result)        # None
\`\`\`

### 4. Modifying a global variable without the global keyword

\`\`\`python
counter = 0

def increment():
    # counter += 1  # UnboundLocalError!
    global counter
    counter += 1
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-022',
        lessonId: lesson8.id,
        prompt: 'Write a function `double(n)` that returns `n * 2`. Call it with the argument `7` and print the result.',
        starterCode: '# Define double(n) and print double(7)\n',
        expectedOutput: '14',
        testCode: '',
        hints: JSON.stringify([
          'Define the function with def double(n): return n * 2',
          'Then call it: print(double(7))',
        ]),
        order: 1,
      },
      {
        id: 'exercise-023',
        lessonId: lesson8.id,
        prompt: 'Write a function `greet(name, greeting="Hello")` with a default parameter. Call it twice: once with `greet("Alice")` and once with `greet("Bob", "Hi")`. Print both results.',
        starterCode: '# Define greet with a default greeting parameter\n',
        expectedOutput: 'Hello, Alice!\nHi, Bob!',
        testCode: '',
        hints: JSON.stringify([
          'def greet(name, greeting="Hello"): print(f"{greeting}, {name}!")',
          'greet("Alice")  # uses default\ngreet("Bob", "Hi")  # overrides default',
        ]),
        order: 2,
      },
      {
        id: 'exercise-024',
        lessonId: lesson8.id,
        prompt: 'Use a lambda function with `sorted()` to sort the list `["banana", "apple", "cherry"]` by the **length** of each string (shortest first). Print the result.',
        starterCode: 'words = ["banana", "apple", "cherry"]\n# Sort by length using a lambda\n',
        expectedOutput: "['apple', 'banana', 'cherry']",
        testCode: '',
        hints: JSON.stringify([
          'sorted() has a key parameter: sorted(words, key=lambda w: len(w))',
          'print(sorted(words, key=lambda w: len(w)))',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-022',
        lessonId: lesson8.id,
        question: 'What does a function return if it has no `return` statement?',
        type: 'MCQ',
        options: JSON.stringify(['0', '""', 'None', 'False']),
        correctAnswer: 'None',
        explanation: 'If a function has no return statement, or uses return without a value, it implicitly returns None.',
        order: 1,
      },
      {
        id: 'quiz-023',
        lessonId: lesson8.id,
        question: 'Lambda functions can contain multiple statements.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation: 'Lambda functions are limited to a single expression. For multi-statement logic, use a regular def function.',
        order: 2,
      },
      {
        id: 'quiz-024',
        lessonId: lesson8.id,
        question: 'What does LEGB stand for in Python scope resolution?',
        type: 'MCQ',
        options: JSON.stringify([
          'Local, Enclosing, Global, Built-in',
          'Load, Execute, Generate, Build',
          'List, Element, Group, Block',
          'Lexical, Environment, Global, Base',
        ]),
        correctAnswer: 'Local, Enclosing, Global, Built-in',
        explanation: 'LEGB describes the order Python searches for variable names: Local scope first, then Enclosing function scopes, then Global (module) scope, and finally Built-in names.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 8: Functions');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 9 — Modules & Imports
  // ═══════════════════════════════════════════════════════════════════
  const lesson9 = await prisma.lesson.create({
    data: {
      id: 'lesson-009',
      moduleId: moduleIntermediatePython.id,
      title: 'Modules & Imports: Standard Library, pip & Virtual Envs',
      slug: 'modules-and-imports',
      order: 1,
      content: `# Modules & Imports

As your programs grow, you will want to split code into multiple files and use libraries written by others. Python's **module system** makes this straightforward.

---

## 1. What Is a Module?

A module is simply a \`.py\` file containing Python code. When you **import** it, you gain access to its functions, classes, and variables.

\`\`\`python
# math is a built-in module
import math
print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.141592653589793
\`\`\`

---

## 2. Import Styles

### Import the whole module

\`\`\`python
import math
print(math.ceil(4.2))  # 5
\`\`\`

### Import specific names

\`\`\`python
from math import sqrt, pi
print(sqrt(25))  # 5.0
print(pi)        # 3.141592653589793
\`\`\`

### Alias imports

\`\`\`python
import math as m
print(m.floor(4.9))  # 4
\`\`\`

### Import all (avoid in practice)

\`\`\`python
from math import *  # pollutes namespace — not recommended
\`\`\`

---

## 3. Useful Standard Library Modules

Python ships with "batteries included" — hundreds of modules ready to use.

| Module | Purpose | Example |
|--------|---------|---------|
| \`math\` | Math functions | \`math.sqrt(9)\` |
| \`random\` | Random numbers | \`random.randint(1, 10)\` |
| \`os\` | Operating system | \`os.getcwd()\` |
| \`sys\` | System info | \`sys.version\` |
| \`json\` | JSON encoding/decoding | \`json.loads(s)\` |
| \`datetime\` | Dates & times | \`datetime.date.today()\` |
| \`collections\` | Specialized containers | \`Counter, defaultdict\` |
| \`pathlib\` | File system paths | \`Path("file.txt").read_text()\` |

\`\`\`python
import random
print(random.choice(["heads", "tails"]))

import datetime
today = datetime.date.today()
print(today)

from collections import Counter
words = ["a", "b", "a", "c", "a", "b"]
print(Counter(words))  # Counter({'a': 3, 'b': 2, 'c': 1})
\`\`\`

---

## 4. Creating Your Own Modules

Create a file called \`helpers.py\`:

\`\`\`python
# helpers.py
def greet(name):
    return f"Hello, {name}!"

PI = 3.14159
\`\`\`

Then import it in another file:

\`\`\`python
# main.py
from helpers import greet, PI
print(greet("Alice"))  # Hello, Alice!
print(PI)              # 3.14159
\`\`\`

---

## 5. Installing Packages with pip

The **Python Package Index** (PyPI) hosts over 400,000 packages. Install them with \`pip\`:

\`\`\`bash
pip install requests
pip install numpy pandas matplotlib
\`\`\`

Then import and use:

\`\`\`python
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200
\`\`\`

### Useful pip commands

\`\`\`bash
pip list                  # list installed packages
pip show requests         # details about a package
pip freeze > requirements.txt  # export dependencies
pip install -r requirements.txt  # install from file
\`\`\`

---

## 6. Virtual Environments

A virtual environment is an **isolated** Python installation where you can install packages without affecting your system Python.

\`\`\`bash
# Create a virtual environment
python3 -m venv myenv

# Activate it
source myenv/bin/activate   # macOS/Linux
myenv\\Scripts\\activate       # Windows

# Now pip installs go into myenv/
pip install flask

# Deactivate
deactivate
\`\`\`

**Best practice:** Always use a virtual environment for each project.

---

## 7. The \`if __name__ == "__main__"\` Guard

\`\`\`python
# helpers.py
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    # This block runs only when the file is executed directly,
    # not when it is imported as a module.
    print(greet("World"))
\`\`\`

This pattern prevents code from running during imports.`,
      commonMistakes: `## Common Mistakes

### 1. Circular imports

\`\`\`python
# a.py imports b.py, and b.py imports a.py
# This can cause ImportError or partially loaded modules.
# Solution: restructure so the shared code lives in a third module.
\`\`\`

### 2. Shadowing a module name

\`\`\`python
# Don't name your file "math.py" or "random.py"!
# It will shadow the standard library module.
import math  # This would import YOUR math.py, not the built-in one.
\`\`\`

### 3. Installing packages globally without a virtual environment

\`\`\`bash
# Bad — may conflict with system packages
pip install flask

# Good — inside a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install flask
\`\`\`

### 4. Forgetting to activate the virtual environment

If you see "ModuleNotFoundError" after installing a package, check that your virtual environment is activated.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-025',
        lessonId: lesson9.id,
        prompt: 'Import the `math` module and use it to print the square root of 144.',
        starterCode: '# Import math and print the square root of 144\n',
        expectedOutput: '12.0',
        testCode: '',
        hints: JSON.stringify([
          'Use import math, then math.sqrt(144).',
          'import math\nprint(math.sqrt(144))',
        ]),
        order: 1,
      },
      {
        id: 'exercise-026',
        lessonId: lesson9.id,
        prompt: 'Import only `pi` and `ceil` from the `math` module. Print `pi` rounded up using `ceil`.',
        starterCode: '# Import pi and ceil from math, then print ceil(pi)\n',
        expectedOutput: '4',
        testCode: '',
        hints: JSON.stringify([
          'Use: from math import pi, ceil',
          'from math import pi, ceil\nprint(ceil(pi))',
        ]),
        order: 2,
      },
      {
        id: 'exercise-027',
        lessonId: lesson9.id,
        prompt: 'Use the `collections.Counter` class to count the letters in the list `["a", "b", "a", "c", "b", "a"]` and print the result.',
        starterCode: '# Use Counter to count the letters\nletters = ["a", "b", "a", "c", "b", "a"]\n',
        expectedOutput: "Counter({'a': 3, 'b': 2, 'c': 1})",
        testCode: '',
        hints: JSON.stringify([
          'Import Counter: from collections import Counter',
          'from collections import Counter\nprint(Counter(letters))',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-025',
        lessonId: lesson9.id,
        question: 'What command creates a Python virtual environment?',
        type: 'MCQ',
        options: JSON.stringify([
          'python -m venv myenv',
          'pip create myenv',
          'virtualenv --init myenv',
          'python --venv myenv',
        ]),
        correctAnswer: 'python -m venv myenv',
        explanation: 'The standard way to create a virtual environment in Python 3 is: python -m venv <directory_name>.',
        order: 1,
      },
      {
        id: 'quiz-026',
        lessonId: lesson9.id,
        question: 'Using `from math import *` is considered best practice.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation: 'Wildcard imports (from X import *) pollute the namespace and make it hard to tell where names come from. Prefer importing specific names.',
        order: 2,
      },
      {
        id: 'quiz-027',
        lessonId: lesson9.id,
        question: 'What does `if __name__ == "__main__":` guard against?',
        type: 'MCQ',
        options: JSON.stringify([
          'Code running when the file is imported as a module',
          'Syntax errors at runtime',
          'Missing module dependencies',
          'Running on the wrong Python version',
        ]),
        correctAnswer: 'Code running when the file is imported as a module',
        explanation: 'When a file is imported, __name__ is set to the module name (not "__main__"). The guard ensures certain code only runs when the file is executed directly.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 9: Modules & Imports');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 10 — File I/O & Exceptions
  // ═══════════════════════════════════════════════════════════════════
  const lesson10 = await prisma.lesson.create({
    data: {
      id: 'lesson-010',
      moduleId: moduleIntermediatePython.id,
      title: 'File I/O & Exceptions: open, read/write, try/except',
      slug: 'file-io-and-exceptions',
      order: 2,
      content: `# File I/O & Exceptions

Real programs need to read and write files, and they need to handle errors gracefully. This lesson covers both essential skills.

---

## 1. Opening Files

Use the built-in \`open()\` function:

\`\`\`python
# Basic syntax
file = open("data.txt", "r")  # "r" = read mode
content = file.read()
file.close()
\`\`\`

### File Modes

| Mode | Description |
|------|-------------|
| \`"r"\` | Read (default) — file must exist |
| \`"w"\` | Write — creates or overwrites |
| \`"a"\` | Append — adds to the end |
| \`"x"\` | Exclusive create — fails if file exists |
| \`"rb"\` / \`"wb"\` | Binary read / write |

---

## 2. The \`with\` Statement (Context Manager)

The preferred way to work with files is the \`with\` statement, which **automatically closes** the file:

\`\`\`python
with open("data.txt", "r") as file:
    content = file.read()
    print(content)
# file is automatically closed here
\`\`\`

---

## 3. Reading Files

### Read the entire file

\`\`\`python
with open("data.txt") as f:
    text = f.read()
    print(text)
\`\`\`

### Read line by line

\`\`\`python
with open("data.txt") as f:
    for line in f:
        print(line.strip())  # strip() removes the trailing newline
\`\`\`

### Read all lines into a list

\`\`\`python
with open("data.txt") as f:
    lines = f.readlines()
    print(lines)  # ['line1\\n', 'line2\\n', ...]
\`\`\`

---

## 4. Writing Files

\`\`\`python
# Write (overwrites existing content)
with open("output.txt", "w") as f:
    f.write("Hello, file!\\n")
    f.write("Second line.\\n")

# Append
with open("output.txt", "a") as f:
    f.write("Third line.\\n")
\`\`\`

### Writing multiple lines

\`\`\`python
lines = ["apple\\n", "banana\\n", "cherry\\n"]
with open("fruits.txt", "w") as f:
    f.writelines(lines)
\`\`\`

---

## 5. Exception Handling with try/except

Errors happen — files might not exist, users might enter bad data. Python uses **exceptions** for error handling:

\`\`\`python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
\`\`\`

Output: \`Cannot divide by zero!\`

### Catching multiple exceptions

\`\`\`python
try:
    number = int("abc")
except ValueError:
    print("Invalid number format!")
except TypeError:
    print("Wrong type!")
\`\`\`

### The full try/except/else/finally block

\`\`\`python
try:
    f = open("data.txt")
    content = f.read()
except FileNotFoundError:
    print("File not found!")
else:
    # Runs only if no exception occurred
    print(f"Read {len(content)} characters")
finally:
    # Always runs — cleanup code
    print("Done.")
\`\`\`

---

## 6. Common Built-in Exceptions

| Exception | When It Occurs |
|-----------|---------------|
| \`FileNotFoundError\` | File does not exist |
| \`ValueError\` | Wrong value (e.g., \`int("abc")\`) |
| \`TypeError\` | Wrong type (e.g., \`"a" + 1\`) |
| \`KeyError\` | Dict key not found |
| \`IndexError\` | List index out of range |
| \`ZeroDivisionError\` | Division by zero |
| \`AttributeError\` | Object has no such attribute |
| \`ImportError\` | Module cannot be imported |

---

## 7. Raising Exceptions

You can raise your own exceptions:

\`\`\`python
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

try:
    set_age(-5)
except ValueError as e:
    print(e)  # Age cannot be negative
\`\`\`

---

## 8. Practical Example: Reading a CSV Manually

\`\`\`python
data = []
try:
    with open("scores.csv") as f:
        for line in f:
            name, score = line.strip().split(",")
            data.append({"name": name, "score": int(score)})
    print(f"Loaded {len(data)} records")
except FileNotFoundError:
    print("scores.csv not found — using empty dataset")
    data = []
except ValueError:
    print("Error parsing a line in the CSV")
\`\`\``,
      commonMistakes: `## Common Mistakes

### 1. Forgetting to close a file

\`\`\`python
# Bad — file may stay open if an error occurs
f = open("data.txt")
content = f.read()
f.close()

# Good — with statement guarantees closing
with open("data.txt") as f:
    content = f.read()
\`\`\`

### 2. Catching too broad an exception

\`\`\`python
# Bad — hides all errors including bugs
try:
    result = do_something()
except:
    pass  # Silently swallows everything

# Better — catch specific exceptions
try:
    result = do_something()
except ValueError as e:
    print(f"Value error: {e}")
\`\`\`

### 3. Writing without specifying mode "w" or "a"

\`\`\`python
# This opens in read mode by default — writing will fail
# with open("out.txt") as f:
#     f.write("test")  # io.UnsupportedOperation

# Right
with open("out.txt", "w") as f:
    f.write("test")
\`\`\`

### 4. Forgetting that "w" mode overwrites the file

\`\`\`python
# This erases everything in the file before writing!
with open("log.txt", "w") as f:
    f.write("new entry")

# Use "a" to append instead
with open("log.txt", "a") as f:
    f.write("new entry")
\`\`\`

### 5. Not stripping newlines when reading lines

\`\`\`python
with open("data.txt") as f:
    for line in f:
        # line includes '\\n' at the end!
        print(repr(line))  # 'hello\\n'
        print(line.strip())  # 'hello'
\`\`\``,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-028',
        lessonId: lesson10.id,
        prompt: 'Write a try/except block that attempts to convert the string `"hello"` to an integer. If a `ValueError` occurs, print `"Not a valid number"`. Otherwise, print the number.',
        starterCode: 'text = "hello"\n# Try to convert to int with error handling\n',
        expectedOutput: 'Not a valid number',
        testCode: '',
        hints: JSON.stringify([
          'Wrap int(text) in a try block and catch ValueError.',
          'try:\n    num = int(text)\n    print(num)\nexcept ValueError:\n    print("Not a valid number")',
        ]),
        order: 1,
      },
      {
        id: 'exercise-029',
        lessonId: lesson10.id,
        prompt: 'Write a try/except block that attempts to divide `10` by `0`. Catch the `ZeroDivisionError` and print `"Cannot divide by zero!"`. Include a `finally` block that prints `"Calculation attempted."`.',
        starterCode: '# Handle division by zero with try/except/finally\n',
        expectedOutput: 'Cannot divide by zero!\nCalculation attempted.',
        testCode: '',
        hints: JSON.stringify([
          'Use try: 10 / 0, except ZeroDivisionError: print(...), finally: print(...)',
          'try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print("Cannot divide by zero!")\nfinally:\n    print("Calculation attempted.")',
        ]),
        order: 2,
      },
      {
        id: 'exercise-030',
        lessonId: lesson10.id,
        prompt: 'Write a function `safe_divide(a, b)` that returns `a / b` if `b` is not zero, otherwise raises a `ValueError` with the message `"Division by zero is not allowed"`. Call it with `safe_divide(10, 0)` inside a try/except and print the error message.',
        starterCode: '# Define safe_divide and test it\n',
        expectedOutput: 'Division by zero is not allowed',
        testCode: '',
        hints: JSON.stringify([
          'In the function, check if b == 0 and raise ValueError("Division by zero is not allowed").',
          'def safe_divide(a, b):\n    if b == 0:\n        raise ValueError("Division by zero is not allowed")\n    return a / b\n\ntry:\n    safe_divide(10, 0)\nexcept ValueError as e:\n    print(e)',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-028',
        lessonId: lesson10.id,
        question: 'Which statement automatically closes a file when the block ends?',
        type: 'MCQ',
        options: JSON.stringify(['with', 'try', 'finally', 'using']),
        correctAnswer: 'with',
        explanation: 'The with statement (context manager) automatically calls close() on the file when the block exits, even if an exception occurs.',
        order: 1,
      },
      {
        id: 'quiz-029',
        lessonId: lesson10.id,
        question: 'The `finally` block runs only when an exception occurs.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation: 'The finally block runs regardless of whether an exception occurred or not. It is used for cleanup code that must always execute.',
        order: 2,
      },
      {
        id: 'quiz-030',
        lessonId: lesson10.id,
        question: 'What happens when you open a file with mode `"w"` and the file already exists?',
        type: 'MCQ',
        options: JSON.stringify([
          'The existing content is preserved and new content is appended',
          'The existing content is erased and the file is overwritten',
          'A FileExistsError is raised',
          'The file is opened in read-only mode',
        ]),
        correctAnswer: 'The existing content is erased and the file is overwritten',
        explanation: 'Write mode ("w") truncates the file to zero length before writing. Use "a" (append) if you want to keep existing content.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 10: File I/O & Exceptions');

  console.log('Part 1 complete. Starting Part 2...');

  // ═══════════════════════════════════════════════════════════════════
  //  MODULES 4 & 5
  // ═══════════════════════════════════════════════════════════════════
  const moduleDataScience = await prisma.module.create({
    data: { id: 'module-004', stageId: stageB.id, title: 'Data Science Essentials', slug: 'data-science-essentials', description: 'Master NumPy, Pandas, and Matplotlib for data analysis and visualization.', order: 2 },
  });
  const moduleML = await prisma.module.create({
    data: { id: 'module-005', stageId: stageC.id, title: 'Machine Learning Fundamentals', slug: 'ml-fundamentals', description: 'Build, evaluate, and improve ML models with scikit-learn.', order: 1 },
  });
  console.log('Created modules 4 & 5');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 11 — NumPy Basics
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-011', moduleId: moduleDataScience.id, title: 'NumPy Basics: Arrays, Operations & Broadcasting', slug: 'numpy-basics', order: 1,
      content: `# NumPy Basics\n\nNumPy is the foundation of scientific computing in Python. It provides powerful N-dimensional arrays and vectorized operations.\n\n---\n\n## 1. Why NumPy?\n\nPython lists are flexible but slow for math. NumPy arrays use contiguous memory and optimized C code.\n\n\`\`\`python\nimport numpy as np\narr = np.array([1, 2, 3, 4, 5])\nprint(type(arr))   # <class 'numpy.ndarray'>\nprint(arr.dtype)   # int64\n\`\`\`\n\n---\n\n## 2. Creating Arrays\n\n\`\`\`python\nnp.zeros(5)            # [0. 0. 0. 0. 0.]\nnp.ones((2, 3))        # 2x3 matrix of ones\nnp.arange(0, 10, 2)    # [0 2 4 6 8]\nnp.linspace(0, 1, 5)   # [0. 0.25 0.5 0.75 1.]\n\`\`\`\n\n---\n\n## 3. Properties\n\n\`\`\`python\narr = np.array([[1,2,3],[4,5,6]])\nprint(arr.shape)  # (2, 3)\nprint(arr.ndim)   # 2\nprint(arr.size)   # 6\n\`\`\`\n\n---\n\n## 4. Vectorized Operations\n\n\`\`\`python\na = np.array([1, 2, 3, 4])\nprint(a + 10)   # [11 12 13 14]\nprint(a * 2)    # [2 4 6 8]\nprint(a ** 2)   # [ 1  4  9 16]\n\`\`\`\n\n---\n\n## 5. Indexing & Slicing\n\n\`\`\`python\narr = np.array([10, 20, 30, 40, 50])\nprint(arr[1:4])   # [20 30 40]\nmat = np.array([[1,2],[3,4],[5,6]])\nprint(mat[:, 0])  # [1 3 5]  first column\n\`\`\`\n\n---\n\n## 6. Broadcasting\n\nNumPy can operate on arrays of different shapes:\n\n\`\`\`python\na = np.array([[1,2,3],[4,5,6]])\nb = np.array([10, 20, 30])\nprint(a + b)  # [[11 22 33] [14 25 36]]\n\`\`\`\n\n---\n\n## 7. Useful Functions\n\n\`\`\`python\narr = np.array([3, 1, 4, 1, 5, 9])\nprint(np.sum(arr))   # 23\nprint(np.mean(arr))  # 3.833...\nprint(np.max(arr))   # 9\nprint(np.sort(arr))  # [1 1 3 4 5 9]\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### 1. Forgetting that slices are views\n\`\`\`python\na = np.array([1,2,3])\nb = a[0:2]  # VIEW not copy\nb[0] = 99\nprint(a)  # [99 2 3] — a changed!\n# Use .copy() to avoid this\n\`\`\`\n\n### 2. Shape mismatch in operations\nArrays must be broadcast-compatible for element-wise operations.`,
      exercises: { create: [
        { id: 'exercise-031', prompt: 'Create a NumPy array [1,2,3,4,5], multiply each element by 3, and print the result.', starterCode: 'import numpy as np\n\n# Create array, multiply by 3, print\n', expectedOutput: '[ 3  6  9 12 15]', testCode: '', hints: JSON.stringify(['Use np.array([1,2,3,4,5]) * 3', 'arr = np.array([1,2,3,4,5])\nprint(arr * 3)']), order: 1 },
        { id: 'exercise-032', prompt: 'Create a 2x3 array of zeros and print its shape.', starterCode: 'import numpy as np\n\n# Create 2x3 zeros, print shape\n', expectedOutput: '(2, 3)', testCode: '', hints: JSON.stringify(['Use np.zeros((2,3))', 'arr = np.zeros((2,3))\nprint(arr.shape)']), order: 2 },
        { id: 'exercise-033', prompt: 'Create array [10,20,30,40,50] and print the mean.', starterCode: 'import numpy as np\n\n# Create array, print mean\n', expectedOutput: '30.0', testCode: '', hints: JSON.stringify(['Use np.mean()', 'arr = np.array([10,20,30,40,50])\nprint(np.mean(arr))']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-031', question: 'What is the main advantage of NumPy arrays over Python lists?', type: 'MCQ', options: JSON.stringify(['They can hold strings','Faster vectorized operations','More memory usage','Support nesting']), correctAnswer: 'Faster vectorized operations', explanation: 'NumPy uses contiguous memory and optimized C code for much faster numerical operations.', order: 1 },
        { id: 'quiz-032', question: 'NumPy array slices return copies of the data.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'False', explanation: 'Slices return views. Use .copy() for independent copies.', order: 2 },
        { id: 'quiz-033', question: 'What does np.linspace(0, 1, 5) return?', type: 'MCQ', options: JSON.stringify(['[0,1,2,3,4]','[0.0, 0.25, 0.5, 0.75, 1.0]','[0, 0.2, 0.4, 0.6, 0.8]','[0, 1, 5]']), correctAnswer: '[0.0, 0.25, 0.5, 0.75, 1.0]', explanation: 'linspace returns num evenly spaced values between start and stop inclusive.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 11');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 12 — Pandas Series & DataFrames
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-012', moduleId: moduleDataScience.id, title: 'Pandas Series & DataFrames', slug: 'pandas-series-dataframes', order: 2,
      content: `# Pandas Series & DataFrames\n\nPandas provides two key structures: **Series** (1D) and **DataFrame** (2D table).\n\n---\n\n## 1. Series\n\n\`\`\`python\nimport pandas as pd\ns = pd.Series([10, 20, 30], index=['a','b','c'])\nprint(s['b'])  # 20\n\`\`\`\n\n---\n\n## 2. DataFrame\n\n\`\`\`python\ndf = pd.DataFrame({\n    'name': ['Alice','Bob','Charlie'],\n    'age': [25, 30, 35],\n    'city': ['NYC','LA','Chicago']\n})\nprint(df)\n\`\`\`\n\n---\n\n## 3. Accessing Data\n\n\`\`\`python\ndf['name']           # single column (Series)\ndf[['name','age']]   # multiple columns (DataFrame)\ndf.iloc[0]           # first row by position\ndf.loc[0, 'name']    # specific cell by label\n\`\`\`\n\n---\n\n## 4. Properties\n\n\`\`\`python\ndf.shape      # (rows, cols)\ndf.columns    # column names\ndf.dtypes     # data types\ndf.describe() # summary stats\n\`\`\`\n\n---\n\n## 5. Reading CSV\n\n\`\`\`python\ndf = pd.read_csv('data.csv')\ndf.head()\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### df['col'] vs df[['col']]\nSingle brackets → Series. Double brackets → DataFrame.\n\n### iloc vs loc\n- iloc: integer positions\n- loc: index labels`,
      exercises: { create: [
        { id: 'exercise-034', prompt: 'Create a Series with values [100,200,300] and index ["x","y","z"]. Print the value at "y".', starterCode: 'import pandas as pd\n\n# Create Series, print value at "y"\n', expectedOutput: '200', testCode: '', hints: JSON.stringify(['pd.Series([100,200,300], index=["x","y","z"])','s = pd.Series([100,200,300], index=["x","y","z"])\nprint(s["y"])']), order: 1 },
        { id: 'exercise-035', prompt: 'Create a DataFrame with "fruit" (["apple","banana","cherry"]) and "count" ([5,3,8]). Print the shape.', starterCode: 'import pandas as pd\n\n# Create DataFrame, print shape\n', expectedOutput: '(3, 2)', testCode: '', hints: JSON.stringify(['pd.DataFrame({"fruit": [...], "count": [...]})','df = pd.DataFrame({"fruit": ["apple","banana","cherry"], "count": [5,3,8]})\nprint(df.shape)']), order: 2 },
        { id: 'exercise-036', prompt: 'Use iloc to print the value at row 1, column 0.\ndf = pd.DataFrame({"a": [10,20,30], "b": [40,50,60]})', starterCode: 'import pandas as pd\ndf = pd.DataFrame({"a": [10,20,30], "b": [40,50,60]})\n# Print value at row 1, col 0\n', expectedOutput: '20', testCode: '', hints: JSON.stringify(['Use df.iloc[1, 0]','print(df.iloc[1, 0])']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-034', question: 'What does df.iloc[0] return?', type: 'MCQ', options: JSON.stringify(['The first column','The first row as a Series','The first cell','An error']), correctAnswer: 'The first row as a Series', explanation: 'iloc[0] returns the first row as a Series.', order: 1 },
        { id: 'quiz-035', question: 'A Pandas Series can only hold numeric data.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'False', explanation: 'Series can hold any type: int, float, string, bool, etc.', order: 2 },
        { id: 'quiz-036', question: 'Which function reads a CSV into a DataFrame?', type: 'MCQ', options: JSON.stringify(['pd.load_csv()','pd.read_csv()','pd.open_csv()','pd.import_csv()']), correctAnswer: 'pd.read_csv()', explanation: 'pd.read_csv() is the standard function for reading CSVs.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 12');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 13 — Pandas Data Wrangling
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-013', moduleId: moduleDataScience.id, title: 'Pandas Data Wrangling: Filtering, GroupBy & Merge', slug: 'pandas-wrangling', order: 3,
      content: `# Pandas Data Wrangling\n\nFilter rows, group data, aggregate, and join datasets.\n\n---\n\n## 1. Filtering\n\n\`\`\`python\nimport pandas as pd\ndf = pd.DataFrame({'name': ['Alice','Bob','Charlie'], 'score': [85, 92, 78]})\nhigh = df[df['score'] >= 85]\n# Multiple conditions: use & (AND), | (OR) with parentheses\nresult = df[(df['score'] >= 80) & (df['score'] <= 90)]\n\`\`\`\n\n---\n\n## 2. Sorting\n\n\`\`\`python\ndf.sort_values('score', ascending=False)\n\`\`\`\n\n---\n\n## 3. GroupBy\n\n\`\`\`python\ndf = pd.DataFrame({'dept': ['A','A','B','B'], 'salary': [50,60,70,80]})\nprint(df.groupby('dept')['salary'].mean())\n\`\`\`\n\n---\n\n## 4. Merge\n\n\`\`\`python\norders = pd.DataFrame({'id': [1,2], 'product': ['X','Y']})\nprices = pd.DataFrame({'id': [1,2], 'price': [10,20]})\nmerged = pd.merge(orders, prices, on='id')\n\`\`\`\n\n---\n\n## 5. Missing Data\n\n\`\`\`python\ndf.isnull().sum()       # count nulls\ndf.dropna()             # drop rows with nulls\ndf.fillna(0)            # fill nulls\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### Forgetting parentheses in multi-condition filters\n\`\`\`python\n# Wrong: df[df['a'] > 1 & df['b'] < 5]\n# Right: df[(df['a'] > 1) & (df['b'] < 5)]\n\`\`\``,
      exercises: { create: [
        { id: 'exercise-037', prompt: 'Filter df for rows where score >= 85 and print the count.\ndf has name=["Alice","Bob","Charlie"], score=[85, 92, 78]', starterCode: 'import pandas as pd\ndf = pd.DataFrame({"name": ["Alice","Bob","Charlie"], "score": [85, 92, 78]})\n# Filter and print count\n', expectedOutput: '2', testCode: '', hints: JSON.stringify(['df[df["score"] >= 85] then len()','print(len(df[df["score"] >= 85]))']), order: 1 },
        { id: 'exercise-038', prompt: 'Print the mean salary per dept using groupby.\ndf has dept=["A","A","B","B"], salary=[50,60,70,80]', starterCode: 'import pandas as pd\ndf = pd.DataFrame({"dept": ["A","A","B","B"], "salary": [50,60,70,80]})\n# Print mean salary by dept\n', expectedOutput: 'dept\nA    55.0\nB    75.0\nName: salary, dtype: float64', testCode: '', hints: JSON.stringify(['df.groupby("dept")["salary"].mean()','print(df.groupby("dept")["salary"].mean())']), order: 2 },
        { id: 'exercise-039', prompt: 'Merge df1 (id=[1,2,3], value=["a","b","c"]) and df2 (id=[1,2,3], score=[10,20,30]) on "id". Print number of columns.', starterCode: 'import pandas as pd\ndf1 = pd.DataFrame({"id": [1,2,3], "value": ["a","b","c"]})\ndf2 = pd.DataFrame({"id": [1,2,3], "score": [10,20,30]})\n# Merge and print column count\n', expectedOutput: '3', testCode: '', hints: JSON.stringify(['pd.merge(df1, df2, on="id")','print(len(pd.merge(df1, df2, on="id").columns))']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-037', question: 'Which operator is AND for Pandas filter conditions?', type: 'MCQ', options: JSON.stringify(['and','&&','&','+']), correctAnswer: '&', explanation: 'Use & for element-wise AND in Pandas boolean indexing.', order: 1 },
        { id: 'quiz-038', question: 'groupby().mean() returns a single value.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'False', explanation: 'It returns one mean value per group, not a single scalar.', order: 2 },
        { id: 'quiz-039', question: 'What does df.dropna() do?', type: 'MCQ', options: JSON.stringify(['Drops all columns','Drops rows with any null','Fills nulls with 0','Drops duplicates']), correctAnswer: 'Drops rows with any null', explanation: 'dropna() removes rows with any NaN values.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 13');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 14 — Matplotlib Basics
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-014', moduleId: moduleDataScience.id, title: 'Matplotlib Basics: Line, Bar, Scatter & Customization', slug: 'matplotlib-basics', order: 4,
      content: `# Matplotlib Basics\n\nMatplotlib is Python's most widely used plotting library.\n\n---\n\n## 1. Line Plot\n\n\`\`\`python\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\nx = [1, 2, 3, 4, 5]\ny = [2, 4, 6, 8, 10]\nplt.plot(x, y)\nplt.title('Line Plot')\nplt.xlabel('X')\nplt.ylabel('Y')\nplt.savefig('plot.png')\nprint("Plot saved!")\n\`\`\`\n\n---\n\n## 2. Bar Chart\n\n\`\`\`python\nplt.bar(['A','B','C'], [25, 40, 30])\nplt.title('Bar Chart')\nplt.savefig('bar.png')\n\`\`\`\n\n---\n\n## 3. Scatter Plot\n\n\`\`\`python\nimport numpy as np\nx = np.random.randn(50)\ny = x + np.random.randn(50) * 0.5\nplt.scatter(x, y, alpha=0.6)\nplt.savefig('scatter.png')\n\`\`\`\n\n---\n\n## 4. Customization\n\n\`\`\`python\nplt.plot(x, y, color='red', linewidth=2, linestyle='--', marker='o')\nplt.grid(True)\nplt.legend(['Data'])\nplt.tight_layout()\n\`\`\`\n\n---\n\n## 5. Subplots\n\n\`\`\`python\nfig, axes = plt.subplots(1, 2, figsize=(10, 4))\naxes[0].plot([1,2,3], [1,4,9])\naxes[1].bar(['a','b'], [3,7])\nplt.tight_layout()\nplt.savefig('subplots.png')\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### Forgetting plt.savefig() or plt.show()\nWithout one, the plot exists only in memory.\n\n### Plots accumulating\nCall plt.clf() or plt.figure() before each new plot.`,
      exercises: { create: [
        { id: 'exercise-040', prompt: 'Import matplotlib and define x=[1,2,3,4], y=[1,4,9,16]. Print "Plot ready".', starterCode: '# Setup plot data\n', expectedOutput: 'Plot ready', testCode: '', hints: JSON.stringify(['Just define the lists and print','import matplotlib\nmatplotlib.use("Agg")\nx = [1,2,3,4]\ny = [1,4,9,16]\nprint("Plot ready")']), order: 1 },
        { id: 'exercise-041', prompt: 'Given values = [10, 25, 15, 30, 20], print their sum.', starterCode: 'values = [10, 25, 15, 30, 20]\n# Print sum\n', expectedOutput: '100', testCode: '', hints: JSON.stringify(['Use sum(values)','print(sum(values))']), order: 2 },
        { id: 'exercise-042', prompt: 'Create np.linspace(0, 4, 5), square each value, print result.', starterCode: 'import numpy as np\n# linspace, square, print\n', expectedOutput: '[ 0.  1.  4.  9. 16.]', testCode: '', hints: JSON.stringify(['np.linspace(0,4,5) ** 2','x = np.linspace(0,4,5)\nprint(x**2)']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-040', question: 'Which function saves a matplotlib plot?', type: 'MCQ', options: JSON.stringify(['plt.save()','plt.savefig()','plt.export()','plt.write()']), correctAnswer: 'plt.savefig()', explanation: 'plt.savefig("file.png") saves the current figure.', order: 1 },
        { id: 'quiz-041', question: 'plt.scatter() creates a bar chart.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'False', explanation: 'scatter() creates scatter plots. Use plt.bar() for bar charts.', order: 2 },
        { id: 'quiz-042', question: 'What does plt.subplots(1, 2) create?', type: 'MCQ', options: JSON.stringify(['1 row, 2 columns of subplots','2 rows, 1 column','A single plot','12 subplots']), correctAnswer: '1 row, 2 columns of subplots', explanation: 'subplots(nrows, ncols) creates a grid of axes.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 14');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 15 — EDA
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-015', moduleId: moduleDataScience.id, title: 'Exploratory Data Analysis', slug: 'eda', order: 5,
      content: `# Exploratory Data Analysis (EDA)\n\nEDA investigates a dataset to discover patterns, spot anomalies, and form hypotheses.\n\n---\n\n## Workflow\n\n1. **Load** the data\n2. **Inspect**: shape, types, head/tail\n3. **Summarize**: describe()\n4. **Check**: missing values, duplicates\n5. **Visualize**: distributions, relationships\n6. **Clean**: transform as needed\n\n---\n\n## Missing Data\n\n\`\`\`python\nimport pandas as pd\ndf.isnull().sum()          # nulls per column\ndf.isnull().sum().sum()    # total nulls\ndf.isnull().mean() * 100   # percent missing\n\`\`\`\n\n---\n\n## Descriptive Stats\n\n\`\`\`python\ndf['price'].mean()\ndf['price'].median()\ndf['category'].value_counts()\ndf.corr()   # correlation matrix\n\`\`\`\n\n---\n\n## Value Counts\n\n\`\`\`python\ndf['color'].nunique()       # unique count\ndf['color'].value_counts()  # frequency table\n\`\`\`\n\n---\n\n## Visualization Ideas\n\n- Histogram: numeric distribution\n- Box plot: outliers\n- Scatter: two numerics\n- Bar chart: category frequencies\n- Heatmap: correlations`,
      commonMistakes: `## Common Mistakes\n\n### Skipping EDA\nAlways explore before modeling. Missing values or errors could ruin results.\n\n### Not checking dtypes\nA "numeric" column may be stored as strings.`,
      exercises: { create: [
        { id: 'exercise-043', prompt: 'Create df with "a"=[1,2,None,4], "b"=[None,2,3,4]. Print total null count.', starterCode: 'import pandas as pd\ndf = pd.DataFrame({"a": [1,2,None,4], "b": [None,2,3,4]})\n# Print total nulls\n', expectedOutput: '2', testCode: '', hints: JSON.stringify(['df.isnull().sum().sum()','print(df.isnull().sum().sum())']), order: 1 },
        { id: 'exercise-044', prompt: 'Given df with color=["red","blue","red","green","blue","red"], print value_counts().', starterCode: 'import pandas as pd\ndf = pd.DataFrame({"color": ["red","blue","red","green","blue","red"]})\n# Print value counts\n', expectedOutput: 'color\nred      3\nblue     2\ngreen    1\nName: count, dtype: int64', testCode: '', hints: JSON.stringify(['df["color"].value_counts()','print(df["color"].value_counts())']), order: 2 },
        { id: 'exercise-045', prompt: 'Create df with x=[10,20,30,40]. Print the median.', starterCode: 'import pandas as pd\ndf = pd.DataFrame({"x": [10,20,30,40]})\n# Print median\n', expectedOutput: '25.0', testCode: '', hints: JSON.stringify(['df["x"].median()','print(df["x"].median())']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-043', question: 'First step in EDA?', type: 'MCQ', options: JSON.stringify(['Build a model','Load and inspect data','Create visualizations','Remove outliers']), correctAnswer: 'Load and inspect data', explanation: 'Always start by loading and inspecting the data.', order: 1 },
        { id: 'quiz-044', question: 'df.describe() includes non-numeric columns by default.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'False', explanation: 'describe() only includes numeric columns by default. Use describe(include="all").', order: 2 },
        { id: 'quiz-045', question: 'Which method counts unique value frequencies?', type: 'MCQ', options: JSON.stringify(['unique()','nunique()','value_counts()','count()']), correctAnswer: 'value_counts()', explanation: 'value_counts() returns frequencies sorted descending.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 15');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 16 — Intro to ML
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-016', moduleId: moduleML.id, title: 'Introduction to Machine Learning', slug: 'intro-to-ml', order: 1,
      content: `# Introduction to Machine Learning\n\nML gets computers to learn patterns from data without explicit programming.\n\n---\n\n## Types of ML\n\n### Supervised Learning\nData has labels. Model predicts them.\n- **Classification**: categories (spam/not spam)\n- **Regression**: numbers (house price)\n\n### Unsupervised Learning\nNo labels. Find hidden structure.\n- Clustering, dimensionality reduction\n\n---\n\n## The ML Workflow\n\n1. Define the problem\n2. Collect & prepare data\n3. Choose a model\n4. Train\n5. Evaluate\n6. Tune\n7. Deploy\n\n---\n\n## Key Terms\n\n| Term | Meaning |\n|------|--------|\n| Feature | Input variable |\n| Target | Value to predict |\n| Training set | Data model learns from |\n| Test set | Held-out data for evaluation |\n| Overfitting | Memorizes training data |\n| Underfitting | Too simple |\n\n---\n\n## Train/Test Split\n\n\`\`\`python\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.25, random_state=42\n)\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### Training and testing on same data\nAlways split first. Testing on training data gives misleading accuracy.\n\n### Not shuffling before splitting\nSorted data leads to biased splits.`,
      exercises: { create: [
        { id: 'exercise-046', prompt: 'Create X=[[1],[2],...,[10]] and y=[2,4,...,20]. Print len(X).', starterCode: '# Create feature and target lists\n', expectedOutput: '10', testCode: '', hints: JSON.stringify(['Define the lists and print len(X)','X = [[i] for i in range(1,11)]\ny = [i*2 for i in range(1,11)]\nprint(len(X))']), order: 1 },
        { id: 'exercise-047', prompt: 'Count how many "cat" labels in y=["cat","dog","cat","cat","dog","cat","dog","dog"].', starterCode: 'y = ["cat","dog","cat","cat","dog","cat","dog","dog"]\n# Count cats\n', expectedOutput: '4', testCode: '', hints: JSON.stringify(['y.count("cat")','print(y.count("cat"))']), order: 2 },
        { id: 'exercise-048', prompt: 'Is predicting house prices Classification or Regression? Print the answer.', starterCode: '# Classification or Regression?\n', expectedOutput: 'Regression', testCode: '', hints: JSON.stringify(['Prices are continuous numbers','print("Regression")']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-046', question: 'Predicting spam/not spam is:', type: 'MCQ', options: JSON.stringify(['Regression','Classification','Clustering','Reinforcement']), correctAnswer: 'Classification', explanation: 'Predicting a category is classification.', order: 1 },
        { id: 'quiz-047', question: 'Overfitting means good training performance but poor test performance.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'True', explanation: 'Overfitting = memorizing training data, failing on new data.', order: 2 },
        { id: 'quiz-048', question: 'Purpose of a test set?', type: 'MCQ', options: JSON.stringify(['Train the model','Evaluate on unseen data','Store features','Clean data']), correctAnswer: 'Evaluate on unseen data', explanation: 'Test set gives unbiased performance estimate.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 16');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 17 — scikit-learn Basics
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-017', moduleId: moduleML.id, title: 'scikit-learn Basics: Estimator API', slug: 'sklearn-basics', order: 2,
      content: `# scikit-learn Basics\n\nsklearn provides a consistent API for dozens of ML algorithms.\n\n---\n\n## The Estimator API\n\n\`\`\`python\nfrom sklearn.linear_model import LinearRegression\nmodel = LinearRegression()        # Create\nmodel.fit(X_train, y_train)       # Train\npredictions = model.predict(X_test)  # Predict\nscore = model.score(X_test, y_test)  # Evaluate\n\`\`\`\n\n---\n\n## Train/Test Split\n\n\`\`\`python\nfrom sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\n\`\`\`\n\n---\n\n## Complete Example\n\n\`\`\`python\nimport numpy as np\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\n\nX = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])\ny = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 19.8])\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\nprint(f"Coefficient: {model.coef_[0]:.2f}")\nprint(f"R² Score: {model.score(X_test, y_test):.2f}")\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### Forgetting to fit before predict\nMust call fit() before predict().\n\n### Passing 1D arrays\nsklearn expects X as 2D: [[1],[2]], not [1,2].`,
      exercises: { create: [
        { id: 'exercise-049', prompt: 'Split X=[[1],[2],...,[10]], y=[1..10] with test_size=0.2, random_state=42. Print len(X_train).', starterCode: 'from sklearn.model_selection import train_test_split\nX = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]\ny = [1,2,3,4,5,6,7,8,9,10]\n# Split and print\n', expectedOutput: '8', testCode: '', hints: JSON.stringify(['train_test_split(X, y, test_size=0.2, random_state=42)','X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint(len(X_train))']), order: 1 },
        { id: 'exercise-050', prompt: 'Fit LinearRegression on X_train=[[1],[2],[3]], y_train=[2,4,6]. Print the coefficient rounded to 1 decimal.', starterCode: 'from sklearn.linear_model import LinearRegression\nX_train = [[1],[2],[3]]\ny_train = [2,4,6]\n# Fit and print coefficient\n', expectedOutput: '2.0', testCode: '', hints: JSON.stringify(['model.coef_[0]','model = LinearRegression()\nmodel.fit(X_train, y_train)\nprint(round(model.coef_[0], 1))']), order: 2 },
        { id: 'exercise-051', prompt: 'Using a fitted LinearRegression (fit on [[1],[2],[3]] → [2,4,6]), predict for [[5]] and print.', starterCode: 'from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit([[1],[2],[3]], [2,4,6])\n# Predict [[5]]\n', expectedOutput: '10.0', testCode: '', hints: JSON.stringify(['model.predict([[5]])[0]','print(model.predict([[5]])[0])']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-049', question: 'Correct sklearn API order?', type: 'MCQ', options: JSON.stringify(['predict→fit→score','fit→predict→score','score→fit→predict','fit→score→predict']), correctAnswer: 'fit→predict→score', explanation: 'fit, predict, then score.', order: 1 },
        { id: 'quiz-050', question: 'train_test_split shuffles data by default.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'True', explanation: 'shuffle=True is the default. Set shuffle=False to disable.', order: 2 },
        { id: 'quiz-051', question: 'What does random_state do?', type: 'MCQ', options: JSON.stringify(['Makes split random each time','Ensures reproducible splits','Determines test size','Disables shuffling']), correctAnswer: 'Ensures reproducible splits', explanation: 'Fixed random_state = same split every time.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 17');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 18 — Classification
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-018', moduleId: moduleML.id, title: 'Classification: Logistic Regression & Decision Trees', slug: 'classification', order: 3,
      content: `# Classification\n\nPredicts which category a data point belongs to.\n\n---\n\n## Logistic Regression\n\n\`\`\`python\nfrom sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\nprint(model.score(X_test, y_test))\n\`\`\`\n\n---\n\n## Decision Trees\n\n\`\`\`python\nfrom sklearn.tree import DecisionTreeClassifier\ntree = DecisionTreeClassifier(max_depth=3)\ntree.fit(X_train, y_train)\n\`\`\`\n\n---\n\n## Metrics\n\n\`\`\`python\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\ny_pred = model.predict(X_test)\nprint(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")\n\`\`\`\n\n---\n\n## Confusion Matrix\n\n| | Pred + | Pred - |\n|---|---|---|\n| **Actual +** | TP | FN |\n| **Actual -** | FP | TN |\n\n- **Precision** = TP/(TP+FP)\n- **Recall** = TP/(TP+FN)\n- **F1** = harmonic mean`,
      commonMistakes: `## Common Mistakes\n\n### Using accuracy on imbalanced data\nUse F1, precision, recall instead.\n\n### No max_depth on trees\nUnlimited depth = overfitting.`,
      exercises: { create: [
        { id: 'exercise-052', prompt: 'Fit LogisticRegression on X=[[1],[2],[3],[4]], y=[0,0,1,1]. Predict [[5]], print result.', starterCode: 'from sklearn.linear_model import LogisticRegression\nX_train = [[1],[2],[3],[4]]\ny_train = [0,0,1,1]\n# Fit and predict [[5]]\n', expectedOutput: '[1]', testCode: '', hints: JSON.stringify(['model.predict([[5]])','model = LogisticRegression()\nmodel.fit(X_train, y_train)\nprint(model.predict([[5]]))']), order: 1 },
        { id: 'exercise-053', prompt: 'Calculate accuracy of y_true=[1,0,1,1,0], y_pred=[1,0,0,1,0]. Print it.', starterCode: 'from sklearn.metrics import accuracy_score\ny_true = [1,0,1,1,0]\ny_pred = [1,0,0,1,0]\n# Print accuracy\n', expectedOutput: '0.8', testCode: '', hints: JSON.stringify(['accuracy_score(y_true, y_pred)','print(accuracy_score(y_true, y_pred))']), order: 2 },
        { id: 'exercise-054', prompt: 'Create DecisionTreeClassifier(max_depth=2, random_state=42). Fit on X=[[1],[2],[3],[4],[5],[6]], y=[0,0,0,1,1,1]. Predict [[3]].', starterCode: 'from sklearn.tree import DecisionTreeClassifier\nX = [[1],[2],[3],[4],[5],[6]]\ny = [0,0,0,1,1,1]\n# Fit and predict [[3]]\n', expectedOutput: '[0]', testCode: '', hints: JSON.stringify(['DecisionTreeClassifier(max_depth=2, random_state=42)','tree = DecisionTreeClassifier(max_depth=2, random_state=42)\ntree.fit(X, y)\nprint(tree.predict([[3]]))']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-052', question: 'Logistic regression is for regression problems.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'False', explanation: 'Despite the name, logistic regression is a classifier.', order: 1 },
        { id: 'quiz-053', question: 'Precision measures:', type: 'MCQ', options: JSON.stringify(['Of actual positives, how many predicted correctly','Of predicted positives, how many are actually positive','Overall correctness','Harmonic mean']), correctAnswer: 'Of predicted positives, how many are actually positive', explanation: 'Precision = TP/(TP+FP).', order: 2 },
        { id: 'quiz-054', question: 'Which parameter prevents decision tree overfitting?', type: 'MCQ', options: JSON.stringify(['random_state','max_depth','test_size','n_jobs']), correctAnswer: 'max_depth', explanation: 'max_depth limits tree growth.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 18');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 19 — Regression
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-019', moduleId: moduleML.id, title: 'Regression: Linear Regression & Pipelines', slug: 'regression', order: 4,
      content: `# Regression\n\nPredicts continuous numbers (price, temperature).\n\n---\n\n## Linear Regression\n\n\`\`\`python\nfrom sklearn.linear_model import LinearRegression\nimport numpy as np\nX = np.array([[1],[2],[3],[4],[5]])\ny = np.array([2.1, 3.9, 6.2, 7.8, 10.1])\nmodel = LinearRegression()\nmodel.fit(X, y)\nprint(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")\n\`\`\`\n\n---\n\n## Metrics\n\n\`\`\`python\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nprint(f"MSE:  {mean_squared_error(y_true, y_pred):.3f}")\nprint(f"MAE:  {mean_absolute_error(y_true, y_pred):.3f}")\nprint(f"R²:   {r2_score(y_true, y_pred):.3f}")\n\`\`\`\n\n---\n\n## Pipelines\n\n\`\`\`python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\npipe = Pipeline([\n    ('scaler', StandardScaler()),\n    ('model', LinearRegression())\n])\npipe.fit(X_train, y_train)\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### Not scaling features\nDifferent scales can hurt some algorithms. Use StandardScaler.\n\n### R² without context\nAlways check residuals and compare baselines.`,
      exercises: { create: [
        { id: 'exercise-055', prompt: 'Fit LinearRegression on X=[[1],[2],[3],[4],[5]], y=[2,4,6,8,10]. Print R² rounded to 1 decimal.', starterCode: 'from sklearn.linear_model import LinearRegression\nimport numpy as np\nX = np.array([[1],[2],[3],[4],[5]])\ny = np.array([2,4,6,8,10])\n# Fit and print R²\n', expectedOutput: '1.0', testCode: '', hints: JSON.stringify(['model.score(X, y)','model = LinearRegression()\nmodel.fit(X, y)\nprint(round(model.score(X, y), 1))']), order: 1 },
        { id: 'exercise-056', prompt: 'Calculate MAE of y_true=[3,5,7], y_pred=[2.5,5.5,7]. Print rounded to 2 decimals.', starterCode: 'from sklearn.metrics import mean_absolute_error\ny_true = [3,5,7]\ny_pred = [2.5, 5.5, 7]\n# Print MAE\n', expectedOutput: '0.33', testCode: '', hints: JSON.stringify(['mean_absolute_error(y_true, y_pred)','print(round(mean_absolute_error(y_true, y_pred), 2))']), order: 2 },
        { id: 'exercise-057', prompt: 'Create a Pipeline with StandardScaler and LinearRegression. Print len(pipe.steps).', starterCode: 'from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LinearRegression\n# Create pipeline\n', expectedOutput: '2', testCode: '', hints: JSON.stringify(['Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])','pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])\nprint(len(pipe.steps))']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-055', question: 'R² = 1.0 means:', type: 'MCQ', options: JSON.stringify(['Overfitting','Model explains all variance','100% accuracy','No noise']), correctAnswer: 'Model explains all variance', explanation: 'R²=1.0 = perfect prediction of target variance.', order: 1 },
        { id: 'quiz-056', question: 'RMSE has the same units as the target.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'True', explanation: 'RMSE is the square root of MSE, bringing error back to target units.', order: 2 },
        { id: 'quiz-057', question: 'Main benefit of Pipelines?', type: 'MCQ', options: JSON.stringify(['Faster training','Prevents data leakage by chaining steps','Reduces overfitting','Auto-selects models']), correctAnswer: 'Prevents data leakage by chaining steps', explanation: 'Pipelines ensure preprocessing is fit only on training data.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 19');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 20 — Feature Engineering & Model Selection
  // ═══════════════════════════════════════════════════════════════════
  await prisma.lesson.create({
    data: {
      id: 'lesson-020', moduleId: moduleML.id, title: 'Feature Engineering & Model Selection', slug: 'feature-engineering', order: 5,
      content: `# Feature Engineering & Model Selection\n\nTransform raw data into better features; find the best model.\n\n---\n\n## Encoding Categoricals\n\n### Label Encoding\n\`\`\`python\nfrom sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nencoded = le.fit_transform(['red','blue','green','red'])\nprint(encoded)  # [2 0 1 2]\n\`\`\`\n\n### One-Hot Encoding\n\`\`\`python\nfrom sklearn.preprocessing import OneHotEncoder\nimport numpy as np\nohe = OneHotEncoder(sparse_output=False)\nencoded = ohe.fit_transform(np.array([['red'],['blue'],['green']]))\n\`\`\`\n\n---\n\n## Feature Scaling\n\n### StandardScaler\n\`\`\`python\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n# mean=0, std=1\n\`\`\`\n\n---\n\n## Cross-Validation\n\n\`\`\`python\nfrom sklearn.model_selection import cross_val_score\nscores = cross_val_score(model, X, y, cv=5, scoring='r2')\nprint(f"Mean R²: {scores.mean():.2f}")\n\`\`\`\n\n---\n\n## GridSearchCV\n\n\`\`\`python\nfrom sklearn.model_selection import GridSearchCV\nparam_grid = {'max_depth': [2,3,5,10]}\ngrid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)\ngrid.fit(X_train, y_train)\nprint(grid.best_params_)\n\`\`\``,
      commonMistakes: `## Common Mistakes\n\n### Fitting scaler on test data\nAlways fit on train only, then transform both.\n\n### One-hot encoding high cardinality\n1000 unique values = 1000 columns. Use target encoding instead.`,
      exercises: { create: [
        { id: 'exercise-058', prompt: 'Use LabelEncoder on ["cat","dog","cat","bird"]. Print the encoded array.', starterCode: 'from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\nanimals = ["cat","dog","cat","bird"]\n# Encode and print\n', expectedOutput: '[1 2 1 0]', testCode: '', hints: JSON.stringify(['le.fit_transform(animals)','print(le.fit_transform(animals))']), order: 1 },
        { id: 'exercise-059', prompt: 'Scale X=[[1,10],[2,20],[3,30]] with StandardScaler. Print rounded mean.', starterCode: 'from sklearn.preprocessing import StandardScaler\nimport numpy as np\nX = [[1,10],[2,20],[3,30]]\n# Scale and print mean\n', expectedOutput: '[0. 0.]', testCode: '', hints: JSON.stringify(['scaler.fit_transform(X) then mean(axis=0)','scaler = StandardScaler()\nX_s = scaler.fit_transform(X)\nprint(np.round(X_s.mean(axis=0)))']), order: 2 },
        { id: 'exercise-060', prompt: 'How many folds in 5-fold cross-validation? Print the answer.', starterCode: '# Answer\n', expectedOutput: '5', testCode: '', hints: JSON.stringify(['5-fold = 5 folds','print(5)']), order: 3 },
      ] },
      quizQuestions: { create: [
        { id: 'quiz-058', question: 'One-hot encoding creates binary columns for each category.', type: 'TRUE_FALSE', options: JSON.stringify(['True','False']), correctAnswer: 'True', explanation: 'Each unique value gets its own 0/1 column.', order: 1 },
        { id: 'quiz-059', question: 'Cross-validation helps prevent:', type: 'MCQ', options: JSON.stringify(['Data leakage','Overfitting to a single split','Missing values','Feature scaling']), correctAnswer: 'Overfitting to a single split', explanation: 'Multiple splits give more robust evaluation.', order: 2 },
        { id: 'quiz-060', question: 'StandardScaler transforms features to have:', type: 'MCQ', options: JSON.stringify(['Mean=0, Std=1','Min=0, Max=1','Sum=1','Median=0']), correctAnswer: 'Mean=0, Std=1', explanation: 'z-score normalization: (x - mean) / std.', order: 3 },
      ] },
    },
  });
  console.log('Seeded Lesson 20');

  // ═══════════════════════════════════════════════════════════════════
  //  PROJECTS
  // ═══════════════════════════════════════════════════════════════════
  await prisma.project.createMany({
    data: [
      {
        id: 'project-001', title: 'CLI Calculator', slug: 'cli-calculator', stage: 'BEGINNER', order: 1,
        brief: 'Build a command-line calculator with basic operations and error handling.',
        requirements: JSON.stringify(['Support +, -, *, /', 'Handle division by zero', 'Accept user input', 'Format results to 2 decimals', 'Loop until "quit"']),
        stretchGoals: JSON.stringify(['Add exponentiation and modulo', 'Add calculation history', 'Support chained operations']),
        steps: JSON.stringify([
          { title: 'Set up main loop', description: 'Create while True loop with quit condition.' },
          { title: 'Parse input', description: 'Ask for two numbers and an operator.' },
          { title: 'Implement operations', description: 'Use if/elif for each operator.' },
          { title: 'Add error handling', description: 'Wrap in try/except for ValueError and ZeroDivisionError.' },
          { title: 'Format output', description: 'Use f-strings with :.2f formatting.' },
        ]),
        rubric: JSON.stringify([
          { criterion: 'Functionality', description: 'All four operations work' },
          { criterion: 'Error handling', description: 'Graceful handling of bad input' },
          { criterion: 'Code quality', description: 'Clean, readable code' },
        ]),
      },
      {
        id: 'project-002', title: 'Text File Analyzer', slug: 'text-file-analyzer', stage: 'BEGINNER', order: 2,
        brief: 'Read a text file and produce statistics: word count, line count, most frequent words.',
        requirements: JSON.stringify(['Read file with "with" statement', 'Count words, lines, characters', 'Find 5 most common words', 'Calculate avg word length', 'Handle FileNotFoundError']),
        stretchGoals: JSON.stringify(['Add sentence count', 'Ignore stop words', 'Export summary to file']),
        steps: JSON.stringify([
          { title: 'Read the file', description: 'Use with open() to read contents.' },
          { title: 'Count stats', description: 'Count lines, words, characters.' },
          { title: 'Word frequency', description: 'Use collections.Counter.' },
          { title: 'Average word length', description: 'Sum of lengths / word count.' },
          { title: 'Error handling', description: 'try/except for FileNotFoundError.' },
          { title: 'Display results', description: 'Print formatted summary.' },
        ]),
        rubric: JSON.stringify([
          { criterion: 'File handling', description: 'Uses context manager properly' },
          { criterion: 'Statistics', description: 'All stats calculated correctly' },
          { criterion: 'Error handling', description: 'Missing file handled gracefully' },
        ]),
      },
      {
        id: 'project-003', title: 'CSV Cleaner + Summary Report', slug: 'csv-cleaner', stage: 'DATA', order: 3,
        brief: 'Clean a messy CSV: handle missing values, remove duplicates, output a clean version with a summary.',
        requirements: JSON.stringify(['Read CSV with Pandas', 'Report missing values per column', 'Fill or drop missing values', 'Remove duplicates', 'Save cleaned CSV', 'Print before/after summary']),
        stretchGoals: JSON.stringify(['Auto-detect data types', 'Outlier detection with IQR', 'Data quality score']),
        steps: JSON.stringify([
          { title: 'Load data', description: 'pd.read_csv() to load the dataset.' },
          { title: 'Inspect', description: 'Print shape, info(), null counts.' },
          { title: 'Handle missing values', description: 'Drop or fill with strategy.' },
          { title: 'Remove duplicates', description: 'df.drop_duplicates().' },
          { title: 'Save clean data', description: 'df.to_csv() to export.' },
          { title: 'Generate report', description: 'Print before/after comparison.' },
        ]),
        rubric: JSON.stringify([
          { criterion: 'Data loading', description: 'CSV loaded correctly' },
          { criterion: 'Cleaning', description: 'Missing values and duplicates handled' },
          { criterion: 'Report', description: 'Meaningful before/after stats' },
        ]),
      },
      {
        id: 'project-004', title: 'EDA Dashboard', slug: 'eda-dashboard', stage: 'DATA', order: 4,
        brief: 'Perform complete EDA on a dataset with visualizations and insights.',
        requirements: JSON.stringify(['Load dataset with 100+ rows', 'Generate descriptive statistics', 'Create 4+ visualization types', 'Identify key patterns', 'Handle missing data', 'Write findings summary']),
        stretchGoals: JSON.stringify(['Correlation heatmap', 'Interactive elements', 'Subgroup comparisons']),
        steps: JSON.stringify([
          { title: 'Choose dataset', description: 'Pick a CSV with enough complexity.' },
          { title: 'Initial inspection', description: 'shape, dtypes, head(), describe().' },
          { title: 'Clean data', description: 'Handle nulls and fix types.' },
          { title: 'Create visualizations', description: 'Histograms, scatter, bar, box plots.' },
          { title: 'Analyze patterns', description: 'Correlations, trends, outliers.' },
          { title: 'Write summary', description: 'Document key findings.' },
        ]),
        rubric: JSON.stringify([
          { criterion: 'Exploration', description: 'Thorough data inspection' },
          { criterion: 'Visualizations', description: '4+ clear, labeled charts' },
          { criterion: 'Insights', description: 'Meaningful findings documented' },
        ]),
      },
      {
        id: 'project-005', title: 'Titanic Survival Classification', slug: 'titanic-classification', stage: 'ML', order: 5,
        brief: 'Build an ML model to predict Titanic passenger survival.',
        requirements: JSON.stringify(['Load and explore Titanic data', 'Handle missing values', 'Encode categorical features', 'Train 2+ classifiers', 'Evaluate with accuracy, F1', 'Compare models']),
        stretchGoals: JSON.stringify(['Engineer new features (FamilySize, Title)', 'GridSearchCV tuning', 'Confusion matrix visualization']),
        steps: JSON.stringify([
          { title: 'Load and explore', description: 'Load CSV, perform EDA.' },
          { title: 'Preprocess', description: 'Handle nulls, encode categoricals.' },
          { title: 'Split data', description: 'train_test_split with test_size=0.2.' },
          { title: 'Train models', description: 'LogisticRegression and DecisionTree.' },
          { title: 'Evaluate', description: 'Compare accuracy, precision, recall, F1.' },
          { title: 'Conclude', description: 'Pick best model and explain why.' },
        ]),
        rubric: JSON.stringify([
          { criterion: 'Preprocessing', description: 'Missing values and encoding handled' },
          { criterion: 'Model training', description: '2+ models trained correctly' },
          { criterion: 'Evaluation', description: 'Multiple metrics compared' },
        ]),
      },
      {
        id: 'project-006', title: 'House Price Regression', slug: 'house-price-regression', stage: 'ML', order: 6,
        brief: 'Build a regression model to predict house prices with pipelines and cross-validation.',
        requirements: JSON.stringify(['Load house price dataset', 'Feature engineering (scaling, encoding)', 'Build Pipeline with preprocessing + model', 'Train 2+ regressors', 'Evaluate with MSE, R²', 'Use cross-validation']),
        stretchGoals: JSON.stringify(['Polynomial features', 'GridSearchCV tuning', 'Residual plots']),
        steps: JSON.stringify([
          { title: 'Load and explore', description: 'Load dataset, perform EDA.' },
          { title: 'Feature engineering', description: 'Scale, encode, create features.' },
          { title: 'Build pipelines', description: 'sklearn Pipeline with steps.' },
          { title: 'Train models', description: 'LinearRegression and DecisionTreeRegressor.' },
          { title: 'Evaluate', description: 'MSE, RMSE, MAE, R² and cross_val_score.' },
          { title: 'Compare', description: 'Select best model, document reasoning.' },
        ]),
        rubric: JSON.stringify([
          { criterion: 'Feature engineering', description: 'Proper scaling and encoding' },
          { criterion: 'Pipeline usage', description: 'Pipelines used correctly' },
          { criterion: 'Evaluation', description: 'Multiple metrics with cross-validation' },
        ]),
      },
    ],
  });
  console.log('Seeded 6 projects');

  console.log('\n=== Seed Complete ===');
  console.log('  20 lessons, 60 exercises, 60 quizzes, 6 projects');
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
