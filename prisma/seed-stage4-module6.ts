async function seedModule6(prisma: any) {
  console.log('Seeding Stage 4 / Module 6: Object-Oriented Python...');

  // ═══════════════════════════════════════════════════════════════════
  //  SKILL TAGS (upsert so we never fail on duplicates)
  // ═══════════════════════════════════════════════════════════════════
  await Promise.all([
    prisma.skillTag.upsert({
      where: { id: 'skill-021' },
      update: {},
      create: { id: 'skill-021', name: 'OOP', slug: 'oop' },
    }),
    prisma.skillTag.upsert({
      where: { id: 'skill-022' },
      update: {},
      create: { id: 'skill-022', name: 'Classes', slug: 'classes' },
    }),
    prisma.skillTag.upsert({
      where: { id: 'skill-023' },
      update: {},
      create: { id: 'skill-023', name: 'Inheritance', slug: 'inheritance' },
    }),
    prisma.skillTag.upsert({
      where: { id: 'skill-024' },
      update: {},
      create: { id: 'skill-024', name: 'Decorators', slug: 'decorators' },
    }),
    prisma.skillTag.upsert({
      where: { id: 'skill-025' },
      update: {},
      create: { id: 'skill-025', name: 'Magic Methods', slug: 'magic-methods' },
    }),
  ]);
  console.log('Upserted 5 skill tags (skill-021 to skill-025)');

  // ═══════════════════════════════════════════════════════════════════
  //  STAGE 4 — Advanced Python
  // ═══════════════════════════════════════════════════════════════════
  const stage4 = await prisma.stage.create({
    data: {
      id: 'stage-004',
      title: 'Advanced Python',
      slug: 'advanced-python',
      description:
        'Write production-quality Python — OOP, clean code patterns, and professional tooling that every developer uses on the job.',
      order: 4,
    },
  });
  console.log('Created Stage 4: Advanced Python');

  // ═══════════════════════════════════════════════════════════════════
  //  MODULE 6 — Object-Oriented Python
  // ═══════════════════════════════════════════════════════════════════
  const module6 = await prisma.module.create({
    data: {
      id: 'module-006',
      stageId: stage4.id,
      title: 'Object-Oriented Python',
      slug: 'object-oriented-python',
      description:
        'Master classes, inheritance, and Pythonic OOP patterns used in every professional codebase.',
      order: 1,
    },
  });
  console.log('Created Module 6: Object-Oriented Python');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 21 — Classes & Objects
  // ═══════════════════════════════════════════════════════════════════
  const lesson21 = await prisma.lesson.create({
    data: {
      id: 'lesson-021',
      moduleId: module6.id,
      title: 'Classes & Objects — __init__, Attributes, Methods',
      slug: 'classes-and-objects',
      order: 1,
      content: `# Classes & Objects — \`__init__\`, Attributes, Methods

Up until now you have been writing **procedural** code — sequences of statements and functions that operate on data stored in plain variables, lists, and dictionaries. That approach works perfectly well for small scripts, but as programs grow in size and complexity, it becomes harder and harder to keep track of which functions belong together and which data they share. **Object-Oriented Programming (OOP)** solves this problem by bundling related data and behaviour into a single unit called a **class**.

---

## 1. What Is a Class?

A class is a **blueprint** for creating objects. Think of it like an architectural plan: the plan itself is not a house, but you can build many houses from the same plan. Each house (object) has the same structure yet can differ in the details — colour, number of rooms, and so on.

\`\`\`python
class Dog:
    """A simple Dog class."""
    pass
\`\`\`

The \`class\` keyword creates a new type. By convention, class names use **CamelCase** (also called PascalCase). Even this minimal class is already useful — you can create objects (also called **instances**) from it:

\`\`\`python
my_dog = Dog()
your_dog = Dog()
print(type(my_dog))  # <class '__main__.Dog'>
\`\`\`

Each call to \`Dog()\` returns a brand-new object that is independent of every other instance.

---

## 2. The \`__init__\` Method

Most classes need to store data when they are created. Python calls the special method \`__init__\` (short for *initialise*) automatically every time you create a new instance:

\`\`\`python
class Dog:
    def __init__(self, name, breed, age):
        self.name = name
        self.breed = breed
        self.age = age

rex = Dog("Rex", "German Shepherd", 3)
print(rex.name)   # Rex
print(rex.breed)  # German Shepherd
print(rex.age)    # 3
\`\`\`

Key things to notice:

1. **\`self\`** is always the first parameter of every method in a class. It refers to the **current instance** — the specific object that the method was called on. Python passes it automatically; you never pass it yourself.
2. \`self.name = name\` creates an **instance attribute** called \`name\` and stores the value that was passed in. Each object gets its own copy of instance attributes.
3. You call the class like a function — \`Dog("Rex", "German Shepherd", 3)\` — and Python internally calls \`__init__\` with the new object as \`self\`.

---

## 3. Instance Methods

Any function defined inside a class that takes \`self\` as its first parameter is an **instance method**. It can read and modify the object's attributes:

\`\`\`python
class Dog:
    def __init__(self, name, breed, age):
        self.name = name
        self.breed = breed
        self.age = age

    def bark(self):
        print(f"{self.name} says Woof!")

    def birthday(self):
        self.age += 1
        print(f"Happy birthday {self.name}! Now {self.age} years old.")

rex = Dog("Rex", "German Shepherd", 3)
rex.bark()       # Rex says Woof!
rex.birthday()   # Happy birthday Rex! Now 4 years old.
\`\`\`

When you write \`rex.bark()\`, Python translates this behind the scenes to \`Dog.bark(rex)\` — that is why \`self\` receives the \`rex\` object.

---

## 4. Instance Attributes vs Class Attributes

Attributes attached to \`self\` inside methods are **instance attributes** — each object gets its own independent copy. Attributes defined directly inside the class body (but outside any method) are **class attributes** — they are shared across every instance:

\`\`\`python
class Dog:
    species = "Canis familiaris"  # class attribute

    def __init__(self, name):
        self.name = name          # instance attribute

rex = Dog("Rex")
fido = Dog("Fido")

print(rex.species)   # Canis familiaris
print(fido.species)  # Canis familiaris  (same value, shared)

print(rex.name)      # Rex
print(fido.name)     # Fido              (different values)
\`\`\`

Class attributes are useful for constants or counters that apply to every instance of the class. Be careful: if you assign to \`rex.species = "Wolf"\`, Python creates a *new instance attribute* on \`rex\` that **shadows** the class attribute — the class attribute itself is unchanged.

\`\`\`python
rex.species = "Wolf"
print(rex.species)   # Wolf   (instance attribute shadows the class one)
print(fido.species)  # Canis familiaris  (still the class attribute)
print(Dog.species)   # Canis familiaris  (class attribute unchanged)
\`\`\`

---

## 5. Creating Multiple Objects

The real power of classes is that you can create as many objects as you need, each with its own state:

\`\`\`python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def passed(self):
        return self.grade >= 50

students = [
    Student("Alice", 85),
    Student("Bob", 42),
    Student("Charlie", 73),
]

for s in students:
    status = "passed" if s.passed() else "failed"
    print(f"{s.name}: {s.grade} — {status}")
\`\`\`

Output:

\`\`\`
Alice: 85 — passed
Bob: 42 — failed
Charlie: 73 — passed
\`\`\`

Each \`Student\` object carries its own \`name\` and \`grade\`, and the \`passed()\` method works on whatever instance calls it.

---

## 6. Why Use OOP?

| Benefit | Explanation |
|---|---|
| **Encapsulation** | Data and the functions that operate on it live together, making code easier to understand. |
| **Reusability** | Once you write a class, you can create as many instances as you need — or inherit from it to make specialised versions. |
| **Modularity** | Each class is a self-contained unit that can be tested, debugged, and maintained independently. |
| **Abstraction** | Users of a class only need to know its public interface, not its internal implementation. |

OOP is not always the right choice — simple scripts and data pipelines often work best as plain functions. But when your program models real-world entities (users, products, game characters, machine-learning models) classes provide a natural and powerful way to organise your code.

---

## 7. Quick Recap

- A **class** is a blueprint; an **object** (instance) is a concrete realisation of that blueprint.
- \`__init__\` runs automatically when you create an instance and is the place to set up instance attributes.
- \`self\` refers to the current instance and must be the first parameter of every instance method.
- **Instance attributes** belong to one object; **class attributes** are shared by all objects of that class.
- OOP helps organise code through encapsulation, reusability, modularity, and abstraction.`,
      commonMistakes: `## Common Mistakes

### 1. Forgetting \`self\` in Method Definitions

\`\`\`python
# Wrong — missing self
class Dog:
    def bark():
        print("Woof!")

rex = Dog()
rex.bark()  # TypeError: bark() takes 0 positional arguments but 1 was given

# Right
class Dog:
    def bark(self):
        print("Woof!")
\`\`\`

Every instance method **must** accept \`self\` as its first parameter. When you call \`rex.bark()\`, Python automatically passes \`rex\` as \`self\`.

### 2. Mutable Default Arguments in \`__init__\`

\`\`\`python
# Wrong — shared list across all instances!
class Team:
    def __init__(self, members=[]):
        self.members = members

a = Team()
b = Team()
a.members.append("Alice")
print(b.members)  # ['Alice'] — Oops! b got Alice too

# Right — use None and create a new list inside __init__
class Team:
    def __init__(self, members=None):
        self.members = members if members is not None else []
\`\`\`

Default mutable arguments (lists, dicts, sets) are created **once** when the function is defined and shared across every call. Always use \`None\` as the default and create a fresh object inside the method.

### 3. Confusing Class Attributes and Instance Attributes

\`\`\`python
class Counter:
    count = 0  # class attribute

    def increment(self):
        self.count += 1  # creates an INSTANCE attribute that shadows the class one!

c1 = Counter()
c2 = Counter()
c1.increment()
c1.increment()
print(c1.count)       # 2  (instance attribute)
print(c2.count)       # 0  (class attribute — unchanged!)
print(Counter.count)  # 0  (class attribute — unchanged!)
\`\`\`

If you intend to share a counter across instances, mutate the class attribute explicitly with \`Counter.count += 1\` instead of \`self.count += 1\`.`,
    },
  });

  // Exercises for Lesson 21
  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-061',
        lessonId: lesson21.id,
        prompt:
          'Create a `Dog` class with attributes `name`, `breed`, and `age` (set in `__init__`). Add a `bark()` method that prints `"{name} says Woof!"`. Create `dog = Dog("Rex", "German Shepherd", 3)` and call `dog.bark()`.',
        starterCode:
          '# Define the Dog class\nclass Dog:\n    pass\n\n# Create a Dog instance and call bark()\n',
        expectedOutput: 'Rex says Woof!',
        testCode: '',
        hints: JSON.stringify([
          'Define __init__(self, name, breed, age) and store each parameter as self.name, self.breed, self.age.',
          'The bark() method should use an f-string: print(f"{self.name} says Woof!")',
          'After the class definition, write: dog = Dog("Rex", "German Shepherd", 3) then dog.bark()',
        ]),
        order: 1,
      },
      {
        id: 'exercise-062',
        lessonId: lesson21.id,
        prompt:
          'Create a `BankAccount` class with `owner` and `balance` (default 0). Add `deposit(amount)` and `withdraw(amount)` methods. `withdraw` should print `"Insufficient funds"` if the amount exceeds the balance. Create an account for "Alice", deposit 100, withdraw 30, then print the balance.',
        starterCode:
          '# Define the BankAccount class\nclass BankAccount:\n    pass\n\n# Create account, deposit, withdraw, print balance\n',
        expectedOutput: '70',
        testCode: '',
        hints: JSON.stringify([
          'In __init__, set self.owner = owner and self.balance = balance (with default 0).',
          'deposit(self, amount) should do self.balance += amount.',
          'withdraw(self, amount) should check if amount > self.balance and print "Insufficient funds" if so, otherwise subtract.',
          'After the operations, print(account.balance) should output 70.',
        ]),
        order: 2,
      },
      {
        id: 'exercise-063',
        lessonId: lesson21.id,
        prompt:
          'Create a `Counter` class with a **class attribute** `total_counts = 0` and an **instance attribute** `count = 0`. The `increment()` method should increase both `self.count` and `Counter.total_counts` by 1. Create two counters `c1` and `c2`, increment `c1` three times and `c2` twice. Print `c1.count`, `c2.count`, and `Counter.total_counts` on separate lines.',
        starterCode:
          '# Define the Counter class\nclass Counter:\n    pass\n\n# Create counters, increment, and print results\n',
        expectedOutput: '3\n2\n5',
        testCode: '',
        hints: JSON.stringify([
          'Define total_counts = 0 directly inside the class body (not in __init__).',
          'In __init__, set self.count = 0 to give each instance its own counter.',
          'In increment(), use Counter.total_counts += 1 (not self.total_counts) to update the shared class attribute.',
          'Call c1.increment() three times, c2.increment() twice, then print each value.',
        ]),
        order: 3,
      },
    ],
  });

  // Quiz for Lesson 21
  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-061',
        lessonId: lesson21.id,
        question: 'What does the `self` parameter represent in a Python class method?',
        type: 'MCQ',
        options: JSON.stringify([
          'The class itself',
          'The current instance of the class',
          'The parent class',
          'A global variable',
        ]),
        correctAnswer: 'The current instance of the class',
        explanation:
          '`self` refers to the specific instance that the method is being called on. When you write `rex.bark()`, Python passes `rex` as the `self` argument automatically.',
        order: 1,
      },
      {
        id: 'quiz-062',
        lessonId: lesson21.id,
        question: 'Class attributes are shared across all instances of a class.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation:
          'Class attributes are defined in the class body outside of any method. They belong to the class itself and are shared by every instance. Modifying a class attribute via the class name affects all instances that have not shadowed it with an instance attribute of the same name.',
        order: 2,
      },
      {
        id: 'quiz-063',
        lessonId: lesson21.id,
        question: 'What method is called when a new object is created from a class?',
        type: 'MCQ',
        options: JSON.stringify(['__new__', '__init__', '__create__', '__start__']),
        correctAnswer: '__init__',
        explanation:
          '`__init__` is the initialiser method that Python calls automatically after creating a new instance. It is where you typically set up instance attributes. (Technically `__new__` creates the object first, but `__init__` is the standard place for initialisation logic.)',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 21: Classes & Objects');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 22 — Inheritance, Polymorphism & Composition
  // ═══════════════════════════════════════════════════════════════════
  const lesson22 = await prisma.lesson.create({
    data: {
      id: 'lesson-022',
      moduleId: module6.id,
      title: 'Inheritance, Polymorphism & Composition',
      slug: 'inheritance-polymorphism',
      order: 2,
      content: `# Inheritance, Polymorphism & Composition

Once you know how to build a single class, the next question is: how do you **reuse and extend** existing classes without copying and pasting code everywhere? Python gives you three powerful tools — **inheritance**, **polymorphism**, and **composition** — that together form the backbone of professional OOP design.

---

## 1. Single Inheritance

Inheritance lets you create a new class (**child** or **subclass**) that automatically gets all the attributes and methods of an existing class (**parent** or **superclass**). You only write the code that is *different* in the child:

\`\`\`python
class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def speak(self):
        return f"{self.name} says {self.sound}"

class Dog(Animal):
    def fetch(self):
        return f"{self.name} fetches the ball"

rex = Dog("Rex", "Woof")
print(rex.speak())   # Rex says Woof   — inherited from Animal
print(rex.fetch())   # Rex fetches the ball — defined in Dog
\`\`\`

\`Dog(Animal)\` means "Dog inherits from Animal". The \`Dog\` class automatically has \`__init__\`, \`name\`, \`sound\`, and \`speak()\` without writing them again. It adds its own \`fetch()\` method on top.

---

## 2. The \`super()\` Function

When a child class needs its own \`__init__\` (for example, to add extra attributes), you should call the parent's \`__init__\` using \`super()\` so the parent's setup logic still runs:

\`\`\`python
class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

class Dog(Animal):
    def __init__(self, name, sound, tricks=None):
        super().__init__(name, sound)   # call Animal.__init__
        self.tricks = tricks or []

    def learn_trick(self, trick):
        self.tricks.append(trick)

rex = Dog("Rex", "Woof")
rex.learn_trick("shake")
print(rex.name)     # Rex
print(rex.tricks)   # ['shake']
\`\`\`

\`super()\` returns a temporary proxy object that lets you call the parent class's methods. This is much better than hard-coding \`Animal.__init__(self, name, sound)\` because it works correctly with multiple inheritance too.

---

## 3. Method Overriding

A child class can **override** a parent method by defining a method with the same name. Python always uses the version defined on the most specific (child) class:

\`\`\`python
class Animal:
    def speak(self):
        return "..."

class Cat(Animal):
    def speak(self):
        return "Meow!"

class Dog(Animal):
    def speak(self):
        return "Woof!"

print(Cat().speak())  # Meow!
print(Dog().speak())  # Woof!
\`\`\`

Overriding is the mechanism behind **polymorphism** — different objects respond to the same method call in their own way.

---

## 4. Polymorphism

Polymorphism means "many forms". In practice, it means you can write code that works with *any* object that has the right method, regardless of its class:

\`\`\`python
animals = [Cat(), Dog(), Cat()]

for animal in animals:
    print(animal.speak())
\`\`\`

Output:

\`\`\`
Meow!
Woof!
Meow!
\`\`\`

The loop does not care whether each item is a \`Cat\` or a \`Dog\` — it just calls \`speak()\`. This is called **duck typing** in Python: "If it walks like a duck and quacks like a duck, it is a duck." You can check an object's type at runtime with \`isinstance()\`:

\`\`\`python
print(isinstance(rex, Dog))     # True
print(isinstance(rex, Animal))  # True  — Dog IS an Animal
print(isinstance(rex, Cat))     # False
\`\`\`

---

## 5. Composition vs Inheritance — "Has-a" vs "Is-a"

Inheritance models an **"is-a"** relationship: a Dog *is an* Animal. But not every relationship is "is-a". A Car *has an* Engine — it does not *inherit from* Engine. When one object **contains** another object, that is **composition**:

\`\`\`python
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower

    def start(self):
        return f"{self.horsepower}HP engine started"

class Car:
    def __init__(self, make, model, engine):
        self.make = make
        self.model = model
        self.engine = engine   # composition: Car HAS-A Engine

    def describe(self):
        return f"{self.make} {self.model} with {self.engine.horsepower}HP engine"

engine = Engine(335)
car = Car("Toyota", "Supra", engine)
print(car.describe())        # Toyota Supra with 335HP engine
print(car.engine.start())    # 335HP engine started
\`\`\`

A widely-cited design guideline from the Gang of Four book says: **"Prefer composition over inheritance."** Composition keeps your classes loosely coupled and easier to change. Use inheritance when there truly is an "is-a" relationship and you want to share behaviour across a family of related types.

---

## 6. Abstract Base Classes (ABCs)

Sometimes you want to define a parent class that **must not be instantiated directly** — it exists only to define an interface that child classes must implement. Python provides the \`abc\` module for this:

\`\`\`python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # TypeError: Can't instantiate abstract class
rect = Rectangle(5, 3)
print(rect.area())       # 15
print(rect.perimeter())  # 16
\`\`\`

If a child class forgets to implement an abstract method, Python raises a \`TypeError\` at instantiation time — catching the bug early.

---

## 7. Quick Recap

| Concept | What it does |
|---|---|
| **Inheritance** | Child class reuses parent class code (\`class Dog(Animal)\`). |
| **\`super()\`** | Calls the parent class's method from the child. |
| **Overriding** | Child replaces a parent method with its own version. |
| **Polymorphism** | Different objects respond to the same method call differently. |
| **\`isinstance()\`** | Checks whether an object is an instance of a given class. |
| **Composition** | One object contains another ("has-a" instead of "is-a"). |
| **ABC** | Abstract base class that enforces an interface on subclasses. |`,
      commonMistakes: `## Common Mistakes

### 1. Forgetting to Call \`super().__init__()\`

\`\`\`python
# Wrong — parent attributes are never set
class Dog(Animal):
    def __init__(self, name, sound, tricks=None):
        self.tricks = tricks or []

rex = Dog("Rex", "Woof")
print(rex.name)  # AttributeError: 'Dog' object has no attribute 'name'

# Right — call super().__init__() first
class Dog(Animal):
    def __init__(self, name, sound, tricks=None):
        super().__init__(name, sound)
        self.tricks = tricks or []
\`\`\`

If your child class defines its own \`__init__\`, the parent's \`__init__\` is **not** called automatically. You must call \`super().__init__(...)\` explicitly.

### 2. The Diamond Problem with Multiple Inheritance

\`\`\`python
class A:
    def greet(self):
        print("Hello from A")

class B(A):
    def greet(self):
        print("Hello from B")

class C(A):
    def greet(self):
        print("Hello from C")

class D(B, C):
    pass

d = D()
d.greet()  # Hello from B — Python uses the MRO (Method Resolution Order)
print(D.__mro__)
\`\`\`

Python resolves the diamond problem using the **C3 linearisation** algorithm (MRO). The order in which you list parent classes matters. Be aware of the MRO and keep your hierarchies simple.

### 3. Creating Deep Inheritance Hierarchies

\`\`\`python
# Bad — too many levels make code hard to follow
class A: ...
class B(A): ...
class C(B): ...
class D(C): ...
class E(D): ...
\`\`\`

A common rule of thumb is to keep inheritance hierarchies **no deeper than 2–3 levels**. If you find yourself going deeper, consider refactoring to **composition**. Flat hierarchies are easier to understand, test, and maintain.`,
    },
  });

  // Exercises for Lesson 22
  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-064',
        lessonId: lesson22.id,
        prompt:
          'Create an `Animal` class with `__init__(self, name, sound)` and a `speak()` method that returns `f"{name} says {sound}"`. Then create a `Dog` class that inherits from `Animal` and adds a `fetch()` method returning `f"{name} fetches the ball"`. Create `dog = Dog("Rex", "Woof")`, print `dog.speak()` and `dog.fetch()`.',
        starterCode:
          '# Define Animal and Dog classes\nclass Animal:\n    pass\n\nclass Dog(Animal):\n    pass\n\n# Create a Dog and print speak() and fetch()\n',
        expectedOutput: 'Rex says Woof\nRex fetches the ball',
        testCode: '',
        hints: JSON.stringify([
          'Animal.__init__ should store self.name and self.sound.',
          'speak() should return (not print) f"{self.name} says {self.sound}".',
          'Dog inherits from Animal, so it already has __init__ and speak(). Just add fetch().',
          'Use print(dog.speak()) and print(dog.fetch()) to produce the output.',
        ]),
        order: 1,
      },
      {
        id: 'exercise-065',
        lessonId: lesson22.id,
        prompt:
          'Create a base `Shape` class with an `area()` method that returns `0`. Create `Rectangle(width, height)` and `Circle(radius)` that inherit from `Shape` and override `area()`. Print the area of `Rectangle(5, 3)` and `Circle(7)` (rounded to 2 decimal places). Use `math.pi` for the circle.',
        starterCode:
          'import math\n\nclass Shape:\n    pass\n\nclass Rectangle(Shape):\n    pass\n\nclass Circle(Shape):\n    pass\n\n# Create shapes and print areas\n',
        expectedOutput: '15\n153.94',
        testCode: '',
        hints: JSON.stringify([
          'Rectangle.area() should return self.width * self.height.',
          'Circle.area() should return math.pi * self.radius ** 2.',
          'For the rectangle, print(rect.area()) will give 15 (an integer result from 5*3).',
          'For the circle, use round(circle.area(), 2) or f"{circle.area():.2f}" to get 153.94.',
        ]),
        order: 2,
      },
      {
        id: 'exercise-066',
        lessonId: lesson22.id,
        prompt:
          'Create an `Engine` class with `__init__(self, horsepower)`. Create a `Car` class with `__init__(self, make, model, engine)` that uses **composition** (Car *has an* Engine). Add a `describe()` method that returns `f"{make} {model} with {engine.horsepower}HP engine"`. Create an engine with 335 HP, a Car("Toyota", "Supra", engine), and print the description.',
        starterCode:
          '# Define Engine and Car classes\nclass Engine:\n    pass\n\nclass Car:\n    pass\n\n# Create engine, car, and print description\n',
        expectedOutput: 'Toyota Supra with 335HP engine',
        testCode: '',
        hints: JSON.stringify([
          'Engine.__init__ should store self.horsepower.',
          'Car.__init__ should store self.make, self.model, and self.engine (the Engine object).',
          'In describe(), access the engine\'s horsepower with self.engine.horsepower.',
          'Create: engine = Engine(335), car = Car("Toyota", "Supra", engine), then print(car.describe()).',
        ]),
        order: 3,
      },
    ],
  });

  // Quiz for Lesson 22
  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-064',
        lessonId: lesson22.id,
        question: 'What does `super()` do in a child class?',
        type: 'MCQ',
        options: JSON.stringify([
          'Creates a new parent instance',
          'Calls the parent class method',
          'Makes the class abstract',
          'Overrides the parent method',
        ]),
        correctAnswer: 'Calls the parent class method',
        explanation:
          '`super()` returns a proxy object that delegates method calls to the parent class. It is most commonly used to call the parent\'s `__init__` from a child class\'s `__init__`, ensuring the parent\'s setup logic still runs.',
        order: 1,
      },
      {
        id: 'quiz-065',
        lessonId: lesson22.id,
        question: 'Python supports multiple inheritance.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation:
          'Python supports multiple inheritance — a class can inherit from more than one parent class. Python uses the C3 linearisation algorithm (Method Resolution Order / MRO) to determine which parent\'s method to call when there are conflicts.',
        order: 2,
      },
      {
        id: 'quiz-066',
        lessonId: lesson22.id,
        question: 'Which design principle says "prefer composition over inheritance"?',
        type: 'MCQ',
        options: JSON.stringify(['DRY', 'SOLID', 'Gang of Four / GoF', 'KISS']),
        correctAnswer: 'Gang of Four / GoF',
        explanation:
          'The Gang of Four (GoF) book "Design Patterns: Elements of Reusable Object-Oriented Software" famously advises developers to "favor object composition over class inheritance" because composition leads to more flexible and loosely-coupled designs.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 22: Inheritance, Polymorphism & Composition');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 23 — Magic Methods, Decorators & Context Managers
  // ═══════════════════════════════════════════════════════════════════
  const lesson23 = await prisma.lesson.create({
    data: {
      id: 'lesson-023',
      moduleId: module6.id,
      title: 'Magic Methods, Decorators & Context Managers',
      slug: 'magic-methods-decorators',
      order: 3,
      content: `# Magic Methods, Decorators & Context Managers

Python's class system becomes truly powerful when you learn to hook into the language's built-in protocols. **Magic methods** (also called *dunder methods* because of their **d**ouble **under**score names) let your objects work seamlessly with Python operators and built-in functions. **Decorators** let you modify or extend functions and methods elegantly. **Context managers** let you control resource setup and teardown with the \`with\` statement. Together, these three features are what separate beginner Python from professional, Pythonic code.

---

## 1. String Representations: \`__str__\` and \`__repr__\`

When you \`print()\` an object or inspect it in the REPL, Python calls one of two methods:

- **\`__str__\`** — meant for end-users; called by \`print()\` and \`str()\`.
- **\`__repr__\`** — meant for developers; called by \`repr()\` and shown in the REPL. Should ideally return a string that could recreate the object.

\`\`\`python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

p = Point(3, 4)
print(repr(p))  # Point(3, 4)   — __repr__
print(p)        # (3, 4)        — __str__
\`\`\`

If you only define one, define \`__repr__\`. Python falls back to \`__repr__\` when \`__str__\` is not defined, but not vice versa.

---

## 2. Comparison and Equality: \`__eq__\`, \`__lt__\`, and Friends

By default, \`==\` checks whether two variables refer to the *same object in memory*. To compare by *value*, implement \`__eq__\`:

\`\`\`python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return (self.x ** 2 + self.y ** 2) < (other.x ** 2 + other.y ** 2)

a = Point(1, 2)
b = Point(1, 2)
c = Point(3, 4)

print(a == b)  # True  (same values)
print(a == c)  # False
print(a < c)   # True  (magnitude comparison)
\`\`\`

The full set of comparison methods: \`__eq__\`, \`__ne__\`, \`__lt__\`, \`__le__\`, \`__gt__\`, \`__ge__\`. You can use the \`@functools.total_ordering\` decorator to auto-generate the rest from just \`__eq__\` and one ordering method.

---

## 3. Operator Overloading: \`__add__\`, \`__len__\`, and More

You can make your objects work with Python's built-in operators and functions by implementing the right magic methods:

\`\`\`python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __len__(self):
        return int((self.x ** 2 + self.y ** 2) ** 0.5)

v1 = Vector(3, 4)
v2 = Vector(1, 2)
print(repr(v1 + v2))  # Vector(4, 6)
print(len(v1))         # 5
\`\`\`

Common magic methods for operators:

| Operator | Method |
|---|---|
| \`+\` | \`__add__\` |
| \`-\` | \`__sub__\` |
| \`*\` | \`__mul__\` |
| \`[]\` | \`__getitem__\` |
| \`len()\` | \`__len__\` |
| \`in\` | \`__contains__\` |
| \`()\` | \`__call__\` |

---

## 4. The \`@property\` Decorator

\`@property\` lets you define a method that is accessed like an attribute — no parentheses needed. This is perfect for computed values or for adding validation to attribute access:

\`\`\`python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):
        import math
        return math.pi * self._radius ** 2

c = Circle(5)
print(c.radius)       # 5        — no parentheses
print(f"{c.area:.2f}")  # 78.54  — computed on access
c.radius = 10         # uses the setter
# c.radius = -1       # raises ValueError
\`\`\`

---

## 5. \`@staticmethod\` and \`@classmethod\`

These two decorators define methods that do not operate on a specific instance:

- **\`@staticmethod\`** — does not receive \`self\` or \`cls\`. It is essentially a regular function that lives inside the class for organisational purposes.
- **\`@classmethod\`** — receives the **class** (\`cls\`) as its first argument instead of an instance. Commonly used as alternative constructors.

\`\`\`python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    @classmethod
    def from_fahrenheit(cls, f):
        return cls((f - 32) * 5 / 9)

    @staticmethod
    def is_boiling(celsius):
        return celsius >= 100

t = Temperature.from_fahrenheit(212)
print(t.celsius)                    # 100.0
print(Temperature.is_boiling(100))  # True
\`\`\`

\`from_fahrenheit\` receives \`cls\` (which is \`Temperature\`), so it works correctly even with subclasses. \`is_boiling\` is a utility that does not need access to any instance or class state.

---

## 6. Custom Decorators with \`functools.wraps\`

A decorator is a function that takes another function, extends its behaviour, and returns the modified function. The \`@\` syntax is just syntactic sugar:

\`\`\`python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Function {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    return sum(range(n))

total = slow_sum(1_000_000)
print(total)
\`\`\`

Output (timing will vary):

\`\`\`
Function slow_sum took 0.0312s
499999500000
\`\`\`

**\`@functools.wraps(func)\`** is critical — it copies the original function's name, docstring, and other metadata onto the wrapper so that debugging tools, \`help()\`, and introspection still work correctly. Without it, \`slow_sum.__name__\` would be \`"wrapper"\` instead of \`"slow_sum"\`.

---

## 7. Context Managers — \`__enter__\` and \`__exit__\`

A context manager is any object that implements \`__enter__\` and \`__exit__\`. The \`with\` statement calls \`__enter__\` at the beginning of the block and \`__exit__\` at the end — even if an exception occurs:

\`\`\`python
class FileLogger:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        print(f"Opening {self.filename}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing {self.filename}")
        return False  # do not suppress exceptions

with FileLogger("app.log") as logger:
    print("Writing data...")
\`\`\`

Output:

\`\`\`
Opening app.log
Writing data...
Closing app.log
\`\`\`

For simpler cases, the \`contextlib\` module provides a generator-based shortcut:

\`\`\`python
from contextlib import contextmanager

@contextmanager
def file_logger(filename):
    print(f"Opening {filename}")
    try:
        yield filename
    finally:
        print(f"Closing {filename}")

with file_logger("app.log") as name:
    print(f"Writing to {name}...")
\`\`\`

Context managers are used everywhere in Python: opening files (\`with open(...)\`), database connections, network sockets, locks in multithreading, and temporary directories.

---

## 8. Quick Recap

| Feature | Purpose |
|---|---|
| \`__str__\` / \`__repr__\` | Human-readable / developer-readable string output. |
| \`__eq__\`, \`__lt__\`, ... | Value-based comparison and ordering. |
| \`__add__\`, \`__len__\`, ... | Operator overloading and built-in function support. |
| \`@property\` | Access a method like an attribute. |
| \`@staticmethod\` | Method that does not need \`self\` or \`cls\`. |
| \`@classmethod\` | Method that receives the class (\`cls\`) — often used as alternative constructors. |
| Decorators | Wrap a function to add behaviour; always use \`@functools.wraps\`. |
| Context managers | \`__enter__\` / \`__exit__\` (or \`@contextmanager\`) for resource management. |`,
      commonMistakes: `## Common Mistakes

### 1. Forgetting \`@functools.wraps\` in Custom Decorators

\`\`\`python
import functools

# Wrong — wrapper hides the original function's identity
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet():
    """Say hello."""
    print("Hello!")

print(greet.__name__)  # 'wrapper' — wrong!

# Right — use @functools.wraps(func)
def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet():
    """Say hello."""
    print("Hello!")

print(greet.__name__)  # 'greet' — correct!
\`\`\`

Always apply \`@functools.wraps(func)\` to the inner wrapper function so that the decorated function retains its original \`__name__\`, \`__doc__\`, and other attributes.

### 2. Confusing \`@staticmethod\` and \`@classmethod\`

\`\`\`python
class MyClass:
    count = 0

    @staticmethod
    def reset():
        MyClass.count = 0  # must hard-code the class name

    @classmethod
    def reset_better(cls):
        cls.count = 0      # works with subclasses too

class SubClass(MyClass):
    pass

# staticmethod always resets MyClass.count, even when called on SubClass
# classmethod resets whichever class calls it
\`\`\`

Use \`@classmethod\` when you need access to the class (especially for alternative constructors or when subclasses are involved). Use \`@staticmethod\` only for utility functions that do not need \`self\` or \`cls\` at all.

### 3. Not Returning \`self\` from \`__enter__\`

\`\`\`python
# Wrong — 'logger' will be None
class Logger:
    def __enter__(self):
        print("Starting")
        # forgot to return self!

    def __exit__(self, *args):
        print("Done")

with Logger() as logger:
    print(logger)  # None

# Right — return self (or another useful object)
class Logger:
    def __enter__(self):
        print("Starting")
        return self

    def __exit__(self, *args):
        print("Done")
\`\`\`

The value after \`as\` in a \`with\` statement is whatever \`__enter__\` returns. If you forget to \`return self\`, the variable will be \`None\`.`,
    },
  });

  // Exercises for Lesson 23
  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-067',
        lessonId: lesson23.id,
        prompt:
          'Create a `Vector` class with `x` and `y` attributes. Implement `__repr__` to return `"Vector(x, y)"`, `__add__` for vector addition (returns a new Vector), and `__len__` that returns the integer magnitude (`int` of the square root of x^2 + y^2). Create `v1 = Vector(3, 4)` and `v2 = Vector(1, 2)`. Print `repr(v1)`, `repr(v1 + v2)`, and `len(v1)`.',
        starterCode:
          '# Define the Vector class with magic methods\nclass Vector:\n    pass\n\n# Create vectors and print results\n',
        expectedOutput: 'Vector(3, 4)\nVector(4, 6)\n5',
        testCode: '',
        hints: JSON.stringify([
          '__repr__ should return f"Vector({self.x}, {self.y})".',
          '__add__(self, other) should return Vector(self.x + other.x, self.y + other.y).',
          '__len__ should return int((self.x ** 2 + self.y ** 2) ** 0.5).',
          'Use print(repr(v1)), print(repr(v1 + v2)), print(len(v1)).',
        ]),
        order: 1,
      },
      {
        id: 'exercise-068',
        lessonId: lesson23.id,
        prompt:
          'Write a decorator `@timer` that prints `"Function {name} took {time:.4f}s"` using the `time` module and `functools.wraps`. Apply it to a function `compute()` that calculates `sum(range(1_000_000))` and prints the result. Call `compute()`.',
        starterCode:
          'import time\nimport functools\n\n# Define the timer decorator\n\n# Define the compute function with @timer\n\n# Call compute()\n',
        expectedOutput: '499999500000',
        testCode:
          'import sys\noutput = sys.stdout.getvalue()\nassert "499999500000" in output, "Should print the sum result"\nassert "took" in output, "Should print timing information"',
        hints: JSON.stringify([
          'The timer decorator takes func as argument and returns a wrapper function.',
          'Inside wrapper, record start = time.time(), call func, record end, print the timing.',
          'Use @functools.wraps(func) on the wrapper to preserve metadata.',
          'compute() should print(sum(range(1_000_000))).',
        ]),
        order: 2,
      },
      {
        id: 'exercise-069',
        lessonId: lesson23.id,
        prompt:
          'Create a `FileLogger` class that works as a context manager. `__init__` takes `filename` and stores it. `__enter__` prints `"Opening {filename}"` and returns `self`. `__exit__` prints `"Closing {filename}"`. Use it: `with FileLogger("app.log") as logger: print("Writing data...")`.',
        starterCode:
          '# Define the FileLogger context manager class\nclass FileLogger:\n    pass\n\n# Use the context manager\n',
        expectedOutput: 'Opening app.log\nWriting data...\nClosing app.log',
        testCode: '',
        hints: JSON.stringify([
          '__init__ should store self.filename = filename.',
          '__enter__ should print(f"Opening {self.filename}") and return self.',
          '__exit__ takes (self, exc_type, exc_val, exc_tb) and prints(f"Closing {self.filename}").',
          'The with block will call __enter__, run the body, then call __exit__ automatically.',
        ]),
        order: 3,
      },
    ],
  });

  // Quiz for Lesson 23
  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-067',
        lessonId: lesson23.id,
        question: 'What magic method is called by `print(obj)`?',
        type: 'MCQ',
        options: JSON.stringify(['__repr__', '__str__', '__print__', '__display__']),
        correctAnswer: '__str__',
        explanation:
          '`print()` calls `str()` on its argument, which in turn calls the object\'s `__str__` method. If `__str__` is not defined, Python falls back to `__repr__`.',
        order: 1,
      },
      {
        id: 'quiz-068',
        lessonId: lesson23.id,
        question: '@staticmethod methods have access to the instance via `self`.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation:
          '@staticmethod methods do not receive `self` or `cls` as a parameter. They behave like regular functions but live inside the class namespace for organisational purposes. Use @classmethod if you need access to the class.',
        order: 2,
      },
      {
        id: 'quiz-069',
        lessonId: lesson23.id,
        question: 'What does `@property` allow you to do?',
        type: 'MCQ',
        options: JSON.stringify([
          'Make a method callable without parentheses',
          'Make an attribute read-only',
          'Access a method like an attribute',
          'All of the above',
        ]),
        correctAnswer: 'Access a method like an attribute',
        explanation:
          '`@property` turns a method into a descriptor so that it can be accessed like an attribute (without parentheses). While it can also be used to make attributes read-only (by not defining a setter), its primary purpose is to let you access a method as if it were an attribute.',
        order: 3,
      },
    ],
  });

  console.log('Seeded Lesson 23: Magic Methods, Decorators & Context Managers');

  console.log('Stage 4 / Module 6 seeding complete!');
}

module.exports = { seedModule6 };
