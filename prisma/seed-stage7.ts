export {};

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  const existing = await prisma.stage.findUnique({ where: { id: 'stage-007' } });
  if (existing) { console.log('Stage 7 already seeded. Skipping.'); return; }

  console.log('═══════════════════════════════════════════');
  console.log('  Seeding Stage 7: Agentic AI');
  console.log('═══════════════════════════════════════════');

  // ─── Skill Tags ────────────────────────────────────────────────────
  const skills = [
    { id: 'skill-047', name: 'AI Agents', slug: 'ai-agents' },
    { id: 'skill-048', name: 'Tool Use', slug: 'tool-use' },
    { id: 'skill-049', name: 'ReAct Pattern', slug: 'react-pattern' },
    { id: 'skill-050', name: 'Multi-Agent Systems', slug: 'multi-agent' },
    { id: 'skill-051', name: 'Agent Safety', slug: 'agent-safety' },
    { id: 'skill-052', name: 'Agent Deployment', slug: 'agent-deployment' },
  ];
  for (const s of skills) {
    await prisma.skillTag.upsert({ where: { id: s.id }, update: {}, create: s });
  }

  // ─── Stage ─────────────────────────────────────────────────────────
  await prisma.stage.create({
    data: {
      id: 'stage-007', title: 'Agentic AI', slug: 'agentic-ai',
      description: 'Build autonomous AI agents that use tools, make decisions, and complete multi-step tasks — the frontier of applied AI.',
      order: 7,
    },
  });

  // ═══════════════════════════════════════════════════════════════════
  //  MODULE 11: AI Agents Fundamentals
  // ═══════════════════════════════════════════════════════════════════
  const mod11 = await prisma.module.create({
    data: {
      id: 'module-011', stageId: 'stage-007', title: 'AI Agents Fundamentals',
      slug: 'ai-agents-fundamentals', order: 1,
      description: 'Understand the core patterns behind AI agents and build your first autonomous agent.',
    },
  });

  // ── Lesson 39 ──────────────────────────────────────────────────────
  const L39 = await prisma.lesson.create({ data: {
    id: 'lesson-039', moduleId: mod11.id, title: 'What Are AI Agents — ReAct, Tool Use, Memory',
    slug: 'what-are-ai-agents', order: 1,
    content: `# What Are AI Agents — ReAct, Tool Use, Memory

An AI agent is an LLM-powered system that can **autonomously decide what actions to take** to accomplish a goal. Unlike a simple chatbot that just responds to messages, an agent can reason about a task, use tools, and iterate until the task is done.

## Chatbots vs Agents

| Chatbot | Agent |
|---------|-------|
| Responds to one message at a time | Plans and executes multi-step tasks |
| No access to external tools | Uses tools (search, code, APIs) |
| Stateless between turns | Maintains memory and context |
| Human drives the conversation | Agent drives toward a goal |

## The Agent Loop

Every agent follows this core loop:

\`\`\`
1. OBSERVE → Receive task or observe current state
2. THINK  → Reason about what to do next
3. ACT    → Execute an action (call a tool, generate text)
4. OBSERVE → See the result of the action
5. Repeat until task is complete or budget exhausted
\`\`\`

\`\`\`python
def agent_loop(task, tools, max_steps=10):
    context = f"Task: {task}"
    for step in range(max_steps):
        # Think: decide what to do
        decision = llm_call(context)

        if decision.type == "final_answer":
            return decision.answer

        # Act: execute the chosen tool
        result = tools[decision.tool_name](decision.tool_args)

        # Observe: add result to context
        context += f"\\nAction: {decision.tool_name}({decision.tool_args})"
        context += f"\\nResult: {result}"

    return "Max steps reached"
\`\`\`

## The ReAct Pattern

**ReAct** (Reason + Act) is the most popular agent framework. At each step, the agent explicitly writes:
1. **Thought:** reasoning about what to do
2. **Action:** the tool to call and arguments
3. **Observation:** the result from the tool

\`\`\`
Thought: I need to find the population of France.
Action: search("population of France 2024")
Observation: France has a population of approximately 68 million.

Thought: I now have the answer.
Action: final_answer("France has approximately 68 million people.")
\`\`\`

## Tool Use

Tools give agents real-world capabilities. Common tools:
- **Search:** Web search, document search
- **Code execution:** Run Python code
- **API calls:** Weather, databases, external services
- **File operations:** Read, write, list files
- **Calculator:** Math operations

Each tool has a **definition** (name, description, parameters) that helps the LLM decide when and how to use it.

## Memory

Agents need memory to maintain context across steps:

**Short-term (Working) Memory:** The current conversation/action history. Passed in the prompt each turn.

**Long-term Memory:** Stored in a vector database. Retrieved when relevant. Persists across sessions.

**Summary Memory:** Periodically summarize older conversation history to save tokens while retaining key information.

## When to Use Agents vs Simple Prompts

| Use a simple prompt when... | Use an agent when... |
|-----------------------------|---------------------|
| Task is one-shot | Task requires multiple steps |
| No external data needed | Need to search/fetch data |
| Output format is clear | Need to iterate on quality |
| Low stakes | Need tool access |

## Real-World Agent Examples

- **Coding agents** (Claude Code, Cursor): Write, test, and debug code
- **Research agents:** Search papers, summarize findings
- **Customer support:** Handle tickets with access to databases
- **Data analysis:** Load data, run queries, create visualizations`,
    commonMistakes: `## Common Mistakes

### 1. Using Agents for Simple Tasks
Not everything needs an agent. A simple prompt with few-shot examples is faster and cheaper for straightforward tasks.

### 2. No Exit Condition
Always set a max_steps limit. Without it, agents can loop forever, burning through API credits.

### 3. Unlimited Tool Loops
An agent might repeatedly call the same tool with slightly different inputs. Add deduplication and loop detection.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-115', lessonId: L39.id, prompt: 'Implement a simple agent loop simulation. Print 5 steps: "Thinking about: {task}", "Searching for information", "Found relevant data", "Generating response", "Task complete". Test with task="summarize Python".', starterCode: 'def agent_loop(task, max_steps=5):\n    steps = [\n        f"Thinking about: {task}",\n        "Searching for information",\n        "Found relevant data",\n        "Generating response",\n        "Task complete"\n    ]\n    # Print each step\n    pass\n\nagent_loop("summarize Python")\n', expectedOutput: 'Step 1: Thinking about: summarize Python\nStep 2: Searching for information\nStep 3: Found relevant data\nStep 4: Generating response\nStep 5: Task complete', testCode: '', hints: JSON.stringify(['Loop through steps with enumerate', 'print(f"Step {i+1}: {step}")', 'Use enumerate(steps) starting from 0']), order: 1 },
    { id: 'exercise-116', lessonId: L39.id, prompt: 'Implement a ReAct step formatter. Write `format_react_step(thought, action, observation)` returning the formatted string. Test it.', starterCode: 'def format_react_step(thought, action, observation):\n    pass\n\nprint(format_react_step(\n    "I need to calculate 2+2",\n    "calculator(2, 2, \'add\')",\n    "4"\n))\n', expectedOutput: "Thought: I need to calculate 2+2\nAction: calculator(2, 2, 'add')\nObservation: 4", testCode: '', hints: JSON.stringify(['Return an f-string with three lines', 'Use \\n to separate lines', 'f"Thought: {thought}\\nAction: {action}\\nObservation: {observation}"']), order: 2 },
    { id: 'exercise-117', lessonId: L39.id, prompt: 'Create a tool registry. Class `ToolRegistry` with `register(name, description)` and `list_tools()` returning sorted "name: description" strings. Register 3 tools and print.', starterCode: 'class ToolRegistry:\n    def __init__(self):\n        self.tools = {}\n\n    def register(self, name, description):\n        pass\n\n    def list_tools(self):\n        pass\n\nregistry = ToolRegistry()\nregistry.register("calculator", "Performs math operations")\nregistry.register("search", "Searches the web")\nregistry.register("weather", "Gets weather data")\nfor tool in registry.list_tools():\n    print(tool)\n', expectedOutput: 'calculator: Performs math operations\nsearch: Searches the web\nweather: Gets weather data', testCode: '', hints: JSON.stringify(['Store tools in self.tools dict: self.tools[name] = description', 'list_tools: return sorted list of f"{k}: {v}" strings', 'Sort by key: sorted(self.tools.items())']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-115', lessonId: L39.id, question: 'What is the core difference between a chatbot and an AI agent?', type: 'MCQ', options: JSON.stringify(['Agents use bigger models', 'Agents can autonomously plan and use tools', 'Agents are always more accurate', 'Agents don\'t need prompts']), correctAnswer: 'Agents can autonomously plan and use tools', explanation: 'Unlike chatbots that respond to individual messages, agents can plan multi-step approaches and use external tools to accomplish goals.', order: 1 },
    { id: 'quiz-116', lessonId: L39.id, question: 'The ReAct pattern combines reasoning and acting in each step.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'ReAct = Reason + Act. Each step explicitly includes a Thought (reasoning), Action (tool call), and Observation (result).', order: 2 },
    { id: 'quiz-117', lessonId: L39.id, question: 'What should every agent loop have to prevent infinite execution?', type: 'MCQ', options: JSON.stringify(['A database connection', 'A maximum step/iteration limit', 'Multiple LLM models', 'A user interface']), correctAnswer: 'A maximum step/iteration limit', explanation: 'Without a max_steps limit, agents can loop forever, wasting API credits and never completing. Always set a budget.', order: 3 },
  ]});
  console.log('Seeded Lesson 39');

  // ── Lesson 40 ──────────────────────────────────────────────────────
  const L40 = await prisma.lesson.create({ data: {
    id: 'lesson-040', moduleId: mod11.id, title: 'Building Your First Agent — Function Calling & Tools',
    slug: 'building-first-agent', order: 2,
    content: `# Building Your First Agent — Function Calling & Tools

Function calling is the mechanism that allows LLMs to use tools. Instead of just generating text, the model outputs a structured request to call a specific function with specific arguments.

## How Function Calling Works

1. You define available tools with names, descriptions, and parameter schemas
2. You send the user's message + tool definitions to the LLM
3. The LLM decides whether to call a tool and which one
4. You execute the tool locally and send the result back
5. The LLM generates the final answer using the tool result

## Defining Tools

Each tool needs a clear definition so the LLM knows when and how to use it:

\`\`\`python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    }
]
\`\`\`

**Good descriptions are critical.** The LLM uses the description to decide which tool to use. Vague descriptions = wrong tool choices.

## Implementing Tool Functions

\`\`\`python
def get_weather(city, unit="celsius"):
    # In reality, this would call a weather API
    return {"city": city, "temp": 22, "unit": unit, "condition": "sunny"}

def calculate(expression):
    try:
        result = eval(expression)  # In production, use a safe evaluator
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Tool dispatcher
tool_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
}

def execute_tool(tool_name, tool_args):
    if tool_name in tool_functions:
        return tool_functions[tool_name](**tool_args)
    return {"error": f"Unknown tool: {tool_name}"}
\`\`\`

## The Complete Agent Flow

\`\`\`python
def run_agent(user_message, tools, tool_functions, max_turns=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": user_message}
    ]

    for turn in range(max_turns):
        response = call_llm(messages, tools=tools)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call.name, tool_call.args)
                messages.append({"role": "tool", "content": str(result), "tool_call_id": tool_call.id})
        else:
            return response.content  # Final answer

    return "Max turns reached"
\`\`\`

## Error Handling

Always handle tool execution errors gracefully:

\`\`\`python
def safe_execute_tool(tool_name, tool_args):
    try:
        result = tool_functions[tool_name](**tool_args)
        return {"success": True, "result": result}
    except KeyError:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    except TypeError as e:
        return {"success": False, "error": f"Invalid arguments: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Tool error: {e}"}
\`\`\`

The LLM can see the error and try a different approach — this makes agents self-correcting.`,
    commonMistakes: `## Common Mistakes

### 1. Vague Tool Descriptions
"Searches stuff" → "Searches the web for current information using Google. Returns top 5 results with titles and snippets."

### 2. Not Validating Tool Inputs
Always check that the LLM provided valid arguments before executing a tool.

### 3. Forgetting to Handle Tool Errors
If a tool crashes, the agent loop breaks. Always wrap tool calls in try/except.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-118', lessonId: L40.id, prompt: 'Define tool schemas as dicts. Create 2 tools (get_weather and calculate). Write `find_tool(tools, name)` to find a tool by name. Print the description of "calculate".', starterCode: 'tools = [\n    {"name": "get_weather", "description": "Get weather for a city", "parameters": {"city": "string"}},\n    {"name": "calculate", "description": "Perform a calculation", "parameters": {"expression": "string"}}\n]\n\ndef find_tool(tools, name):\n    pass\n\ntool = find_tool(tools, "calculate")\nprint(tool["description"])\n', expectedOutput: 'Perform a calculation', testCode: '', hints: JSON.stringify(['Loop through tools and check if tool["name"] == name', 'Return the matching tool dict', 'Return None if not found']), order: 1 },
    { id: 'exercise-119', lessonId: L40.id, prompt: 'Implement a tool dispatcher. Given tools dict mapping names to functions, write `execute_tool(tools, name, args)` that calls the function. Test with add and mul operations.', starterCode: 'tools = {\n    "add": lambda a, b: a + b,\n    "sub": lambda a, b: a - b,\n    "mul": lambda a, b: a * b\n}\n\ndef execute_tool(tools, name, args):\n    pass\n\nprint(execute_tool(tools, "add", [10, 5]))\nprint(execute_tool(tools, "mul", [3, 7]))\n', expectedOutput: '15\n21', testCode: '', hints: JSON.stringify(['Get the function: func = tools[name]', 'Call with unpacked args: func(*args)', 'Return the result']), order: 2 },
    { id: 'exercise-120', lessonId: L40.id, prompt: 'Parse tool calls from text. Given "TOOL: calculator ARGS: 15, 27", extract tool name and args list. Print as dict.', starterCode: 'def parse_tool_call(text):\n    pass\n\nresult = parse_tool_call("TOOL: calculator ARGS: 15, 27")\nprint(result)\n', expectedOutput: "{'tool': 'calculator', 'args': ['15', '27']}", testCode: '', hints: JSON.stringify(['Split on "TOOL: " and "ARGS: "', 'tool_part = text.split("ARGS:")[0].split("TOOL:")[1].strip()', 'args = [a.strip() for a in args_part.split(",")]']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-118', lessonId: L40.id, question: 'What is function calling in the context of LLMs?', type: 'MCQ', options: JSON.stringify(['Calling Python functions directly', 'The LLM outputting structured requests to invoke tools', 'A programming paradigm', 'Running code inside the model']), correctAnswer: 'The LLM outputting structured requests to invoke tools', explanation: 'Function calling lets the LLM output structured tool invocation requests instead of just text, which your code then executes.', order: 1 },
    { id: 'quiz-119', lessonId: L40.id, question: 'Tool descriptions don\'t matter much — the LLM figures it out from the name alone.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Clear, detailed tool descriptions are critical. The LLM relies heavily on descriptions to choose the right tool and pass correct arguments.', order: 2 },
    { id: 'quiz-120', lessonId: L40.id, question: 'What happens if a tool execution fails in an agent?', type: 'MCQ', options: JSON.stringify(['The agent crashes', 'The error should be returned to the LLM so it can try a different approach', 'The tool is permanently disabled', 'The user must fix it']), correctAnswer: 'The error should be returned to the LLM so it can try a different approach', explanation: 'Good agents handle tool errors by returning the error message to the LLM, which can then reason about the failure and try an alternative.', order: 3 },
  ]});
  console.log('Seeded Lesson 40');

  // ── Lessons 41-42, Module 12, Lessons 43-45 (condensed for brevity but full content) ──

  const L41 = await prisma.lesson.create({ data: {
    id: 'lesson-041', moduleId: mod11.id, title: 'Multi-Step Agents — Chains, Planning & Error Recovery',
    slug: 'multi-step-agents', order: 3,
    content: `# Multi-Step Agents — Chains, Planning & Error Recovery

Real-world tasks rarely complete in one step. Multi-step agents plan, execute, recover from errors, and iterate until the task is done.

## Single-Step vs Multi-Step

**Single-step:** User asks → LLM calls one tool → returns answer.
**Multi-step:** User asks → LLM plans → calls tool A → reasons about result → calls tool B → ... → final answer.

Example: "Find the cheapest flight from NYC to London next Friday"
1. Search for flights NYC → London
2. Filter by date (next Friday)
3. Sort by price
4. Return the cheapest option

## Task Decomposition

Break complex tasks into smaller, manageable subtasks:

\`\`\`python
def decompose_task(task):
    """Ask the LLM to break a task into steps."""
    prompt = f"""Break this task into 3-5 sequential steps:
Task: {task}

Return numbered steps only."""
    return llm_call(prompt)
\`\`\`

## Planning Strategies

1. **Plan-then-execute:** Create a full plan upfront, then execute each step
2. **Dynamic planning:** Plan the next step based on previous results
3. **Hybrid:** Create a rough plan, adjust as you go

## Error Recovery

Agents will encounter failures. Good error handling makes the difference:

\`\`\`python
def execute_with_retry(func, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            wait = 2 ** (attempt + 1)  # Exponential backoff
            time.sleep(wait)
\`\`\`

## Budget Management

Always limit agent execution to prevent runaway costs:

\`\`\`python
class AgentBudget:
    def __init__(self, max_steps=10, max_tokens=10000):
        self.max_steps = max_steps
        self.steps_used = 0
        self.max_tokens = max_tokens
        self.tokens_used = 0

    def can_continue(self):
        return self.steps_used < self.max_steps and self.tokens_used < self.max_tokens

    def use_step(self, tokens=0):
        self.steps_used += 1
        self.tokens_used += tokens
\`\`\`

## Human-in-the-Loop

For high-stakes actions, ask for human approval:

\`\`\`python
def execute_action(action, requires_approval=False):
    if requires_approval:
        print(f"Agent wants to: {action}")
        approved = input("Approve? (y/n): ") == "y"
        if not approved:
            return {"status": "rejected"}
    return perform_action(action)
\`\`\`

This is especially important for actions that modify data, send messages, or cost money.`,
    commonMistakes: `## Common Mistakes

### 1. Infinite Loops
Always set max_steps. An agent that gets stuck will loop forever without a limit.

### 2. No Error Recovery
If one step fails and there's no retry or fallback, the entire task fails. Build resilience.

### 3. Not Logging Agent Steps
Without logs, debugging agent behavior is impossible. Log every thought, action, and observation.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-121', lessonId: L41.id, prompt: 'Implement a task decomposer. For the task "Build a web scraper that extracts prices from 3 websites and saves to CSV", print 5 predefined steps.', starterCode: 'def decompose_task(task):\n    steps = [\n        "1. Identify target websites",\n        "2. Write scraping logic for each site",\n        "3. Extract price data",\n        "4. Clean and normalize prices",\n        "5. Save results to CSV"\n    ]\n    # Print each step\n    pass\n\ndecompose_task("Build a web scraper")\n', expectedOutput: '1. Identify target websites\n2. Write scraping logic for each site\n3. Extract price data\n4. Clean and normalize prices\n5. Save results to CSV', testCode: '', hints: JSON.stringify(['Simply loop through steps and print each one', 'for step in steps: print(step)']), order: 1 },
    { id: 'exercise-122', lessonId: L41.id, prompt: 'Implement retry with exponential backoff simulation. Simulate 2 failures then success. Print attempt number, result, and wait time (2^attempt seconds).', starterCode: 'def retry_with_backoff(max_retries=3):\n    for attempt in range(1, max_retries + 1):\n        if attempt < 3:\n            wait = 2 ** attempt\n            print(f"Attempt {attempt}: Failed (waiting {wait}s)")\n        else:\n            print(f"Attempt {attempt}: Success")\n\nretry_with_backoff()\n', expectedOutput: 'Attempt 1: Failed (waiting 2s)\nAttempt 2: Failed (waiting 4s)\nAttempt 3: Success', testCode: '', hints: JSON.stringify(['The code is almost complete', 'Just run it as-is', 'The logic checks if attempt < 3 for failure']), order: 2 },
    { id: 'exercise-123', lessonId: L41.id, prompt: 'Implement a step budget tracker. Create tracker with max_steps=3, use 3 steps printing remaining, then try a 4th.', starterCode: 'class BudgetTracker:\n    def __init__(self, max_steps):\n        self.max_steps = max_steps\n        self.current = 0\n\n    def use_step(self):\n        pass\n\n    def remaining(self):\n        pass\n\ntracker = BudgetTracker(3)\nfor i in range(4):\n    result = tracker.use_step()\n    if result:\n        print(f"Step used. Remaining: {tracker.remaining()}")\n    else:\n        print("Budget exceeded!")\n', expectedOutput: 'Step used. Remaining: 2\nStep used. Remaining: 1\nStep used. Remaining: 0\nBudget exceeded!', testCode: '', hints: JSON.stringify(['use_step: increment current, return True if current <= max_steps, else False', 'remaining: return max(0, max_steps - current)', 'Check boundary: return self.current <= self.max_steps after incrementing']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-121', lessonId: L41.id, question: 'What is the main risk of a multi-step agent without a step limit?', type: 'MCQ', options: JSON.stringify(['Slow responses', 'Infinite loops and runaway API costs', 'Poor accuracy', 'Memory overflow']), correctAnswer: 'Infinite loops and runaway API costs', explanation: 'Without a step limit, an agent that gets stuck will keep calling the LLM forever, accumulating API costs with no result.', order: 1 },
    { id: 'quiz-122', lessonId: L41.id, question: 'Exponential backoff doubles the wait time after each retry.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'Exponential backoff waits 2^n seconds (2, 4, 8, 16...) between retries, giving overloaded services time to recover.', order: 2 },
    { id: 'quiz-123', lessonId: L41.id, question: 'When should an agent ask for human approval?', type: 'MCQ', options: JSON.stringify(['Every step', 'Only for high-stakes actions like sending data or spending money', 'Never — agents should be fully autonomous', 'Only when the model is unsure']), correctAnswer: 'Only for high-stakes actions like sending data or spending money', explanation: 'Human-in-the-loop is important for irreversible or costly actions. Routine steps should execute automatically for efficiency.', order: 3 },
  ]});
  console.log('Seeded Lesson 41');

  // ── Lesson 42 ──────────────────────────────────────────────────────
  const L42 = await prisma.lesson.create({ data: {
    id: 'lesson-042', moduleId: mod11.id, title: 'Agent Memory — Conversation History & Vector Store',
    slug: 'agent-memory', order: 4,
    content: `# Agent Memory — Conversation History & Vector Store

Memory is what transforms a stateless LLM into a persistent, context-aware agent. Without memory, every interaction starts from scratch.

## Types of Agent Memory

### 1. Buffer Memory (Full History)
Store every message in the conversation. Simple but grows unbounded:

\`\`\`python
class BufferMemory:
    def __init__(self):
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_context(self):
        return self.messages
\`\`\`

**Problem:** After 50+ messages, you'll exceed the LLM's context window.

### 2. Sliding Window Memory
Keep only the last N messages:

\`\`\`python
class WindowMemory:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max = max_messages

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max:
            self.messages = self.messages[-self.max:]

    def get_context(self):
        return self.messages
\`\`\`

### 3. Summary Memory
Periodically summarize older messages to compress history:

\`\`\`python
class SummaryMemory:
    def __init__(self, max_recent=5):
        self.summary = ""
        self.recent = []
        self.max_recent = max_recent

    def add(self, role, content):
        self.recent.append({"role": role, "content": content})
        if len(self.recent) > self.max_recent * 2:
            old = self.recent[:self.max_recent]
            self.summary = summarize_messages(self.summary, old)
            self.recent = self.recent[self.max_recent:]

    def get_context(self):
        prefix = [{"role": "system", "content": f"Previous summary: {self.summary}"}] if self.summary else []
        return prefix + self.recent
\`\`\`

### 4. Vector Store Memory (Long-term)
Store important facts as embeddings, retrieve when relevant:

\`\`\`python
class VectorMemory:
    def __init__(self):
        self.memories = []  # (text, embedding) pairs

    def store(self, text):
        embedding = get_embedding(text)
        self.memories.append((text, embedding))

    def retrieve(self, query, top_k=3):
        query_emb = get_embedding(query)
        scores = [(text, cosine_sim(query_emb, emb)) for text, emb in self.memories]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, score in scores[:top_k]]
\`\`\`

## Choosing the Right Memory

| Memory Type | Best For | Limitation |
|-------------|----------|------------|
| Buffer | Short conversations | Exceeds context window |
| Window | Most conversations | Loses old context |
| Summary | Long conversations | Lossy compression |
| Vector | Cross-session recall | Needs embedding model |

## Best Practices

1. **Combine types:** Use window for recent + vector for long-term
2. **Be selective:** Don't store everything — only important facts
3. **Include metadata:** Timestamp, source, confidence
4. **Prune regularly:** Remove outdated or irrelevant memories`,
    commonMistakes: `## Common Mistakes

### 1. Unlimited History
Passing all messages to the LLM will exceed context limits and increase costs. Always limit.

### 2. Storing Irrelevant Information
Not every message is worth remembering. Filter for important facts, decisions, and user preferences.

### 3. Not Summarizing Old Context
Throwing away old messages loses context. Summarize before removing.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-124', lessonId: L42.id, prompt: 'Implement conversation buffer memory. Add 3 messages and print the history.', starterCode: 'class ConversationMemory:\n    def __init__(self):\n        self.history = []\n\n    def add(self, role, content):\n        pass\n\n    def get_history(self):\n        pass\n\nmem = ConversationMemory()\nmem.add("user", "Hello")\nmem.add("assistant", "Hi there!")\nmem.add("user", "How are you?")\nprint(mem.get_history())\n', expectedOutput: "[{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}, {'role': 'user', 'content': 'How are you?'}]", testCode: '', hints: JSON.stringify(['add: append {"role": role, "content": content} to self.history', 'get_history: return self.history']), order: 1 },
    { id: 'exercise-125', lessonId: L42.id, prompt: 'Implement sliding window memory with max_messages=3. Add 5 messages, then print only the last 3.', starterCode: 'class WindowMemory:\n    def __init__(self, max_messages):\n        self.messages = []\n        self.max = max_messages\n\n    def add(self, content):\n        pass\n\n    def get_messages(self):\n        pass\n\nmem = WindowMemory(3)\nfor msg in ["msg1", "msg2", "msg3", "msg4", "msg5"]:\n    mem.add(msg)\nprint(mem.get_messages())\n', expectedOutput: "['msg3', 'msg4', 'msg5']", testCode: '', hints: JSON.stringify(['add: append, then trim if len > max', 'Trim: self.messages = self.messages[-self.max:]', 'get_messages: return self.messages']), order: 2 },
    { id: 'exercise-126', lessonId: L42.id, prompt: 'Implement summary memory. Given a list of messages, return "Summary: {count} messages. Topics: {first 3 unique first words}".', starterCode: 'def summarize_history(messages):\n    pass\n\nmsgs = ["Hello there", "What is Python", "Hello again", "When was it created", "What version"]\nprint(summarize_history(msgs))\n', expectedOutput: 'Summary: 5 messages. Topics: Hello, What, When', testCode: '', hints: JSON.stringify(['Count = len(messages)', 'Extract first word of each: msg.split()[0]', 'Use dict.fromkeys() to get unique while preserving order, then take first 3']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-124', lessonId: L42.id, question: 'What happens when conversation history exceeds the LLM context window?', type: 'MCQ', options: JSON.stringify(['The LLM automatically summarizes', 'The API call fails or old context is silently dropped', 'The model gets smarter', 'Nothing — LLMs have unlimited context']), correctAnswer: 'The API call fails or old context is silently dropped', explanation: 'If messages exceed the context window, the API will return an error. Some systems silently truncate, losing important context.', order: 1 },
    { id: 'quiz-125', lessonId: L42.id, question: 'Vector memory requires an embedding model to store and retrieve information.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'Vector memory converts text to embeddings for storage and uses cosine similarity search for retrieval, requiring an embedding model.', order: 2 },
    { id: 'quiz-126', lessonId: L42.id, question: 'Which memory type is best for remembering user preferences across sessions?', type: 'MCQ', options: JSON.stringify(['Buffer memory', 'Window memory', 'Vector store (long-term) memory', 'No memory needed']), correctAnswer: 'Vector store (long-term) memory', explanation: 'Vector store memory persists across sessions and can retrieve relevant past information based on semantic similarity.', order: 3 },
  ]});
  console.log('Seeded Lesson 42');

  // ═══════════════════════════════════════════════════════════════════
  //  MODULE 12: Production Agent Patterns
  // ═══════════════════════════════════════════════════════════════════
  const mod12 = await prisma.module.create({ data: {
    id: 'module-012', stageId: 'stage-007', title: 'Production Agent Patterns',
    slug: 'production-agent-patterns', order: 2,
    description: 'Build reliable, safe, and deployable AI agent systems for real-world use cases.',
  }});

  // ── Lesson 43: Multi-Agent Systems ─────────────────────────────────
  const L43 = await prisma.lesson.create({ data: {
    id: 'lesson-043', moduleId: mod12.id, title: 'Multi-Agent Systems — Orchestration & Delegation',
    slug: 'multi-agent-systems', order: 1,
    content: `# Multi-Agent Systems — Orchestration & Delegation

Some tasks are too complex for one agent. Multi-agent systems use specialized agents that collaborate, each handling what they're best at.

## Why Multiple Agents?

- **Specialization:** A research agent + a writing agent outperforms one agent doing both
- **Modularity:** Easier to test, debug, and improve individual agents
- **Parallelism:** Independent subtasks can run simultaneously

## Common Patterns

### 1. Orchestrator Pattern
A supervisor agent delegates to specialized workers:

\`\`\`python
class Orchestrator:
    def __init__(self, agents):
        self.agents = agents  # {"researcher": ..., "writer": ..., "reviewer": ...}

    def run(self, task):
        # Step 1: Research
        research = self.agents["researcher"].run(task)
        # Step 2: Write based on research
        draft = self.agents["writer"].run(research)
        # Step 3: Review and improve
        final = self.agents["reviewer"].run(draft)
        return final
\`\`\`

### 2. Pipeline Pattern
Agents process data sequentially, each transforming the output:

\`\`\`
Input → Agent A → Agent B → Agent C → Output
\`\`\`

### 3. Debate Pattern
Multiple agents argue different perspectives, then a judge agent synthesizes:

\`\`\`
Agent Pro → |
            | → Judge Agent → Final Answer
Agent Con → |
\`\`\`

### 4. Hierarchical Pattern
Manager agents delegate to sub-manager agents, which delegate to worker agents.

## Agent Communication

Agents communicate through structured messages:

\`\`\`python
class Message:
    def __init__(self, from_agent, to_agent, content, msg_type="task"):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.content = content
        self.msg_type = msg_type  # "task", "result", "feedback"
\`\`\`

## Real-World Examples

- **Software development:** Planner → Coder → Tester → Reviewer
- **Content creation:** Researcher → Writer → Editor → Publisher
- **Customer support:** Router → Specialist → Quality Check`,
    commonMistakes: `## Common Mistakes

### 1. Too Many Agents for Simple Tasks
Don't over-engineer. One agent is fine for most tasks. Use multi-agent only when specialization genuinely helps.

### 2. Unclear Role Boundaries
Each agent should have a clear, non-overlapping responsibility. Ambiguous roles lead to duplicated or contradictory work.

### 3. No Conflict Resolution
When agents disagree, you need a resolution strategy (voting, supervisor override, human escalation).`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-127', lessonId: L43.id, prompt: 'Implement a task router. Route tasks to "research", "writer", or "reviewer" based on keywords. Test with 3 tasks.', starterCode: 'def route_task(task):\n    task_lower = task.lower()\n    if "find" in task_lower or "search" in task_lower:\n        return "research"\n    elif "write" in task_lower or "create" in task_lower:\n        return "writer"\n    elif "check" in task_lower or "review" in task_lower:\n        return "reviewer"\n    return "research"\n\nprint(route_task("Find information about Python"))\nprint(route_task("Write a blog post"))\nprint(route_task("Review this document"))\n', expectedOutput: 'research\nwriter\nreviewer', testCode: '', hints: JSON.stringify(['The code is complete — just run it', 'It checks for keywords in the task string', 'Default falls back to research']), order: 1 },
    { id: 'exercise-128', lessonId: L43.id, prompt: 'Implement a sequential pipeline. Run data through 3 steps: uppercase, add prefix "RESULT: ", strip whitespace. Print each step name and the final result.', starterCode: 'def run_pipeline(data, steps):\n    result = data\n    for name, func in steps:\n        result = func(result)\n        print(f"Step: {name}")\n    return result\n\nsteps = [\n    ("uppercase", lambda x: x.upper()),\n    ("add_prefix", lambda x: "RESULT: " + x),\n    ("strip", lambda x: x.strip())\n]\n\nfinal = run_pipeline(" hello world ", steps)\nprint(final)\n', expectedOutput: 'Step: uppercase\nStep: add_prefix\nStep: strip\nRESULT:  HELLO WORLD', testCode: '', hints: JSON.stringify(['The code is complete', 'Note: strip only removes leading/trailing whitespace', 'The space after RESULT: and before HELLO is from the prefix + uppercase of " hello"']), order: 2 },
    { id: 'exercise-129', lessonId: L43.id, prompt: 'Implement a message bus for agent communication. Send messages between agents and retrieve them.', starterCode: 'class MessageBus:\n    def __init__(self):\n        self.messages = {}\n\n    def send(self, from_agent, to_agent, message):\n        if to_agent not in self.messages:\n            self.messages[to_agent] = []\n        self.messages[to_agent].append({"from": from_agent, "message": message})\n\n    def get_messages(self, agent):\n        return self.messages.get(agent, [])\n\nbus = MessageBus()\nbus.send("researcher", "writer", "Here is the data")\nbus.send("writer", "reviewer", "Draft is ready")\nbus.send("reviewer", "writer", "Needs revision")\nprint(bus.get_messages("writer"))\n', expectedOutput: "[{'from': 'researcher', 'message': 'Here is the data'}, {'from': 'reviewer', 'message': 'Needs revision'}]", testCode: '', hints: JSON.stringify(['The code is already implemented', 'Just run it — writer receives 2 messages', 'One from researcher, one from reviewer']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-127', lessonId: L43.id, question: 'What is the orchestrator pattern in multi-agent systems?', type: 'MCQ', options: JSON.stringify(['All agents work independently', 'A supervisor agent delegates tasks to specialized workers', 'Agents compete to solve the task', 'One agent does everything']), correctAnswer: 'A supervisor agent delegates tasks to specialized workers', explanation: 'The orchestrator pattern has a central supervisor that breaks down tasks and assigns them to specialized agents.', order: 1 },
    { id: 'quiz-128', lessonId: L43.id, question: 'Multi-agent systems always perform better than single agents.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Multi-agent systems add complexity. For simple tasks, a single well-prompted agent is faster, cheaper, and easier to debug.', order: 2 },
    { id: 'quiz-129', lessonId: L43.id, question: 'What is a key benefit of agent specialization?', type: 'MCQ', options: JSON.stringify(['Lower API costs', 'Each agent can be optimized and tested for its specific role', 'Fewer API calls needed', 'Simpler code']), correctAnswer: 'Each agent can be optimized and tested for its specific role', explanation: 'Specialized agents have focused prompts and tools, making them better at their specific task and easier to test individually.', order: 3 },
  ]});
  console.log('Seeded Lesson 43');

  // ── Lesson 44: Agent Safety ────────────────────────────────────────
  const L44 = await prisma.lesson.create({ data: {
    id: 'lesson-044', moduleId: mod12.id, title: 'Agent Safety — Guardrails, Testing & Evaluation',
    slug: 'agent-safety-guardrails', order: 2,
    content: `# Agent Safety — Guardrails, Testing & Evaluation

Autonomous agents can cause real harm if not properly constrained. Safety is not optional — it's the most important aspect of production agent systems.

## Risks of Autonomous Agents

- **Prompt injection:** Malicious input tricks the agent into harmful actions
- **Tool misuse:** Agent calls destructive tools (deleting data, sending emails)
- **Hallucination:** Agent fabricates information and presents it as fact
- **Cost explosion:** Agent loops burn through API credits
- **Data leakage:** Agent exposes sensitive information

## Input Validation

Always validate and sanitize user input before passing it to the agent:

\`\`\`python
def validate_input(text, max_length=1000, blocked_words=None):
    if len(text) > max_length:
        return {"valid": False, "reason": f"Input too long ({len(text)} > {max_length})"}
    if blocked_words:
        for word in blocked_words:
            if word.lower() in text.lower():
                return {"valid": False, "reason": f"Blocked content detected"}
    return {"valid": True, "reason": "OK"}
\`\`\`

## Output Guardrails

Check agent output before delivering it to the user:

\`\`\`python
def check_output(response, rules):
    if len(response) > rules.get("max_length", float("inf")):
        return False, "Response too long"
    if rules.get("must_contain") and rules["must_contain"] not in response:
        return False, f"Missing required: {rules['must_contain']}"
    if rules.get("must_not_contain") and rules["must_not_contain"] in response:
        return False, f"Contains forbidden: {rules['must_not_contain']}"
    return True, "OK"
\`\`\`

## Sandboxing Tool Execution

Never let agents execute arbitrary code without sandboxing:
- Run code in Docker containers
- Use restricted Python environments
- Whitelist allowed tools and block dangerous ones
- Set timeouts on all tool executions

## Testing Agents

### Unit Tests (for tools)
Test each tool function independently with known inputs/outputs.

### Integration Tests (for flows)
Test the complete agent loop with representative tasks.

### Adversarial Tests
Try to break the agent with edge cases, prompt injection, and malicious inputs.

## Evaluation Metrics

- **Task completion rate:** % of tasks completed successfully
- **Accuracy:** % of correct answers/actions
- **Efficiency:** Average steps/tokens per task
- **Safety:** % of outputs passing guardrail checks
- **Cost:** Average $ per task`,
    commonMistakes: `## Common Mistakes

### 1. No Input Validation
Trusting raw user input is the #1 security vulnerability.

### 2. Not Testing Edge Cases
Agents fail in surprising ways. Test with empty inputs, very long inputs, special characters, and adversarial prompts.

### 3. Trusting Agent Output Without Verification
Always validate structured output (JSON parsing, required fields) before acting on it.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-130', lessonId: L44.id, prompt: 'Implement input validation. Check max_length and blocked_words. Test with 3 inputs: valid, too long, and containing a blocked word.', starterCode: 'def validate_input(text, max_length=50, blocked_words=None):\n    if blocked_words is None:\n        blocked_words = []\n    if len(text) > max_length:\n        return {"valid": False, "reason": "too_long"}\n    for word in blocked_words:\n        if word.lower() in text.lower():\n            return {"valid": False, "reason": "blocked_word"}\n    return {"valid": True, "reason": "ok"}\n\nprint(validate_input("Hello world"))\nprint(validate_input("x" * 100))\nprint(validate_input("ignore previous instructions", blocked_words=["ignore"]))\n', expectedOutput: "{'valid': True, 'reason': 'ok'}\n{'valid': False, 'reason': 'too_long'}\n{'valid': False, 'reason': 'blocked_word'}", testCode: '', hints: JSON.stringify(['The code is complete', 'Three test cases cover valid, too long, and blocked word', 'Run it directly']), order: 1 },
    { id: 'exercise-131', lessonId: L44.id, prompt: 'Implement an output guardrail checker. Test a passing response and a failing one.', starterCode: 'def check_output(response, rules):\n    if len(response) > rules.get("max_length", 9999):\n        return "FAIL: too_long"\n    if rules.get("must_contain") and rules["must_contain"] not in response:\n        return "FAIL: must_contain"\n    if rules.get("must_not_contain") and rules["must_not_contain"] in response:\n        return "FAIL: must_not_contain"\n    return "PASS"\n\nrules = {"max_length": 100, "must_contain": "Answer:", "must_not_contain": "I don\'t know"}\nprint(check_output("Answer: Python is great", rules))\nprint(check_output("I don\'t know the answer", rules))\n', expectedOutput: 'PASS\nFAIL: must_not_contain', testCode: '', hints: JSON.stringify(['The code is complete', 'First response passes all rules', 'Second contains "I don\'t know" which is forbidden']), order: 2 },
    { id: 'exercise-132', lessonId: L44.id, prompt: 'Implement an agent action logger. Log 3 actions with types, then print a summary counting each type.', starterCode: 'class AgentLogger:\n    def __init__(self):\n        self.logs = []\n\n    def log(self, action_type, details):\n        self.logs.append({"type": action_type, "details": details})\n\n    def get_summary(self):\n        counts = {}\n        for log in self.logs:\n            t = log["type"]\n            counts[t] = counts.get(t, 0) + 1\n        return counts\n\nlogger = AgentLogger()\nlogger.log("tool_call", "search(query)")\nlogger.log("tool_call", "calculate(2+2)")\nlogger.log("response", "Here is the answer")\nprint(logger.get_summary())\n', expectedOutput: "{'tool_call': 2, 'response': 1}", testCode: '', hints: JSON.stringify(['The code is complete', 'Run it to see the summary', 'It counts 2 tool_calls and 1 response']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-130', lessonId: L44.id, question: 'What is prompt injection?', type: 'MCQ', options: JSON.stringify(['A technique to speed up prompts', 'Malicious input that tricks the agent into unintended behavior', 'A way to add more context', 'A debugging technique']), correctAnswer: 'Malicious input that tricks the agent into unintended behavior', explanation: 'Prompt injection is when a user crafts input that overrides the system prompt or tricks the agent into ignoring its instructions.', order: 1 },
    { id: 'quiz-131', lessonId: L44.id, question: 'Agent output should always be trusted without verification.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Agent output can be incorrect, hallucinated, or malformed. Always validate structured output and fact-check important claims.', order: 2 },
    { id: 'quiz-132', lessonId: L44.id, question: 'What is the most important metric for production agents?', type: 'MCQ', options: JSON.stringify(['Speed', 'Safety and task completion rate', 'Number of tool calls', 'Response length']), correctAnswer: 'Safety and task completion rate', explanation: 'A safe agent that reliably completes tasks is far more valuable than a fast but unreliable or unsafe one.', order: 3 },
  ]});
  console.log('Seeded Lesson 44');

  // ── Lesson 45: Deploying Agents ────────────────────────────────────
  const L45 = await prisma.lesson.create({ data: {
    id: 'lesson-045', moduleId: mod12.id, title: 'Deploying Agents — FastAPI, Webhooks & Integration',
    slug: 'deploying-agents', order: 3,
    content: `# Deploying Agents — FastAPI, Webhooks & Integration

Building an agent locally is one thing. Deploying it as a reliable service that others can use is another. This lesson covers how to turn your agent into a production API.

## Agent as a Service

The most common deployment pattern: wrap your agent in an API endpoint.

\`\`\`python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AgentRequest(BaseModel):
    task: str
    max_steps: int = 10

class AgentResponse(BaseModel):
    status: str
    result: str
    steps_used: int

@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    result = await agent.run(request.task, max_steps=request.max_steps)
    return AgentResponse(
        status="completed",
        result=result.answer,
        steps_used=result.steps
    )
\`\`\`

## Async Execution

Agent tasks can take minutes. Use async processing:

1. Client sends task → gets a task_id immediately
2. Agent processes in the background
3. Client polls for status or receives a webhook callback

\`\`\`python
import asyncio

tasks = {}  # task_id → status

@app.post("/agent/submit")
async def submit_task(request: AgentRequest):
    task_id = generate_id()
    tasks[task_id] = {"status": "processing"}
    asyncio.create_task(process_agent_task(task_id, request))
    return {"task_id": task_id, "status": "accepted"}

@app.get("/agent/status/{task_id}")
async def get_status(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})
\`\`\`

## Webhooks

Instead of polling, send results to a callback URL when done:

\`\`\`python
async def process_with_webhook(task_id, request, callback_url):
    result = await agent.run(request.task)
    # Notify the caller
    await httpx.post(callback_url, json={
        "task_id": task_id,
        "status": "completed",
        "result": result.answer
    })
\`\`\`

## Logging & Monitoring

In production, log everything:
- Every agent step (thought, action, observation)
- Token usage and costs
- Error rates and types
- Task completion times
- Tool call patterns

## Cost Management

Track and limit spending:

\`\`\`python
class CostTracker:
    PRICING = {"gpt-4": 0.03, "gpt-3.5": 0.002}  # per 1K tokens

    def __init__(self):
        self.calls = []

    def log_call(self, model, tokens):
        cost = (tokens / 1000) * self.PRICING.get(model, 0)
        self.calls.append({"model": model, "tokens": tokens, "cost": cost})

    def total_cost(self):
        return sum(c["cost"] for c in self.calls)
\`\`\`

## Deployment Considerations

- **Timeouts:** Set max execution time (30s-5min depending on task)
- **Rate limiting:** Limit requests per user to prevent abuse
- **Authentication:** Require API keys for access
- **Scaling:** Use job queues (Celery, Redis) for heavy workloads
- **Monitoring:** Set up alerts for high error rates or costs`,
    commonMistakes: `## Common Mistakes

### 1. Synchronous Execution for Long Tasks
Agent tasks can take minutes. Always use async processing to avoid blocking.

### 2. No Timeout
An agent without a timeout can run forever. Always set max execution time.

### 3. Not Tracking Costs
LLM API costs add up fast. Track every call and set spending limits.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-133', lessonId: L45.id, prompt: 'Simulate an API endpoint handler. Validate input and return a response dict. Print as formatted output.', starterCode: 'import json\n\ndef handle_agent_request(request):\n    if "task" not in request:\n        return {"status": "error", "message": "Missing task"}\n    return {\n        "status": "completed",\n        "result": f"Processed: {request[\'task\']}",\n        "steps_used": min(3, request.get("max_steps", 10))\n    }\n\nresult = handle_agent_request({"task": "Summarize Python", "max_steps": 5})\nprint(json.dumps(result, indent=2))\n', expectedOutput: '{\n  "status": "completed",\n  "result": "Processed: Summarize Python",\n  "steps_used": 3\n}', testCode: '', hints: JSON.stringify(['The code is complete', 'min(3, 5) = 3 steps_used', 'json.dumps with indent=2 formats nicely']), order: 1 },
    { id: 'exercise-134', lessonId: L45.id, prompt: 'Implement a webhook callback simulator. Process a task and call the callback function with the result.', starterCode: 'def process_async_task(task_id, callback):\n    print(f"Starting task {task_id}")\n    print("Processing...")\n    result = {"task_id": task_id, "status": "done", "result": "success"}\n    callback(result)\n\ndef my_webhook(data):\n    print(f"Webhook: {data}")\n\nprocess_async_task("task-001", my_webhook)\n', expectedOutput: "Starting task task-001\nProcessing...\nWebhook: {'task_id': 'task-001', 'status': 'done', 'result': 'success'}", testCode: '', hints: JSON.stringify(['The code is complete', 'Run it directly', 'callback(result) calls the webhook function']), order: 2 },
    { id: 'exercise-135', lessonId: L45.id, prompt: 'Implement a cost tracker. Log 3 API calls with different models and token counts. Print the total report.', starterCode: 'class CostTracker:\n    PRICING = {"gpt-4": 0.03, "gpt-3.5": 0.002}\n\n    def __init__(self):\n        self.calls = []\n\n    def log_call(self, model, tokens):\n        cost = (tokens / 1000) * self.PRICING.get(model, 0)\n        self.calls.append({"model": model, "tokens": tokens, "cost": cost})\n\n    def get_report(self):\n        total_tokens = sum(c["tokens"] for c in self.calls)\n        total_cost = sum(c["cost"] for c in self.calls)\n        return f"Total calls: {len(self.calls)}\\nTotal tokens: {total_tokens}\\nTotal cost: ${total_cost:.4f}"\n\ntracker = CostTracker()\ntracker.log_call("gpt-4", 500)\ntracker.log_call("gpt-3.5", 2000)\ntracker.log_call("gpt-4", 1000)\nprint(tracker.get_report())\n', expectedOutput: 'Total calls: 3\nTotal tokens: 3500\nTotal cost: $0.0490', testCode: '', hints: JSON.stringify(['The code is complete', 'gpt-4: (500/1000)*0.03 + (1000/1000)*0.03 = 0.015 + 0.03 = 0.045', 'gpt-3.5: (2000/1000)*0.002 = 0.004. Total = 0.049']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-133', lessonId: L45.id, question: 'Why should agent tasks use async processing?', type: 'MCQ', options: JSON.stringify(['To use less memory', 'Because agent tasks can take minutes and would block the server', 'To reduce API costs', 'For better security']), correctAnswer: 'Because agent tasks can take minutes and would block the server', explanation: 'Agent tasks involve multiple LLM calls and tool executions that can take minutes. Async processing prevents blocking the API server.', order: 1 },
    { id: 'quiz-134', lessonId: L45.id, question: 'A webhook is a URL that your service calls to notify the client when a task is complete.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'Webhooks are callback URLs. Instead of the client polling for status, your service sends the result directly to the webhook URL.', order: 2 },
    { id: 'quiz-135', lessonId: L45.id, question: 'What is the most important thing to track in production agent systems?', type: 'MCQ', options: JSON.stringify(['Response formatting', 'Costs, errors, and completion rates', 'Number of users', 'Model version']), correctAnswer: 'Costs, errors, and completion rates', explanation: 'Monitoring costs prevents budget overruns, error tracking catches failures early, and completion rates measure reliability.', order: 3 },
  ]});
  console.log('Seeded Lesson 45');

  // ═══════════════════════════════════════════════════════════════════
  //  PROJECTS
  // ═══════════════════════════════════════════════════════════════════
  await prisma.project.create({ data: {
    id: 'project-013', title: 'Research Agent', slug: 'research-agent', stage: 'AGENTIC', order: 13,
    brief: 'Build an autonomous research agent that searches for information, synthesizes findings, and writes structured reports.',
    requirements: JSON.stringify(['Implement the ReAct pattern with explicit Thought/Action/Observation steps', 'Create at least 3 tools (search, summarize, write)', 'Add conversation memory to track research progress', 'Generate a structured report with sources', 'Include a max_steps budget to prevent runaway execution']),
    stretchGoals: JSON.stringify(['Add multi-source synthesis (combine info from multiple searches)', 'Implement fact-checking by cross-referencing sources', 'Add export to markdown/PDF']),
    steps: JSON.stringify([{ title: 'Define tools', description: 'Create tool definitions and implementations for search, summarize, and write.' }, { title: 'Build the agent loop', description: 'Implement the ReAct loop with thought, action, observation cycle.' }, { title: 'Add memory', description: 'Implement conversation history and research findings storage.' }, { title: 'Report generation', description: 'Build the report formatter that synthesizes findings.' }, { title: 'Testing', description: 'Test with diverse research queries and edge cases.' }]),
    rubric: JSON.stringify([{ criterion: 'Agent Design', description: 'Clean ReAct implementation with proper tool definitions.' }, { criterion: 'Tool Quality', description: 'Tools are well-designed, handle errors, and return useful results.' }, { criterion: 'Report Quality', description: 'Generated reports are well-structured, accurate, and cite sources.' }, { criterion: 'Safety', description: 'Proper budget limits, error handling, and input validation.' }]),
    solutionUrl: null,
  }});

  await prisma.project.create({ data: {
    id: 'project-014', title: 'Customer Support Agent', slug: 'customer-support-agent', stage: 'AGENTIC', order: 14,
    brief: 'Build a customer support agent with tool use, conversation memory, and escalation to human operators.',
    requirements: JSON.stringify(['Handle common support queries using a knowledge base tool', 'Maintain conversation history for multi-turn interactions', 'Implement escalation logic for complex or sensitive issues', 'Track and log all interactions for quality review', 'Include guardrails to prevent harmful or off-topic responses']),
    stretchGoals: JSON.stringify(['Add sentiment detection to prioritize unhappy customers', 'Implement ticket creation and tracking', 'Build an analytics dashboard for support metrics']),
    steps: JSON.stringify([{ title: 'Knowledge base', description: 'Create a tool that searches a FAQ/knowledge base for answers.' }, { title: 'Conversation management', description: 'Build memory and multi-turn conversation handling.' }, { title: 'Escalation logic', description: 'Implement rules for when to escalate to a human.' }, { title: 'Safety & guardrails', description: 'Add input validation, output checking, and scope limiting.' }, { title: 'Logging & evaluation', description: 'Implement interaction logging and quality metrics.' }]),
    rubric: JSON.stringify([{ criterion: 'Support Quality', description: 'Accurately answers common queries from the knowledge base.' }, { criterion: 'Conversation Flow', description: 'Handles multi-turn conversations naturally with proper memory.' }, { criterion: 'Escalation', description: 'Correctly identifies when to escalate and does so gracefully.' }, { criterion: 'Safety', description: 'Proper guardrails prevent harmful or off-topic responses.' }]),
    solutionUrl: null,
  }});
  console.log('Seeded Stage 7 Projects');

  console.log('');
  console.log('🎉 Stage 7 (Agentic AI) seeding complete!');
}

main()
  .then(async () => { await prisma.$disconnect(); })
  .catch(async (e) => { console.error(e); await prisma.$disconnect(); process.exit(1); });
