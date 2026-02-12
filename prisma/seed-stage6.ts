export {};

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  const existing = await prisma.stage.findUnique({ where: { id: 'stage-006' } });
  if (existing) { console.log('Stage 6 already seeded. Skipping.'); return; }

  console.log('═══════════════════════════════════════════');
  console.log('  Seeding Stage 6: NLP & LLMs');
  console.log('═══════════════════════════════════════════');

  // ─── Skill Tags ────────────────────────────────────────────────────
  const skills = [
    { id: 'skill-040', name: 'NLP', slug: 'nlp' },
    { id: 'skill-041', name: 'Text Preprocessing', slug: 'text-preprocessing' },
    { id: 'skill-042', name: 'Embeddings', slug: 'embeddings' },
    { id: 'skill-043', name: 'Transformers', slug: 'transformers' },
    { id: 'skill-044', name: 'LLM APIs', slug: 'llm-apis' },
    { id: 'skill-045', name: 'Prompt Engineering', slug: 'prompt-engineering' },
    { id: 'skill-046', name: 'RAG', slug: 'rag' },
  ];
  for (const s of skills) {
    await prisma.skillTag.upsert({ where: { id: s.id }, update: {}, create: s });
  }
  console.log('Created skill tags');

  // ─── Stage ─────────────────────────────────────────────────────────
  await prisma.stage.create({
    data: {
      id: 'stage-006',
      title: 'NLP & LLMs',
      slug: 'nlp-llms',
      description: 'Work with text data and large language models — the hottest skill in the market. From text preprocessing to building RAG systems.',
      order: 6,
    },
  });

  // ═══════════════════════════════════════════════════════════════════
  //  MODULE 9: Natural Language Processing
  // ═══════════════════════════════════════════════════════════════════
  const mod9 = await prisma.module.create({
    data: {
      id: 'module-009',
      stageId: 'stage-006',
      title: 'Natural Language Processing',
      slug: 'natural-language-processing',
      description: 'Learn the fundamentals of processing and understanding text data.',
      order: 1,
    },
  });

  // ── Lesson 33 ──────────────────────────────────────────────────────
  const lesson33 = await prisma.lesson.create({
    data: {
      id: 'lesson-033',
      moduleId: mod9.id,
      title: 'Text Preprocessing — Tokenization, Stemming, TF-IDF',
      slug: 'text-preprocessing',
      order: 1,
      content: `# Text Preprocessing — Tokenization, Stemming, TF-IDF

Before a machine learning model can understand text, we need to convert raw text into numbers. This process — text preprocessing — is the foundation of all NLP work.

## Why Preprocessing Matters

Raw text is messy: it has punctuation, mixed case, stop words, and inconsistent formatting. A model trained on raw text will treat "Python", "python", and "PYTHON" as three different words. Preprocessing cleans and normalizes text so models can focus on meaning.

## Tokenization

Tokenization splits text into individual units (tokens). The most common approach is **word tokenization**:

\`\`\`python
text = "Hello, World! This is NLP."
# Simple tokenization
tokens = text.lower().split()
# ['hello,', 'world!', 'this', 'is', 'nlp.']

# Better: remove punctuation first
import re
tokens = re.findall(r'\\b\\w+\\b', text.lower())
# ['hello', 'world', 'this', 'is', 'nlp']
\`\`\`

**Sentence tokenization** splits text into sentences, typically by splitting on period/exclamation/question marks.

## Lowercasing

Always lowercase text for consistency (unless case matters for your task):

\`\`\`python
text = "Python is GREAT"
text = text.lower()  # "python is great"
\`\`\`

## Stop Words Removal

Stop words are common words ("the", "is", "a", "an") that add little meaning:

\`\`\`python
stop_words = {"the", "is", "a", "an", "in", "on", "at", "to", "and", "or", "of"}
tokens = ["the", "cat", "is", "on", "the", "mat"]
filtered = [t for t in tokens if t not in stop_words]
# ['cat', 'mat']
\`\`\`

## Stemming vs Lemmatization

**Stemming** chops word endings to find the root (fast but crude):
- "running" → "run", "better" → "bet" (wrong!), "studies" → "studi"

**Lemmatization** uses a dictionary to find the proper base form (slower but accurate):
- "running" → "run", "better" → "good", "studies" → "study"

\`\`\`python
# Simple suffix-based stemmer
def simple_stem(word):
    suffixes = ['ing', 'ed', 'ly', 'er', 'est', 's']
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word
\`\`\`

## Bag of Words

The simplest text representation: count how many times each word appears:

\`\`\`python
from collections import Counter

doc = "the cat sat on the mat the cat"
bow = Counter(doc.split())
# {'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1}
\`\`\`

## Term Frequency (TF)

TF measures how often a term appears in a document, normalized by document length:

\`\`\`python
def compute_tf(document):
    words = document if isinstance(document, list) else document.split()
    word_count = Counter(words)
    total = len(words)
    return {word: count / total for word, count in word_count.items()}
\`\`\`

## TF-IDF

**TF-IDF** (Term Frequency - Inverse Document Frequency) balances term frequency with how unique a term is across all documents. Words that appear in every document (like "the") get low scores; rare but frequent terms get high scores.

**Formula:** TF-IDF(t, d) = TF(t, d) × IDF(t)

Where IDF(t) = log(N / df(t)), N = total documents, df(t) = documents containing term t.

## N-grams

N-grams are sequences of N consecutive words:
- Unigrams (1): ["the", "cat", "sat"]
- Bigrams (2): ["the cat", "cat sat"]
- Trigrams (3): ["the cat sat"]

\`\`\`python
def get_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
\`\`\`

These capture word relationships that single words miss: "not good" as a bigram carries different meaning than "not" and "good" separately.`,
      commonMistakes: `## Common Mistakes

### 1. Not Lowercasing Before Comparison
"Python" and "python" will be treated as different tokens if you forget to lowercase.

### 2. Removing Too Many Stop Words
Some stop words carry meaning in context: "not good" loses its meaning if you remove "not".

### 3. Confusing Stemming and Lemmatization
Stemming is a crude rule-based approach. Use lemmatization when accuracy matters.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-097',
        lessonId: lesson33.id,
        prompt: 'Write a function `tokenize(text)` that lowercases text and extracts only alphanumeric words (no punctuation). Test with "Hello, World! This is NLP."',
        starterCode: 'import re\n\ndef tokenize(text):\n    # Your code here\n    pass\n\nprint(tokenize("Hello, World! This is NLP."))\n',
        expectedOutput: "['hello', 'world', 'this', 'is', 'nlp']",
        testCode: '',
        hints: JSON.stringify(['Use re.findall() with pattern r"\\b\\w+\\b"', 'Lowercase the text first with text.lower()', 'return re.findall(r"\\b\\w+\\b", text.lower())']),
        order: 1,
      },
      {
        id: 'exercise-098',
        lessonId: lesson33.id,
        prompt: 'Implement `compute_tf(words)` that takes a list of words and returns a dict of term frequencies (count/total), rounded to 4 decimals. Test with ["the", "cat", "sat", "on", "the", "mat"]. Print sorted by key.',
        starterCode: 'def compute_tf(words):\n    # Your code here\n    pass\n\nwords = ["the", "cat", "sat", "on", "the", "mat"]\ntf = compute_tf(words)\nfor k in sorted(tf):\n    print(f"{k}: {tf[k]}")\n',
        expectedOutput: 'cat: 0.1667\nmat: 0.1667\non: 0.1667\nsat: 0.1667\nthe: 0.3333',
        testCode: '',
        hints: JSON.stringify(['Count each word with a dict or Counter', 'Divide each count by len(words)', 'Round to 4 decimals: round(count/total, 4)']),
        order: 2,
      },
      {
        id: 'exercise-099',
        lessonId: lesson33.id,
        prompt: 'Implement `remove_stopwords(tokens, stopwords)` that filters out stop words. Test with tokens=["the","quick","brown","fox","is","very","fast"] and stopwords={"the","is","very","a","an"}.',
        starterCode: 'def remove_stopwords(tokens, stopwords):\n    pass\n\ntokens = ["the", "quick", "brown", "fox", "is", "very", "fast"]\nstopwords = {"the", "is", "very", "a", "an"}\nprint(remove_stopwords(tokens, stopwords))\n',
        expectedOutput: "['quick', 'brown', 'fox', 'fast']",
        testCode: '',
        hints: JSON.stringify(['Use a list comprehension', 'Filter: [t for t in tokens if t not in stopwords]']),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      { id: 'quiz-097', lessonId: lesson33.id, question: 'What does TF-IDF stand for?', type: 'MCQ', options: JSON.stringify(['Text Format - Index Data Format', 'Term Frequency - Inverse Document Frequency', 'Token Filter - Indexed Data Features', 'Text Feature - Inverse Data Frequency']), correctAnswer: 'Term Frequency - Inverse Document Frequency', explanation: 'TF-IDF combines how often a term appears in a document (TF) with how rare it is across all documents (IDF).', order: 1 },
      { id: 'quiz-098', lessonId: lesson33.id, question: 'Stemming always produces valid dictionary words.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Stemming uses crude rules to chop suffixes, which often produces non-words (e.g., "studies" → "studi"). Lemmatization produces valid words.', order: 2 },
      { id: 'quiz-099', lessonId: lesson33.id, question: 'What is a bigram?', type: 'MCQ', options: JSON.stringify(['A two-letter word', 'A sequence of two consecutive tokens', 'A binary encoding of text', 'A pair of documents']), correctAnswer: 'A sequence of two consecutive tokens', explanation: 'An n-gram is a sequence of n consecutive tokens. A bigram (n=2) captures pairs like "machine learning" or "not good".', order: 3 },
    ],
  });
  console.log('Seeded Lesson 33');

  // ── Lesson 34 ──────────────────────────────────────────────────────
  const lesson34 = await prisma.lesson.create({
    data: {
      id: 'lesson-034',
      moduleId: mod9.id,
      title: 'Word Embeddings — Word2Vec & GloVe',
      slug: 'word-embeddings',
      order: 2,
      content: `# Word Embeddings — Word2Vec & GloVe

Bag-of-words and TF-IDF represent words as sparse, high-dimensional vectors where each dimension is a unique word. They capture frequency but miss **meaning**. Word embeddings solve this by mapping words to dense, low-dimensional vectors where similar words are close together.

## The Problem with One-Hot Encoding

In one-hot encoding, "king" = [1,0,0,...], "queen" = [0,1,0,...]. These vectors are orthogonal — the model sees no relationship between "king" and "queen". With 100,000 words, each vector has 100,000 dimensions, mostly zeros.

## What Are Embeddings?

An embedding maps each word to a dense vector (typically 50-300 dimensions) learned from data:

\`\`\`python
# Conceptual example
"king"  → [0.2, -0.4, 0.7, 0.1, ...]  # 300 dimensions
"queen" → [0.21, -0.38, 0.69, 0.12, ...] # Similar!
"car"   → [-0.5, 0.3, -0.1, 0.8, ...]  # Very different
\`\`\`

The famous example: king - man + woman ≈ queen. Embeddings capture semantic relationships as vector arithmetic.

## Word2Vec

Word2Vec (Google, 2013) learns embeddings by predicting words from context. Two architectures:

**CBOW (Continuous Bag of Words):** Predicts the center word from surrounding context words.
- Input: "The cat ___ on the mat" → Predict: "sat"

**Skip-gram:** Predicts context words from the center word.
- Input: "sat" → Predict: "The", "cat", "on", "the", "mat"

Skip-gram works better for rare words; CBOW is faster.

## Cosine Similarity

To measure how similar two word vectors are, use cosine similarity:

\`\`\`python
import math

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b)
\`\`\`

Values range from -1 (opposite) to 1 (identical). Similar words have scores near 1.

## GloVe

GloVe (Global Vectors, Stanford, 2014) learns embeddings from global word co-occurrence statistics. It builds a co-occurrence matrix (how often word pairs appear together) then factorizes it into embedding vectors.

GloVe often performs better than Word2Vec for analogy tasks because it captures global patterns.

## Pre-trained Embeddings

Training embeddings requires massive corpora. In practice, we use pre-trained models:
- **Word2Vec** (Google News, 3M words, 300d)
- **GloVe** (Wikipedia + Gigaword, 400K words, 50-300d)
- **FastText** (Facebook, handles subwords/OOV)

## Using Embeddings in Practice

1. Load pre-trained embeddings
2. Create an embedding matrix for your vocabulary
3. Use as the first layer of your neural network
4. Optionally fine-tune during training

## Limitations

- Static embeddings: each word has ONE vector regardless of context ("bank" as river bank vs money bank)
- This limitation is solved by contextual embeddings (BERT, GPT) — covered next lesson.`,
      commonMistakes: `## Common Mistakes

### 1. Using Raw Embeddings Without Fine-Tuning
Pre-trained embeddings are general. Fine-tune them on your specific task for better performance.

### 2. Wrong Dimensionality
Ensure your embedding dimension matches the pre-trained model you're using (50d GloVe ≠ 300d GloVe).

### 3. Not Handling Out-of-Vocabulary (OOV) Words
Words not in the pre-trained vocabulary get no embedding. Use a zero vector, random initialization, or FastText (which handles subwords).`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      { id: 'exercise-100', lessonId: lesson34.id, prompt: 'Implement `cosine_similarity(a, b)` using only math. Test with a=[1,2,3] and b=[4,5,6]. Print rounded to 4 decimals.', starterCode: 'import math\n\ndef cosine_similarity(a, b):\n    pass\n\nprint(round(cosine_similarity([1,2,3], [4,5,6]), 4))\n', expectedOutput: '0.9746', testCode: '', hints: JSON.stringify(['dot = sum(x*y for x,y in zip(a,b))', 'magnitude = math.sqrt(sum(x**2 for x in vec))', 'return dot / (mag_a * mag_b)']), order: 1 },
      { id: 'exercise-101', lessonId: lesson34.id, prompt: 'Build a vocabulary from sentences. Given sentences=["the cat sat", "the dog sat", "the cat ran"], extract all unique words, sort alphabetically, and assign indices. Print the vocab dict.', starterCode: 'sentences = ["the cat sat", "the dog sat", "the cat ran"]\n\n# Build vocabulary\n', expectedOutput: "{'cat': 0, 'dog': 1, 'ran': 2, 'sat': 3, 'the': 4}", testCode: '', hints: JSON.stringify(['Split each sentence and collect all words into a set', 'Sort the unique words', 'Create dict with enumerate: {word: i for i, word in enumerate(sorted_words)}']), order: 2 },
      { id: 'exercise-102', lessonId: lesson34.id, prompt: 'Implement one-hot encoding. Given vocab={"cat": 0, "dog": 1, "fish": 2} and words=["cat", "fish", "dog"], create one-hot vectors and print each.', starterCode: 'vocab = {"cat": 0, "dog": 1, "fish": 2}\nwords = ["cat", "fish", "dog"]\n\n# One-hot encode each word\n', expectedOutput: '[1, 0, 0]\n[0, 0, 1]\n[0, 1, 0]', testCode: '', hints: JSON.stringify(['For each word, create a list of zeros with length = len(vocab)', 'Set the index vocab[word] to 1', 'vec = [0]*len(vocab); vec[vocab[word]] = 1']), order: 3 },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      { id: 'quiz-100', lessonId: lesson34.id, question: 'What is the main advantage of word embeddings over one-hot encoding?', type: 'MCQ', options: JSON.stringify(['They use less storage', 'They capture semantic similarity between words', 'They are faster to compute', 'They work for any language']), correctAnswer: 'They capture semantic similarity between words', explanation: 'Embeddings place semantically similar words close together in vector space, capturing relationships that one-hot encoding misses entirely.', order: 1 },
      { id: 'quiz-101', lessonId: lesson34.id, question: 'Word2Vec embeddings capture the context of a word in a sentence, changing based on usage.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Word2Vec produces static embeddings — each word always has the same vector regardless of context. Contextual embeddings like BERT solve this.', order: 2 },
      { id: 'quiz-102', lessonId: lesson34.id, question: 'What does cosine similarity of 1.0 between two vectors mean?', type: 'MCQ', options: JSON.stringify(['The vectors are perpendicular', 'The vectors point in the same direction', 'The vectors are opposite', 'The vectors have equal magnitude']), correctAnswer: 'The vectors point in the same direction', explanation: 'Cosine similarity of 1 means the vectors are perfectly aligned (same direction), indicating maximum similarity.', order: 3 },
    ],
  });
  console.log('Seeded Lesson 34');

  // ── Lesson 35 ──────────────────────────────────────────────────────
  const lesson35 = await prisma.lesson.create({
    data: {
      id: 'lesson-035',
      moduleId: mod9.id,
      title: 'Transformers Explained — Attention, BERT, GPT',
      slug: 'transformers-explained',
      order: 3,
      content: `# Transformers Explained — Attention, BERT, GPT

The Transformer architecture (2017) revolutionized NLP and is the foundation of every modern large language model. Understanding transformers is essential for working with AI today.

## Why Not RNNs?

RNNs process text sequentially — word by word. This has two major problems:
1. **Slow:** Can't parallelize (each step depends on the previous)
2. **Forgets:** Long-range dependencies fade (even with LSTMs)

Transformers process all words simultaneously using **attention**, making them fast and effective at capturing relationships across any distance.

## The Attention Mechanism

Attention answers: "When processing this word, which other words should I focus on?"

For the sentence "The cat sat on the mat because it was tired":
- When processing "it", attention helps the model focus on "cat" (the referent).

### Self-Attention Step by Step

For each word, compute three vectors from the embedding:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

Attention score = softmax(Q · K^T / √d_k) · V

\`\`\`python
import math

def attention(query, keys, values):
    d_k = len(query)
    scores = [sum(q*k for q,k in zip(query, key)) / math.sqrt(d_k) for key in keys]
    # Softmax
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    weights = [e / total for e in exp_scores]
    # Weighted sum of values
    output = [sum(w * v[i] for w, v in zip(weights, values)) for i in range(len(values[0]))]
    return output, weights
\`\`\`

### Multi-Head Attention

Instead of one attention computation, transformers use multiple "heads" that each learn different relationships (syntax, semantics, coreference). Their outputs are concatenated and projected.

## The Transformer Architecture

**Encoder** (processes input):
- Multi-head self-attention → Add & Normalize → Feed-forward → Add & Normalize
- Stacked 6-12 times

**Decoder** (generates output):
- Masked self-attention → Cross-attention (attends to encoder) → Feed-forward
- Also stacked 6-12 times

### Positional Encoding

Since transformers process all words simultaneously, they need positional encoding to know word order. This uses sine/cosine functions at different frequencies:

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

## BERT — Bidirectional Encoder

BERT (Google, 2018) uses only the **encoder** part. It reads text in both directions simultaneously.

**Pre-training tasks:**
1. **Masked Language Modeling:** Randomly mask 15% of words, predict them. "The [MASK] sat on the mat" → predict "cat"
2. **Next Sentence Prediction:** Given two sentences, predict if the second follows the first.

**Fine-tuning:** Add a task-specific layer on top and train on your data (classification, NER, QA).

## GPT — Generative Pre-trained Transformer

GPT (OpenAI) uses only the **decoder** part. It reads text left-to-right and predicts the next word.

**Pre-training:** Given a sequence, predict the next token. Trained on massive internet text.

**Key insight:** Scaling up (more data, more parameters) leads to emergent abilities — few-shot learning, reasoning, code generation.

## BERT vs GPT

| | BERT | GPT |
|---|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Direction | Bidirectional | Left-to-right |
| Best for | Classification, NER, QA | Text generation, chat, code |
| Examples | BERT, RoBERTa, DeBERTa | GPT-4, Claude, LLaMA |`,
      commonMistakes: `## Common Mistakes

### 1. Thinking Transformers = GPT
Transformers are the architecture. BERT, GPT, T5 are all transformers but with different designs and use cases.

### 2. Ignoring Attention as "Soft Lookup"
Attention is essentially a differentiable lookup table — queries look up relevant keys to retrieve values. This mental model helps understand how it works.

### 3. Confusing Encoder-Only vs Decoder-Only
BERT (encoder) is bidirectional and great for understanding. GPT (decoder) is autoregressive and great for generation.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      { id: 'exercise-103', lessonId: lesson35.id, prompt: 'Implement a simplified attention score calculator. Given query=[1,0,1], keys=[[1,0,1],[0,1,0],[1,1,0]], compute dot product scores then apply softmax. Print weights rounded to 4 decimals.', starterCode: 'import math\n\nquery = [1, 0, 1]\nkeys = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]\n\n# Compute dot products and softmax\n', expectedOutput: '[0.7054, 0.0953, 0.2594]', testCode: '', hints: JSON.stringify(['Dot product: sum(q*k for q,k in zip(query, key))', 'Scores: [2, 0, 1]', 'Softmax: exp(s) / sum(exp(s)) for each score']), order: 1 },
      { id: 'exercise-104', lessonId: lesson35.id, prompt: 'Write a function that simulates masked language modeling. Given words=["the","cat","sat","on","the","mat"] and mask_index=2, replace that word with "[MASK]" and return the string. Print the result.', starterCode: 'def mask_word(words, mask_index):\n    pass\n\nresult = mask_word(["the", "cat", "sat", "on", "the", "mat"], 2)\nprint(result)\n', expectedOutput: 'the cat [MASK] on the mat', testCode: '', hints: JSON.stringify(['Copy the list to avoid modifying original', 'Set words_copy[mask_index] = "[MASK]"', 'Return " ".join(words_copy)']), order: 2 },
      { id: 'exercise-105', lessonId: lesson35.id, prompt: 'Implement softmax for a list of scores. Test with [2.0, 1.0, 0.1]. Print result rounded to 4 decimals.', starterCode: 'import math\n\ndef softmax(scores):\n    pass\n\nprint([round(x, 4) for x in softmax([2.0, 1.0, 0.1])])\n', expectedOutput: '[0.659, 0.2424, 0.0986]', testCode: '', hints: JSON.stringify(['exp_scores = [math.exp(s) for s in scores]', 'total = sum(exp_scores)', 'return [e/total for e in exp_scores]']), order: 3 },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      { id: 'quiz-103', lessonId: lesson35.id, question: 'What problem does the attention mechanism solve?', type: 'MCQ', options: JSON.stringify(['Reducing model size', 'Capturing relationships between any words regardless of distance', 'Making training faster', 'Reducing vocabulary size']), correctAnswer: 'Capturing relationships between any words regardless of distance', explanation: 'Attention allows each word to directly attend to every other word, solving the long-range dependency problem that RNNs struggle with.', order: 1 },
      { id: 'quiz-104', lessonId: lesson35.id, question: 'BERT processes text left-to-right like GPT.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'BERT is bidirectional — it reads text in both directions simultaneously. GPT is autoregressive and reads left-to-right.', order: 2 },
      { id: 'quiz-105', lessonId: lesson35.id, question: 'What are Q, K, V in the attention mechanism?', type: 'MCQ', options: JSON.stringify(['Quality, Knowledge, Validation', 'Query, Key, Value', 'Quantization, Kernel, Vector', 'Queue, Keep, Verify']), correctAnswer: 'Query, Key, Value', explanation: 'Query represents what a word is looking for, Key represents what a word contains, and Value represents the information to pass along when attention is high.', order: 3 },
    ],
  });
  console.log('Seeded Lesson 35');

  // ═══════════════════════════════════════════════════════════════════
  //  MODULE 10: Working with LLM APIs
  // ═══════════════════════════════════════════════════════════════════
  const mod10 = await prisma.module.create({
    data: {
      id: 'module-010',
      stageId: 'stage-006',
      title: 'Working with LLM APIs',
      slug: 'working-with-llm-apis',
      description: 'Learn to build real applications powered by large language models through their APIs.',
      order: 2,
    },
  });

  // ── Lesson 36 ──────────────────────────────────────────────────────
  const lesson36 = await prisma.lesson.create({
    data: {
      id: 'lesson-036',
      moduleId: mod10.id,
      title: 'LLM APIs — Chat Completions & System Prompts',
      slug: 'llm-apis-chat-completions',
      order: 1,
      content: `# LLM APIs — Chat Completions & System Prompts

Large Language Models like GPT-4 and Claude are accessed through APIs. Understanding how these APIs work is essential for building AI-powered applications.

## How LLM APIs Work

You send a **request** containing messages and receive a **response** with the model's completion. The core pattern:

\`\`\`python
import requests
import json

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
)

result = response.json()
answer = result["choices"][0]["message"]["content"]
\`\`\`

## Message Roles

Every conversation has three types of messages:

- **system:** Sets the AI's behavior, personality, and constraints. Sent first.
- **user:** The human's input.
- **assistant:** The AI's previous responses (for multi-turn conversations).

\`\`\`python
messages = [
    {"role": "system", "content": "You are a Python tutor. Give concise answers with code examples."},
    {"role": "user", "content": "How do I read a file?"},
    {"role": "assistant", "content": "Use open() with a context manager:\\n\\nwith open('file.txt') as f:\\n    content = f.read()"},
    {"role": "user", "content": "How do I read line by line?"}
]
\`\`\`

## Key Parameters

**temperature** (0-2): Controls randomness. 0 = deterministic, 1 = balanced, 2 = very creative.
- Use 0 for factual tasks (extraction, classification)
- Use 0.7-1.0 for creative tasks (writing, brainstorming)

**max_tokens:** Maximum length of the response. Set this to control cost and length.

**top_p:** Alternative to temperature. Nucleus sampling — only consider tokens whose cumulative probability exceeds top_p.

## Streaming Responses

For long responses, use streaming to show output as it's generated:

\`\`\`python
response = requests.post(url, json={..., "stream": True}, stream=True)
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode().removeprefix("data: "))
        chunk = data["choices"][0]["delta"].get("content", "")
        print(chunk, end="", flush=True)
\`\`\`

## API Key Management

**Never hardcode API keys!** Always use environment variables:

\`\`\`python
import os
api_key = os.getenv("OPENAI_API_KEY")
# Store in .env file, never commit to git
\`\`\`

## Cost Awareness

API calls cost money based on token usage:
- GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens
- GPT-3.5: ~$0.001/1K input tokens

Always estimate costs before running large batches. Set max_tokens to limit spending.`,
      commonMistakes: `## Common Mistakes

### 1. Exposing API Keys
Never put API keys in source code. Use environment variables and .env files.

### 2. Not Setting Temperature for Deterministic Tasks
For classification or extraction, set temperature=0 for consistent results.

### 3. Ignoring Rate Limits
APIs have rate limits. Implement retry logic with exponential backoff for production code.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      { id: 'exercise-106', lessonId: lesson36.id, prompt: 'Build a chat message formatter. Write `format_messages(system, user)` returning a list of message dicts. Test with system="You are a tutor" and user="Explain Python". Print the result.', starterCode: 'def format_messages(system, user):\n    pass\n\nresult = format_messages("You are a tutor", "Explain Python")\nprint(result)\n', expectedOutput: "[{'role': 'system', 'content': 'You are a tutor'}, {'role': 'user', 'content': 'Explain Python'}]", testCode: '', hints: JSON.stringify(['Return a list of two dicts', 'Each dict has "role" and "content" keys', 'First dict role is "system", second is "user"']), order: 1 },
      { id: 'exercise-107', lessonId: lesson36.id, prompt: 'Write `estimate_tokens(text)` that estimates token count as len(text.split()) * 1.3 (rounded to int). Test with "The quick brown fox jumps over the lazy dog". Print "Estimated tokens: {n}".', starterCode: 'def estimate_tokens(text):\n    pass\n\nprint(f"Estimated tokens: {estimate_tokens(\'The quick brown fox jumps over the lazy dog\')}")\n', expectedOutput: 'Estimated tokens: 12', testCode: '', hints: JSON.stringify(['Split text by spaces to count words', 'Multiply word count by 1.3', 'Round to int with round() or int()']), order: 2 },
      { id: 'exercise-108', lessonId: lesson36.id, prompt: 'Write `parse_llm_json(text)` that extracts JSON from text that may have extra content. Find the first { and last } and parse between them. Test with \'Here is the result: {"name": "Alice", "score": 95} Hope this helps!\'', starterCode: 'import json\n\ndef parse_llm_json(text):\n    pass\n\nresult = parse_llm_json(\'Here is the result: {"name": "Alice", "score": 95} Hope this helps!\')\nprint(result)\n', expectedOutput: "{'name': 'Alice', 'score': 95}", testCode: '', hints: JSON.stringify(['Find first { with text.index("{")', 'Find last } with text.rindex("}")', 'Parse with json.loads(text[start:end+1])']), order: 3 },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      { id: 'quiz-106', lessonId: lesson36.id, question: 'What does temperature=0 do in an LLM API call?', type: 'MCQ', options: JSON.stringify(['Makes the model refuse to answer', 'Produces deterministic, most likely output', 'Disables the model', 'Maximizes creativity']), correctAnswer: 'Produces deterministic, most likely output', explanation: 'Temperature=0 makes the model always choose the most probable next token, giving deterministic and focused outputs.', order: 1 },
      { id: 'quiz-107', lessonId: lesson36.id, question: 'The system message in a chat API sets the AI\'s behavior and personality.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'The system message is the first message sent and instructs the AI on how to behave, what role to play, and what constraints to follow.', order: 2 },
      { id: 'quiz-108', lessonId: lesson36.id, question: 'What is the safest way to store API keys?', type: 'MCQ', options: JSON.stringify(['In the source code', 'In environment variables', 'In comments', 'In the database']), correctAnswer: 'In environment variables', explanation: 'Environment variables keep secrets out of source code and version control. Use .env files locally and platform env vars in production.', order: 3 },
    ],
  });
  console.log('Seeded Lesson 36');

  // ── Lesson 37 ──────────────────────────────────────────────────────
  const lesson37 = await prisma.lesson.create({
    data: {
      id: 'lesson-037',
      moduleId: mod10.id,
      title: 'Prompt Engineering — Few-Shot, Chain-of-Thought',
      slug: 'prompt-engineering',
      order: 2,
      content: `# Prompt Engineering — Few-Shot, Chain-of-Thought

Prompt engineering is the art of crafting inputs that guide LLMs to produce the best possible output. It's the most important skill for working with AI today — the same model can give terrible or brilliant results depending on how you prompt it.

## Zero-Shot Prompting

Give the model a task with no examples:

\`\`\`
Classify the sentiment of this review as POSITIVE or NEGATIVE:
"The food was absolutely terrible and the service was slow."
\`\`\`

Works well for simple, well-defined tasks.

## Few-Shot Prompting

Provide examples to teach the model the pattern:

\`\`\`
Classify sentiment:

Review: "Amazing product, works perfectly!" → POSITIVE
Review: "Broke after one day, waste of money" → NEGATIVE
Review: "Decent quality for the price" → POSITIVE

Review: "The food was absolutely terrible" →
\`\`\`

Few-shot dramatically improves accuracy for complex or ambiguous tasks.

## Chain-of-Thought (CoT)

Ask the model to show its reasoning step by step:

\`\`\`
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 each. How many does he have?
A: Let me think step by step.
   Roger starts with 5 balls.
   He buys 2 cans × 3 balls = 6 new balls.
   Total = 5 + 6 = 11 balls.
   Answer: 11
\`\`\`

Simply adding "Let's think step by step" to a prompt significantly improves reasoning accuracy.

## Structured Output

Ask for specific formats to make parsing easier:

\`\`\`
Extract the following from the email and return as JSON:
- sender_name
- subject
- urgency (low/medium/high)

Return ONLY valid JSON, no other text.
\`\`\`

## System Prompt Design

A good system prompt includes:
1. **Role:** "You are an expert Python developer"
2. **Task:** "You help users debug their code"
3. **Constraints:** "Always include code examples. Never write unsafe code."
4. **Format:** "Respond in markdown with code blocks."

## Prompt Templates

Build reusable prompts with placeholders:

\`\`\`python
def build_prompt(template, **kwargs):
    return template.format(**kwargs)

template = "Translate '{text}' from {src} to {dst}."
prompt = build_prompt(template, text="Hello", src="English", dst="French")
\`\`\`

## Evaluation

How to tell if your prompts are good:
- Test with diverse inputs (edge cases, adversarial inputs)
- Measure accuracy on labeled test sets
- A/B test different prompt versions
- Check for consistency (same input → similar output)`,
      commonMistakes: `## Common Mistakes

### 1. Vague Prompts
"Tell me about Python" → too broad. "List 5 key features of Python with one-sentence explanations" → specific and actionable.

### 2. Not Providing Examples for Complex Tasks
If the task requires a specific format or nuanced judgment, always provide 2-3 examples.

### 3. Not Validating Structured Output
When asking for JSON, always validate the response parses correctly. LLMs sometimes produce invalid JSON.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      { id: 'exercise-109', lessonId: lesson37.id, prompt: 'Create a prompt template function. `build_prompt(template, **kwargs)` replaces placeholders. Test: build_prompt("Translate {text} from {src} to {dst}", text="Hello", src="English", dst="French").', starterCode: 'def build_prompt(template, **kwargs):\n    pass\n\nresult = build_prompt("Translate {text} from {src} to {dst}", text="Hello", src="English", dst="French")\nprint(result)\n', expectedOutput: 'Translate Hello from English to French', testCode: '', hints: JSON.stringify(['Use str.format(**kwargs)', 'return template.format(**kwargs)']), order: 1 },
      { id: 'exercise-110', lessonId: lesson37.id, prompt: 'Build a few-shot prompt. Write `build_few_shot(examples, query)` where examples is a list of {"input":..., "output":...} dicts. Format each as "Input: {input}\\nOutput: {output}\\n" then add "Input: {query}\\nOutput:". Test with 2 examples.', starterCode: 'def build_few_shot(examples, query):\n    pass\n\nexamples = [\n    {"input": "happy", "output": "POSITIVE"},\n    {"input": "terrible", "output": "NEGATIVE"}\n]\nprint(build_few_shot(examples, "wonderful"))\n', expectedOutput: 'Input: happy\nOutput: POSITIVE\n\nInput: terrible\nOutput: NEGATIVE\n\nInput: wonderful\nOutput:', testCode: '', hints: JSON.stringify(['Loop through examples, format each pair', 'Join with "\\n\\n"', 'Add the query at the end without an output']), order: 2 },
      { id: 'exercise-111', lessonId: lesson37.id, prompt: 'Write a chain-of-thought solver for simple addition. Given "What is 15 + 27?", extract numbers, detect "+" operation, compute, and print step-by-step.', starterCode: 'import re\n\ndef solve_with_cot(problem):\n    pass\n\nsolve_with_cot("What is 15 + 27?")\n', expectedOutput: 'Step 1: Identify numbers: 15, 27\nStep 2: Operation: addition\nStep 3: 15 + 27 = 42\nAnswer: 42', testCode: '', hints: JSON.stringify(['Use re.findall(r"\\d+", problem) to extract numbers', 'Check if "+" is in the problem string', 'Compute the result and print each step']), order: 3 },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      { id: 'quiz-109', lessonId: lesson37.id, question: 'What is few-shot prompting?', type: 'MCQ', options: JSON.stringify(['Training the model on a few examples', 'Providing examples in the prompt to demonstrate the task', 'Using a small model', 'Sending multiple API requests']), correctAnswer: 'Providing examples in the prompt to demonstrate the task', explanation: 'Few-shot prompting includes input-output examples in the prompt itself, teaching the model the expected pattern without any actual training.', order: 1 },
      { id: 'quiz-110', lessonId: lesson37.id, question: 'Adding "Let\'s think step by step" to a prompt can improve reasoning accuracy.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'Chain-of-thought prompting has been shown to significantly improve performance on reasoning tasks by encouraging the model to show intermediate steps.', order: 2 },
      { id: 'quiz-111', lessonId: lesson37.id, question: 'What is the purpose of a system prompt?', type: 'MCQ', options: JSON.stringify(['To train the model', 'To set the AI\'s behavior, role, and constraints', 'To authenticate the API request', 'To format the output as HTML']), correctAnswer: "To set the AI's behavior, role, and constraints", explanation: 'The system prompt instructs the model on how to behave, what role to play, and what rules to follow throughout the conversation.', order: 3 },
    ],
  });
  console.log('Seeded Lesson 37');

  // ── Lesson 38 ──────────────────────────────────────────────────────
  const lesson38 = await prisma.lesson.create({
    data: {
      id: 'lesson-038',
      moduleId: mod10.id,
      title: 'RAG — Retrieval-Augmented Generation',
      slug: 'rag-retrieval-augmented-generation',
      order: 3,
      content: `# RAG — Retrieval-Augmented Generation

LLMs have two major limitations: they hallucinate (make up facts) and their knowledge has a cutoff date. **RAG** solves both by giving the model access to external documents at query time.

## What is RAG?

RAG = Retrieve relevant documents → Augment the prompt with them → Generate an answer.

Instead of relying solely on the LLM's training data, you search your own documents and inject the relevant pieces into the prompt:

\`\`\`
Context:
Python was created by Guido van Rossum and released in 1991.
Python 3.0 was released in December 2008.

Question: When was Python 3 released?
Answer based on the context above.
\`\`\`

## The RAG Pipeline

### Step 1: Document Chunking

Split large documents into smaller pieces (chunks) that fit in the LLM context:

\`\`\`python
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
\`\`\`

**Chunk size matters:** Too small = missing context. Too large = noise drowns out the answer. Typical: 200-500 words with 10-20% overlap.

### Step 2: Create Embeddings

Convert each chunk into a vector using an embedding model:

\`\`\`python
# Conceptual — using OpenAI embeddings
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=["Python was created by Guido van Rossum"]
)
vector = response.data[0].embedding  # 1536-dimensional vector
\`\`\`

### Step 3: Store in Vector Database

Store embeddings in a vector database for fast similarity search:
- **Pinecone** — managed, scalable
- **ChromaDB** — open source, local
- **Weaviate** — open source, feature-rich
- **FAISS** — Facebook's fast similarity search

### Step 4: Query & Retrieve

When a user asks a question:
1. Embed the question using the same model
2. Search the vector DB for the most similar chunks
3. Return top-k results

### Step 5: Augment & Generate

Inject retrieved chunks into the prompt and call the LLM:

\`\`\`python
def build_rag_prompt(query, retrieved_chunks):
    context = "\\n".join(retrieved_chunks)
    return f"""Context:
{context}

Question: {query}
Answer based only on the context above. If the answer is not in the context, say "I don't know."
"""
\`\`\`

## Semantic Search vs Keyword Search

**Keyword search:** Matches exact words ("python creator" won't find "Guido van Rossum")

**Semantic search:** Matches meaning via embeddings ("python creator" WILL find "Guido van Rossum" because the vectors are similar)

## Evaluation

RAG quality depends on:
1. **Retrieval quality:** Are the right chunks being found?
2. **Generation quality:** Does the LLM answer correctly from the chunks?
3. **Faithfulness:** Does the answer stick to the provided context?

Metrics: precision@k, recall, answer relevance, faithfulness score.`,
      commonMistakes: `## Common Mistakes

### 1. Chunks Too Large or Small
Too large: irrelevant content dilutes the answer. Too small: important context is split across chunks.

### 2. No Chunk Overlap
Without overlap, sentences at chunk boundaries get split. Use 10-20% overlap.

### 3. Not Telling the LLM to Use Only the Context
Without explicit instructions, the LLM will use its training data and may hallucinate. Always add "Answer based only on the context provided."

### 4. Poor Retrieval Quality
If your embeddings don't capture domain-specific meaning, retrieval will fail. Consider fine-tuning the embedding model.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      { id: 'exercise-112', lessonId: lesson38.id, prompt: 'Implement text chunking. Write `chunk_text(text, chunk_size, overlap)` that splits text into chunks of chunk_size words with overlap. Test with "one two three four five six seven eight nine ten eleven twelve" with chunk_size=5, overlap=2.', starterCode: 'def chunk_text(text, chunk_size, overlap):\n    pass\n\ntext = "one two three four five six seven eight nine ten eleven twelve"\nchunks = chunk_text(text, 5, 2)\nfor c in chunks:\n    print(c)\n', expectedOutput: 'one two three four five\nfour five six seven eight\nseven eight nine ten eleven\nten eleven twelve', testCode: '', hints: JSON.stringify(['Split text into words list', 'Step through with range(0, len(words), chunk_size - overlap)', 'Each chunk = words[i:i+chunk_size] joined by spaces']), order: 1 },
      { id: 'exercise-113', lessonId: lesson38.id, prompt: 'Implement simple keyword search scoring. Given docs and a query word, score each doc by (count of query term / total words). Print docs sorted by score descending.', starterCode: 'docs = [\n    "the cat sat on the mat",\n    "dogs are great pets",\n    "cats and dogs play together"\n]\nquery = "cat"\n\n# Score and rank documents\n', expectedOutput: 'Score 0.1667: the cat sat on the mat\nScore 0.0000: dogs are great pets\nScore 0.0000: cats and dogs play together', testCode: '', hints: JSON.stringify(['For each doc, count occurrences of the exact query word', 'Score = count / len(doc.split())', 'Sort by score descending and print']), order: 2 },
      { id: 'exercise-114', lessonId: lesson38.id, prompt: 'Build a RAG prompt. Given chunks=["Python was created by Guido van Rossum.", "Python 3 was released in 2008."] and query="Who created Python?", build the augmented prompt.', starterCode: 'def build_rag_prompt(chunks, query):\n    pass\n\nchunks = ["Python was created by Guido van Rossum.", "Python 3 was released in 2008."]\nprint(build_rag_prompt(chunks, "Who created Python?"))\n', expectedOutput: 'Context:\nPython was created by Guido van Rossum.\nPython 3 was released in 2008.\n\nQuestion: Who created Python?\nAnswer:', testCode: '', hints: JSON.stringify(['Join chunks with newline', 'Format as "Context:\\n{chunks}\\n\\nQuestion: {query}\\nAnswer:"']), order: 3 },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      { id: 'quiz-112', lessonId: lesson38.id, question: 'What does RAG stand for?', type: 'MCQ', options: JSON.stringify(['Random Access Generation', 'Retrieval-Augmented Generation', 'Recursive Attention Graph', 'Real-time AI Gateway']), correctAnswer: 'Retrieval-Augmented Generation', explanation: 'RAG retrieves relevant documents, augments the prompt with them, and then generates an answer — combining search with generation.', order: 1 },
      { id: 'quiz-113', lessonId: lesson38.id, question: 'Vector databases store text as raw strings for fast retrieval.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Vector databases store text as embedding vectors (numerical arrays) and use similarity search algorithms to find the closest matches.', order: 2 },
      { id: 'quiz-114', lessonId: lesson38.id, question: 'Why is chunk overlap important in RAG?', type: 'MCQ', options: JSON.stringify(['It makes chunks bigger', 'It prevents sentences from being split at boundaries', 'It reduces storage cost', 'It speeds up embedding']), correctAnswer: 'It prevents sentences from being split at boundaries', explanation: 'Without overlap, a sentence at the boundary between two chunks gets split, losing its meaning. Overlap ensures continuity.', order: 3 },
    ],
  });
  console.log('Seeded Lesson 38');

  // ═══════════════════════════════════════════════════════════════════
  //  PROJECTS
  // ═══════════════════════════════════════════════════════════════════
  await prisma.project.create({
    data: {
      id: 'project-011', title: 'Document Q&A Bot with RAG', slug: 'document-qa-rag', stage: 'NLP', order: 11,
      brief: 'Build a document Q&A system that chunks text documents, creates embeddings, and uses RAG to answer questions accurately.',
      requirements: JSON.stringify(['Load and chunk text documents into appropriate sizes', 'Generate embeddings for each chunk using an embedding API', 'Store embeddings in a vector database (ChromaDB or similar)', 'Implement semantic search to retrieve relevant chunks for a query', 'Build the RAG pipeline: retrieve → augment → generate using an LLM API']),
      stretchGoals: JSON.stringify(['Support PDF and markdown document loading', 'Add source citations in answers', 'Implement a web interface with Streamlit']),
      steps: JSON.stringify([{ title: 'Document loading & chunking', description: 'Build a pipeline that reads text files and splits them into overlapping chunks of appropriate size.' }, { title: 'Embedding generation', description: 'Use an embedding model to convert chunks into vectors and store them.' }, { title: 'Vector store setup', description: 'Set up ChromaDB or FAISS to store and query embeddings.' }, { title: 'RAG pipeline', description: 'Combine retrieval and generation: search for relevant chunks, build augmented prompt, call LLM.' }, { title: 'Testing & evaluation', description: 'Test with various questions, measure retrieval quality and answer accuracy.' }]),
      rubric: JSON.stringify([{ criterion: 'Chunking Strategy', description: 'Documents are split with appropriate size and overlap, preserving context.' }, { criterion: 'Retrieval Quality', description: 'Relevant chunks are found for diverse queries.' }, { criterion: 'Answer Quality', description: 'Generated answers are accurate, faithful to sources, and well-formatted.' }, { criterion: 'Code Quality', description: 'Clean, modular code with proper error handling.' }]),
      solutionUrl: null,
    },
  });

  await prisma.project.create({
    data: {
      id: 'project-012', title: 'Automated Email Classifier', slug: 'email-classifier', stage: 'NLP', order: 12,
      brief: 'Build a system that classifies emails into categories (support, billing, feature request, spam) using LLM APIs with prompt engineering.',
      requirements: JSON.stringify(['Design effective prompts for email classification', 'Implement few-shot prompting with example emails', 'Handle edge cases and ambiguous emails gracefully', 'Return structured JSON output with category and confidence', 'Track accuracy metrics on a test set']),
      stretchGoals: JSON.stringify(['Add automatic response drafting for each category', 'Implement priority scoring', 'Build a batch processing pipeline']),
      steps: JSON.stringify([{ title: 'Design classification prompts', description: 'Create system and few-shot prompts that accurately classify emails.' }, { title: 'Build the classifier', description: 'Implement the API call pipeline with structured JSON output.' }, { title: 'Handle edge cases', description: 'Test with ambiguous emails and add fallback handling.' }, { title: 'Evaluation', description: 'Run on a test set and calculate accuracy per category.' }, { title: 'Optimization', description: 'Iterate on prompts to improve accuracy on failing cases.' }]),
      rubric: JSON.stringify([{ criterion: 'Prompt Design', description: 'Prompts are clear, well-structured, and include effective examples.' }, { criterion: 'Classification Accuracy', description: 'Correctly classifies >80% of test emails.' }, { criterion: 'Error Handling', description: 'Gracefully handles API errors, invalid responses, and edge cases.' }, { criterion: 'Code Architecture', description: 'Modular design with reusable prompt templates and evaluation framework.' }]),
      solutionUrl: null,
    },
  });
  console.log('Seeded Stage 6 Projects');

  console.log('');
  console.log('🎉 Stage 6 (NLP & LLMs) seeding complete!');
}

main()
  .then(async () => { await prisma.$disconnect(); })
  .catch(async (e) => { console.error(e); await prisma.$disconnect(); process.exit(1); });
