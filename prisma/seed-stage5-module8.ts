export {};

async function seedModule8(prisma: any) {
  console.log('Seeding Module 8: Deep Learning Foundations...');

  // ─── Stage 5 ──────────────────────────────────────────────────────
  await prisma.stage.upsert({
    where: { id: 'stage-005' },
    update: {},
    create: {
      id: 'stage-005',
      title: 'Deep Learning & Neural Networks',
      slug: 'deep-learning',
      description:
        'Understand how neural networks work from scratch, then build them with modern frameworks like TensorFlow and Keras.',
      order: 5,
    },
  });
  console.log('Upserted Stage 5');

  // ─── Skill Tags ──────────────────────────────────────────────────
  const newSkills = [
    { id: 'skill-033', name: 'Neural Networks', slug: 'neural-networks' },
    { id: 'skill-034', name: 'Deep Learning', slug: 'deep-learning' },
    { id: 'skill-035', name: 'TensorFlow', slug: 'tensorflow' },
    { id: 'skill-036', name: 'CNNs', slug: 'cnns' },
    { id: 'skill-037', name: 'RNNs', slug: 'rnns' },
    { id: 'skill-038', name: 'Computer Vision', slug: 'computer-vision' },
    { id: 'skill-039', name: 'NLP Basics', slug: 'nlp-basics' },
  ];
  for (const s of newSkills) {
    await prisma.skillTag.upsert({
      where: { id: s.id },
      update: {},
      create: s,
    });
  }
  console.log('Created skill tags for Module 8');

  // ─── Module ───────────────────────────────────────────────────────
  const mod = await prisma.module.create({
    data: {
      id: 'module-008',
      stageId: 'stage-005',
      title: 'Deep Learning Foundations',
      slug: 'deep-learning-foundations',
      description:
        'Learn neural network concepts from perceptrons to CNNs and RNNs, and build models with TensorFlow/Keras.',
      order: 1,
    },
  });

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 28 — Neural Network Intuition
  // ═══════════════════════════════════════════════════════════════════
  const lesson28 = await prisma.lesson.create({
    data: {
      id: 'lesson-028',
      moduleId: mod.id,
      title: 'Neural Network Intuition — Perceptrons, Activation & Backprop',
      slug: 'neural-network-intuition',
      order: 1,
      content: `# Neural Network Intuition — Perceptrons, Activation & Backprop

Neural networks are the foundation of modern artificial intelligence. From image recognition to language translation, almost every breakthrough in AI over the past decade has been powered by neural networks. In this lesson we will build an intuition for how they work — starting from a single artificial neuron and working our way up to understanding how networks learn through backpropagation.

## Biological Inspiration

The idea behind artificial neural networks comes from the human brain. Our brains contain roughly 86 billion neurons, each connected to thousands of others through synapses. When a neuron receives enough stimulation from its incoming connections, it "fires" and sends a signal to the next neurons in the chain. Artificial neural networks mimic this pattern: each artificial neuron receives inputs, applies weights to them, sums everything up, and then passes the result through an activation function to decide what signal to send forward.

It is important to understand that artificial neurons are a very simplified mathematical abstraction of biological neurons — they are not a faithful simulation of biology. However, this abstraction turns out to be extremely powerful for learning patterns in data.

## The Perceptron Model

The simplest neural network is a single neuron called a **perceptron**, invented by Frank Rosenblatt in 1958. A perceptron works as follows:

1. **Inputs** (x1, x2, ..., xn): The features of your data.
2. **Weights** (w1, w2, ..., wn): Each input has an associated weight that determines its importance.
3. **Bias** (b): An extra parameter that shifts the decision boundary.
4. **Weighted sum**: z = w1*x1 + w2*x2 + ... + wn*xn + b
5. **Activation function**: The weighted sum is passed through an activation function to produce the output.

In a diagram (text-based):

\`\`\`
  x1 ---(w1)---\\
  x2 ---(w2)----[Sum + Bias] ---> [Activation] ---> output
  x3 ---(w3)---/
\`\`\`

Here is a simple perceptron implemented from scratch with NumPy:

\`\`\`python
import numpy as np

def perceptron(inputs, weights, bias):
    """A single perceptron with step activation."""
    weighted_sum = np.dot(inputs, weights) + bias
    # Step activation: output 1 if weighted_sum >= 0, else 0
    return 1 if weighted_sum >= 0 else 0

# Example: AND gate
weights = np.array([0.5, 0.5])
bias = -0.7

print(perceptron([0, 0], weights, bias))  # 0
print(perceptron([0, 1], weights, bias))  # 0
print(perceptron([1, 0], weights, bias))  # 0
print(perceptron([1, 1], weights, bias))  # 1
\`\`\`

This perceptron has learned the AND logic gate — it only outputs 1 when both inputs are 1.

## Activation Functions

The activation function introduces **non-linearity** into the network. Without activation functions, stacking multiple layers of neurons would just produce a linear transformation — no better than a single layer. Here are the most important activation functions:

### Sigmoid

Squashes any input into the range (0, 1). Historically popular but suffers from the **vanishing gradient problem** for very large or very small inputs.

\`\`\`python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

values = np.array([-2, -1, 0, 1, 2])
print(sigmoid(values))
# [0.11920292 0.26894142 0.5        0.73105858 0.88079708]
\`\`\`

### ReLU (Rectified Linear Unit)

Returns the input if it is positive, otherwise returns zero. It is the most widely used activation function in modern deep learning because it is computationally efficient and helps avoid the vanishing gradient problem.

\`\`\`python
def relu(x):
    return np.maximum(0, x)

print(relu(np.array([-3, -1, 0, 2, 5])))
# [0 0 0 2 5]
\`\`\`

### Tanh (Hyperbolic Tangent)

Squashes inputs into the range (-1, 1). It is zero-centered, which can make optimization easier compared to sigmoid.

\`\`\`python
import numpy as np

values = np.array([-2, -1, 0, 1, 2])
print(np.tanh(values))
# [-0.96402758 -0.76159416  0.          0.76159416  0.96402758]
\`\`\`

### Softmax

Converts a vector of raw scores into a probability distribution that sums to 1. Typically used in the output layer for multi-class classification.

\`\`\`python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

scores = np.array([2.0, 1.0, 0.1])
print(softmax(scores))
# [0.65900114 0.24243297 0.09856589]
\`\`\`

## The Forward Pass

When data flows through a neural network from input to output, this is called the **forward pass**. For a network with multiple layers, the output of one layer becomes the input to the next:

\`\`\`
Input Layer ---> Hidden Layer 1 ---> Hidden Layer 2 ---> Output Layer
   (x)            (h1 = f(W1*x + b1))  (h2 = f(W2*h1 + b2))  (y = f(W3*h2 + b3))
\`\`\`

Each layer performs: **output = activation(weights * input + bias)**

\`\`\`python
import numpy as np

# Simple 2-layer network forward pass
def forward_pass(x, W1, b1, W2, b2):
    # Hidden layer with ReLU
    z1 = np.dot(W1, x) + b1
    h1 = np.maximum(0, z1)  # ReLU

    # Output layer with sigmoid
    z2 = np.dot(W2, h1) + b2
    output = 1 / (1 + np.exp(-z2))  # Sigmoid
    return output

# Example with random weights
np.random.seed(42)
x = np.array([1.0, 0.5])
W1 = np.random.randn(3, 2)  # 3 hidden neurons, 2 inputs
b1 = np.zeros(3)
W2 = np.random.randn(1, 3)  # 1 output, 3 hidden
b2 = np.zeros(1)

result = forward_pass(x, W1, b1, W2, b2)
print(f"Output: {result}")
\`\`\`

## Loss Functions

After the forward pass we need a way to measure how wrong our predictions are. This measurement is called the **loss** (or cost). Common loss functions include:

- **Mean Squared Error (MSE)**: For regression tasks. Computes the average of squared differences between predicted and actual values.
- **Binary Cross-Entropy**: For binary classification (0 or 1). Measures how well predicted probabilities match actual labels.
- **Categorical Cross-Entropy**: For multi-class classification. Extends binary cross-entropy to multiple classes.

\`\`\`python
# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy
def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print(mse(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8])))  # 0.02
print(binary_crossentropy(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8])))
\`\`\`

## Gradient Descent

Now we know how wrong our network is (the loss), but how do we improve it? The answer is **gradient descent**. The idea is simple:

1. Compute the gradient (derivative) of the loss with respect to each weight.
2. Move each weight a small step in the direction that **reduces** the loss.
3. Repeat.

The size of each step is controlled by the **learning rate** (often denoted as alpha or lr). If the learning rate is too large, you might overshoot the minimum. If it is too small, training takes forever.

\`\`\`python
# Simple gradient descent example: minimize f(w) = (w - 3)^2
w = 0.0
lr = 0.1

for i in range(20):
    gradient = 2 * (w - 3)   # derivative of (w-3)^2
    w = w - lr * gradient
    if i < 5:
        print(f"Step {i+1}: w = {w:.4f}")

print(f"Final w: {w:.4f}")  # Should be close to 3.0
\`\`\`

## Backpropagation Intuition

**Backpropagation** is the algorithm that computes gradients efficiently for every weight in the network. It uses the **chain rule** from calculus to propagate the error backward from the output layer to the input layer.

Think of it this way: the loss depends on the output, the output depends on the last layer's weights, the last layer depends on the second-to-last layer, and so on. By chaining these dependencies together (the chain rule), we can compute how much each individual weight contributed to the final error.

The process works in two phases:

1. **Forward pass**: Compute all intermediate values and the final output.
2. **Backward pass**: Starting from the loss, compute the gradient of the loss with respect to each weight by applying the chain rule layer by layer, moving from the output back to the input.

\`\`\`
Loss -----> dL/dW3 (output layer gradients)
  |
  +-------> dL/dW2 (hidden layer 2 gradients, via chain rule)
  |
  +-------> dL/dW1 (hidden layer 1 gradients, via chain rule)
\`\`\`

Modern frameworks like TensorFlow and PyTorch handle backpropagation automatically using a technique called **automatic differentiation**, so you rarely need to compute gradients by hand. However, understanding the concept is crucial for debugging training issues and designing effective architectures.

## Putting It All Together

A complete neural network training loop looks like this conceptually:

1. **Initialize** weights randomly.
2. **Forward pass**: Feed input data through the network to get predictions.
3. **Compute loss**: Measure how far predictions are from the true labels.
4. **Backward pass (backpropagation)**: Compute gradients of the loss with respect to all weights.
5. **Update weights**: Adjust weights using gradient descent.
6. **Repeat** steps 2-5 for many iterations (epochs) until the loss is sufficiently low.

This loop is the heart of all neural network training, whether you are building a simple classifier or training a billion-parameter language model.`,
      commonMistakes: `## Common Mistakes

### 1. Confusing Activation Functions
\`\`\`python
# Sigmoid outputs (0, 1) — good for probabilities
# ReLU outputs [0, inf) — good for hidden layers
# Softmax outputs a probability distribution that sums to 1

# WRONG: using sigmoid for multi-class output
# RIGHT: use softmax for multi-class, sigmoid for binary
\`\`\`

### 2. Not Understanding Gradient Descent Conceptually
Many beginners think neural networks "just work magically." In reality, they learn by iteratively adjusting weights to minimize a loss function. If you do not understand this optimization loop, you will struggle to debug training problems like vanishing gradients, exploding gradients, or non-converging models.

### 3. Thinking Neural Networks Are Magic
Neural networks are powerful function approximators, but they are not magic. They require:
- Enough quality data to learn from
- Appropriate architecture for the problem
- Careful tuning of hyperparameters (learning rate, number of layers, etc.)
- Proper preprocessing of input data

A neural network cannot learn patterns that are not present in the data, and it can easily memorize noise if not properly regularized.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-082',
        lessonId: lesson28.id,
        prompt:
          'Implement a sigmoid function using numpy and apply it to the values [-2, -1, 0, 1, 2]. Print the result as a list with each value rounded to 4 decimal places.',
        starterCode:
          'import numpy as np\n\ndef sigmoid(x):\n    # Implement the sigmoid function: 1 / (1 + e^(-x))\n    pass\n\nvalues = [-2, -1, 0, 1, 2]\nresult = [round(sigmoid(v), 4) for v in values]\nprint(result)\n',
        expectedOutput: '[0.1192, 0.2689, 0.5, 0.7311, 0.8808]',
        testCode: '',
        hints: JSON.stringify([
          'The sigmoid formula is: 1 / (1 + np.exp(-x))',
          'Use np.exp() for the exponential function',
          'def sigmoid(x): return 1 / (1 + np.exp(-x))',
        ]),
        order: 1,
      },
      {
        id: 'exercise-083',
        lessonId: lesson28.id,
        prompt:
          'Implement a ReLU (Rectified Linear Unit) function and apply it to the values [-3, -1, 0, 2, 5]. Print the result as a list.',
        starterCode:
          'import numpy as np\n\ndef relu(x):\n    # ReLU returns x if x > 0, else 0\n    pass\n\nvalues = [-3, -1, 0, 2, 5]\nresult = [relu(v) for v in values]\nprint(result)\n',
        expectedOutput: '[0, 0, 0, 2, 5]',
        testCode: '',
        hints: JSON.stringify([
          'ReLU returns max(0, x)',
          'You can use: return max(0, x)',
          'Or with numpy: return np.maximum(0, x)',
        ]),
        order: 2,
      },
      {
        id: 'exercise-084',
        lessonId: lesson28.id,
        prompt:
          'Implement a single perceptron with weights=[0.5, -0.3], bias=0.1, using step activation (output 1 if weighted_sum + bias >= 0, else 0). Test with inputs [1, 1] and [0, 1]. Print both outputs on separate lines.',
        starterCode:
          'import numpy as np\n\ndef perceptron(inputs, weights, bias):\n    # Compute weighted sum, add bias, apply step activation\n    pass\n\nweights = [0.5, -0.3]\nbias = 0.1\n\nprint(perceptron([1, 1], weights, bias))\nprint(perceptron([0, 1], weights, bias))\n',
        expectedOutput: '1\n0',
        testCode: '',
        hints: JSON.stringify([
          'Compute weighted_sum = sum of (input_i * weight_i) for all i',
          'Add bias to the weighted sum, then check if >= 0',
          'Use np.dot(inputs, weights) + bias to get the weighted sum, return 1 if >= 0 else 0',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-082',
        lessonId: lesson28.id,
        question: 'What does the ReLU activation function return for negative inputs?',
        type: 'MCQ',
        options: JSON.stringify([
          'The input value',
          'Zero',
          'The absolute value',
          'Negative one',
        ]),
        correctAnswer: 'Zero',
        explanation:
          'ReLU (Rectified Linear Unit) returns max(0, x). For any negative input, the output is zero. For positive inputs, the output equals the input.',
        order: 1,
      },
      {
        id: 'quiz-083',
        lessonId: lesson28.id,
        question: 'Backpropagation computes gradients using the chain rule.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation:
          'Backpropagation uses the chain rule from calculus to compute the gradient of the loss function with respect to each weight by propagating the error backward through the network layers.',
        order: 2,
      },
      {
        id: 'quiz-084',
        lessonId: lesson28.id,
        question: 'What is the purpose of a bias term in a neuron?',
        type: 'MCQ',
        options: JSON.stringify([
          'To normalize the output',
          'To shift the activation function',
          'To reduce overfitting',
          'To speed up training',
        ]),
        correctAnswer: 'To shift the activation function',
        explanation:
          'The bias allows the activation function to be shifted left or right, which gives the neuron more flexibility to fit the data. Without bias, the activation function is anchored at the origin.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 28: Neural Network Intuition');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 29 — TensorFlow / Keras Basics
  // ═══════════════════════════════════════════════════════════════════
  const lesson29 = await prisma.lesson.create({
    data: {
      id: 'lesson-029',
      moduleId: mod.id,
      title: 'TensorFlow / Keras Basics — Sequential API',
      slug: 'tensorflow-keras-basics',
      order: 2,
      content: `# TensorFlow / Keras Basics — Sequential API

TensorFlow is Google's open-source deep learning framework, and Keras is its high-level API that makes building neural networks as simple as stacking layers together. In this lesson you will learn how to build, compile, train, and evaluate neural networks using the Keras Sequential API.

## Installing TensorFlow

TensorFlow can be installed with pip. It includes Keras automatically:

\`\`\`python
# Install TensorFlow (includes Keras)
# pip install tensorflow

import tensorflow as tf
print(tf.__version__)  # e.g., 2.15.0
\`\`\`

TensorFlow works on CPU by default and can leverage GPUs for faster training if you have the appropriate CUDA drivers installed.

## The Sequential API

The \`tf.keras.Sequential\` model is the simplest way to build a neural network. You stack layers one on top of another, and data flows through them in order — from the first layer to the last.

\`\`\`python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
\`\`\`

You can also add layers one at a time using \`model.add()\`:

\`\`\`python
model = keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
\`\`\`

## Understanding Dense Layers

A \`Dense\` layer is a **fully connected** layer where every neuron in the layer is connected to every neuron in the previous layer. The key parameters are:

- **units**: The number of neurons in the layer (e.g., 128).
- **activation**: The activation function to apply (e.g., 'relu', 'sigmoid', 'softmax').
- **input_shape**: Only needed for the first layer — tells the model the shape of the input data.

Each Dense layer with \`n\` input features and \`m\` output neurons has \`n * m + m\` trainable parameters (weights plus biases).

\`\`\`python
# Layer parameter count example:
# Dense(128, input_shape=(784,)) → 784 * 128 + 128 = 100,480 parameters
# Dense(64)                      → 128 * 64 + 64  = 8,256 parameters
# Dense(10)                      → 64 * 10 + 10   = 650 parameters

model.summary()  # Prints architecture and parameter counts
\`\`\`

## Compiling the Model

Before training, you must **compile** the model by specifying three things:

1. **Optimizer**: The algorithm that updates weights (e.g., 'adam', 'sgd').
2. **Loss function**: How to measure prediction error.
3. **Metrics**: What to track during training (e.g., 'accuracy').

\`\`\`python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
\`\`\`

### Choosing the Right Loss Function

| Task | Output Activation | Loss Function |
|------|-------------------|---------------|
| Binary classification (0/1) | sigmoid (1 neuron) | binary_crossentropy |
| Multi-class (one label) | softmax (N neurons) | categorical_crossentropy |
| Multi-class (integer labels) | softmax (N neurons) | sparse_categorical_crossentropy |
| Regression (continuous value) | linear / none | mean_squared_error |

This is one of the most important decisions in building a neural network. Using the wrong loss function for your task will result in the model not learning properly or producing meaningless outputs.

## Training the Model with model.fit()

The \`model.fit()\` method trains the model on your data:

\`\`\`python
history = model.fit(
    X_train,           # Training features
    y_train,           # Training labels
    epochs=20,         # Number of full passes through the data
    batch_size=32,     # Number of samples per gradient update
    validation_split=0.2,  # Use 20% of training data for validation
)
\`\`\`

The \`history\` object contains the training and validation metrics for each epoch, which is extremely useful for diagnosing how well training is progressing:

\`\`\`python
print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
\`\`\`

## Making Predictions

After training, use \`model.predict()\` to make predictions on new data:

\`\`\`python
predictions = model.predict(X_test)
# For classification: predictions is an array of probability distributions
# predictions[0] might be [0.01, 0.02, 0.95, 0.01, ...]
# meaning the model is 95% confident this is class 2

predicted_classes = predictions.argmax(axis=1)
print(f"Predicted class: {predicted_classes[0]}")
\`\`\`

## Evaluating the Model

Use \`model.evaluate()\` to measure performance on test data:

\`\`\`python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
\`\`\`

## Complete Example: Classifying Digits

Here is a complete example that loads a dataset, builds a model, trains it, and evaluates it:

\`\`\`python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Load and prepare data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1]
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 2. Build the model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# 4. Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

# 5. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 6. Predict
predictions = model.predict(X_test[:5])
print("Predicted classes:", predictions.argmax(axis=1))
print("Actual classes:", y_test[:5].argmax(axis=1))
\`\`\`

## Model Summary

You can always inspect your model's architecture with \`model.summary()\`:

\`\`\`python
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 128)               100480
#  dense_1 (Dense)             (None, 64)                8256
#  dense_2 (Dense)             (None, 10)                650
# =================================================================
# Total params: 109386 (427.29 KB)
# Trainable params: 109386 (427.29 KB)
\`\`\`

Understanding the parameter count for each layer is essential for estimating model complexity and memory requirements. A Dense layer with input size I and output size O has I * O + O parameters (the +O accounts for the bias terms).`,
      commonMistakes: `## Common Mistakes

### 1. Wrong Loss Function for the Task
\`\`\`python
# WRONG: using MSE for classification
model.compile(optimizer='adam', loss='mse')

# RIGHT: use crossentropy for classification
model.compile(optimizer='adam', loss='categorical_crossentropy')
\`\`\`
Using MSE for classification will technically work but converges poorly and produces sub-optimal results.

### 2. Forgetting to Normalize Data
\`\`\`python
# WRONG: feeding raw pixel values (0-255)
model.fit(X_train, y_train)

# RIGHT: normalize to 0-1 range first
X_train = X_train.astype('float32') / 255.0
model.fit(X_train, y_train)
\`\`\`
Neural networks learn much better when input features are on a similar scale, typically 0 to 1 or -1 to 1.

### 3. Mismatched Output Shape
\`\`\`python
# If labels are integers [0, 1, 2, ...]:
# WRONG: using categorical_crossentropy (expects one-hot)
model.compile(loss='categorical_crossentropy')

# RIGHT: use sparse_categorical_crossentropy (accepts integer labels)
model.compile(loss='sparse_categorical_crossentropy')
\`\`\`
Always match the loss function to the format of your labels.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-085',
        lessonId: lesson29.id,
        prompt:
          'Create a function build_model_summary(layers) that takes a list of tuples (name, input_size, output_size) and prints a formatted model summary. For each layer, parameter count = input_size * output_size + output_size (for bias). Print each layer as "{name}: {params} params" then "Total: {total} params". Test with [("Dense_1", 784, 128), ("Dense_2", 128, 10)].',
        starterCode:
          'def build_model_summary(layers):\n    # Calculate params for each layer and print summary\n    pass\n\nlayers = [("Dense_1", 784, 128), ("Dense_2", 128, 10)]\nbuild_model_summary(layers)\n',
        expectedOutput:
          'Dense_1: 100480 params\nDense_2: 1290 params\nTotal: 101770 params',
        testCode: '',
        hints: JSON.stringify([
          'For each layer (name, inp, out): params = inp * out + out',
          'Keep a running total of all params',
          'Print each layer with f"{name}: {params} params" then print the total',
        ]),
        order: 1,
      },
      {
        id: 'exercise-086',
        lessonId: lesson29.id,
        prompt:
          'Write a function one_hot_encode(labels, num_classes) that converts integer labels to one-hot vectors using numpy. Test with labels=[0, 2, 1] and num_classes=3. Print the result array.',
        starterCode:
          'import numpy as np\n\ndef one_hot_encode(labels, num_classes):\n    # Create a zero matrix and set the appropriate positions to 1\n    pass\n\nresult = one_hot_encode([0, 2, 1], 3)\nprint(result)\n',
        expectedOutput: '[[1. 0. 0.]\n [0. 0. 1.]\n [0. 1. 0.]]',
        testCode: '',
        hints: JSON.stringify([
          'Create a zeros matrix: np.zeros((len(labels), num_classes))',
          'Loop through labels and set the corresponding column to 1.0',
          'For each i, label: result[i][label] = 1.0',
        ]),
        order: 2,
      },
      {
        id: 'exercise-087',
        lessonId: lesson29.id,
        prompt:
          'Implement binary_crossentropy(y_true, y_pred) using numpy. Clip predictions to [1e-7, 1-1e-7] to avoid log(0). Compute the mean of per-sample losses: -[y*log(p) + (1-y)*log(1-p)]. Test with y_true=[1,0,1,1] and y_pred=[0.9,0.1,0.8,0.7]. Print the loss rounded to 4 decimal places.',
        starterCode:
          'import numpy as np\n\ndef binary_crossentropy(y_true, y_pred):\n    # Clip predictions, compute BCE loss\n    pass\n\ny_true = np.array([1, 0, 1, 1])\ny_pred = np.array([0.9, 0.1, 0.8, 0.7])\nprint(round(binary_crossentropy(y_true, y_pred), 4))\n',
        expectedOutput: '0.1976',
        testCode: '',
        hints: JSON.stringify([
          'First clip: y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)',
          'Then compute: -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))',
          'Return the scalar result from np.mean',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-085',
        lessonId: lesson29.id,
        question:
          'Which loss function is typically used for binary classification in Keras?',
        type: 'MCQ',
        options: JSON.stringify([
          'mean_squared_error',
          'binary_crossentropy',
          'categorical_crossentropy',
          'sparse_crossentropy',
        ]),
        correctAnswer: 'binary_crossentropy',
        explanation:
          'binary_crossentropy is the standard loss function for binary classification problems where the output is a single probability (sigmoid activation). It measures the divergence between predicted probabilities and actual binary labels.',
        order: 1,
      },
      {
        id: 'quiz-086',
        lessonId: lesson29.id,
        question:
          'In Keras, model.fit() both trains the model and returns a history object.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation:
          'model.fit() trains the model on the provided data and returns a History object containing training metrics (loss, accuracy, etc.) for each epoch, which is useful for plotting learning curves.',
        order: 2,
      },
      {
        id: 'quiz-087',
        lessonId: lesson29.id,
        question: "What does the 'Dense' layer in Keras represent?",
        type: 'MCQ',
        options: JSON.stringify([
          'A convolutional layer',
          'A fully connected layer',
          'A pooling layer',
          'A dropout layer',
        ]),
        correctAnswer: 'A fully connected layer',
        explanation:
          'A Dense layer is a fully connected (FC) layer where every neuron is connected to every neuron in the previous layer. It is the most basic building block in neural networks.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 29: TensorFlow / Keras Basics');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 30 — Training Deep Networks
  // ═══════════════════════════════════════════════════════════════════
  const lesson30 = await prisma.lesson.create({
    data: {
      id: 'lesson-030',
      moduleId: mod.id,
      title: 'Training Deep Networks — Optimizers, Loss, Callbacks',
      slug: 'training-deep-networks',
      order: 3,
      content: `# Training Deep Networks — Optimizers, Loss, Callbacks

Building a neural network architecture is only half the battle. How you **train** it determines whether it learns meaningful patterns or fails spectacularly. In this lesson we cover the critical training decisions: which optimizer to use, how to set learning rate and batch size, how to detect and prevent overfitting, and how to use Keras callbacks to automate training management.

## Optimizers: How Weights Get Updated

An optimizer determines **how** the weights are adjusted based on the computed gradients. Different optimizers have different strategies.

### Stochastic Gradient Descent (SGD)

The simplest optimizer. It updates weights by subtracting the gradient multiplied by the learning rate:

\`\`\`python
# Conceptual SGD update:
# w = w - learning_rate * gradient

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='categorical_crossentropy')
\`\`\`

SGD is simple and works well for many problems but can be slow to converge and gets stuck in local minima. Adding **momentum** helps SGD continue moving in a consistent direction:

\`\`\`python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
\`\`\`

### Adam (Adaptive Moment Estimation)

Adam is the most popular optimizer in deep learning. It combines the benefits of two other methods — AdaGrad and RMSprop — by maintaining per-parameter learning rates that adapt based on the history of gradients.

\`\`\`python
model.compile(optimizer='adam',  # Uses default lr=0.001
              loss='categorical_crossentropy')

# Or with custom learning rate:
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy')
\`\`\`

**When to use Adam:** It is a great default choice. It converges faster than plain SGD in most cases and requires less hyperparameter tuning.

### RMSprop

Divides the learning rate by a running average of recent gradient magnitudes. It works well for recurrent neural networks and non-stationary problems.

\`\`\`python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
\`\`\`

### Optimizer Comparison

| Optimizer | Best For | Learning Rate | Notes |
|-----------|----------|---------------|-------|
| SGD + Momentum | Large-scale training, fine-tuning | 0.01 - 0.1 | Simple, reliable, needs tuning |
| Adam | General purpose, quick experiments | 0.001 (default) | Adapts per parameter |
| RMSprop | RNNs, non-stationary data | 0.001 (default) | Good for sequences |

## Learning Rate

The **learning rate** is arguably the single most important hyperparameter. It controls how large a step the optimizer takes during each weight update.

- **Too high**: The model overshoots the optimal weights, loss oscillates or diverges.
- **Too low**: Training takes extremely long and may get stuck in a poor local minimum.
- **Just right**: Loss decreases steadily and converges to a good solution.

A common strategy is to start with a moderate learning rate (e.g., 0.001 for Adam) and reduce it as training progresses.

\`\`\`python
# Simple gradient descent simulation
w = 0.0
lr = 0.1

for i in range(10):
    gradient = 2 * (w - 3)  # Minimize (w - 3)^2
    w -= lr * gradient
    print(f"Step {i+1}: w = {w:.4f}")
\`\`\`

## Batch Size and Epochs

- **Epoch**: One complete pass through the entire training dataset.
- **Batch size**: The number of samples the model sees before updating weights.

\`\`\`python
model.fit(X_train, y_train, epochs=50, batch_size=32)
\`\`\`

| Batch Size | Effect |
|------------|--------|
| Small (16-32) | Noisier updates, better generalization, slower |
| Large (256-512) | Smoother updates, faster per epoch, may generalize worse |
| Full dataset | True gradient descent, computationally expensive |

## Overfitting vs. Underfitting

**Overfitting** happens when the model learns the training data too well, including its noise, and performs poorly on new data. **Underfitting** happens when the model is too simple to capture the underlying patterns.

\`\`\`
Training Loss: 0.01  |  Validation Loss: 0.50  →  Overfitting
Training Loss: 0.40  |  Validation Loss: 0.42  →  Good fit
Training Loss: 0.80  |  Validation Loss: 0.85  →  Underfitting
\`\`\`

### Detecting Overfitting

Always use a **validation split** or separate validation set to monitor overfitting:

\`\`\`python
history = model.fit(X_train, y_train,
                    epochs=50,
                    validation_split=0.2)  # 20% for validation
\`\`\`

When validation loss starts increasing while training loss continues decreasing, you are overfitting.

## Keras Callbacks

Callbacks are special functions that run at specific points during training. They automate common tasks like saving the best model or stopping training early.

### EarlyStopping

Stops training when a monitored metric (like validation loss) stops improving:

\`\`\`python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,        # Wait 5 epochs for improvement
    restore_best_weights=True,
)

model.fit(X_train, y_train, epochs=100,
          validation_split=0.2,
          callbacks=[early_stop])
\`\`\`

### ModelCheckpoint

Saves the model at regular intervals or whenever it improves:

\`\`\`python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
)
\`\`\`

### ReduceLROnPlateau

Reduces the learning rate when training plateaus:

\`\`\`python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,     # Multiply LR by 0.5
    patience=3,     # Wait 3 epochs
    min_lr=1e-6,
)

model.fit(X_train, y_train, epochs=100,
          validation_split=0.2,
          callbacks=[early_stop, checkpoint, reduce_lr])
\`\`\`

## Dropout

Dropout is a regularization technique that randomly sets a fraction of neuron outputs to zero during each training step. This forces the network to be more robust because it cannot rely on any single neuron.

\`\`\`python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),   # Drop 30% of neurons randomly
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax'),
])
\`\`\`

**Important:** Dropout is only active during training. During prediction (\`model.predict()\`), all neurons are used, and the outputs are scaled appropriately.

## Batch Normalization

Batch normalization normalizes the inputs to each layer, which helps the network train faster and more stably:

\`\`\`python
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax'),
])
\`\`\`

## Plotting Training History

Visualizing the training history is essential for diagnosing problems:

\`\`\`python
import matplotlib.pyplot as plt

# Plot training & validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
\`\`\`

If you see the training loss continuing to decrease while the validation loss goes up, that is the classic sign of overfitting. The gap between the two curves tells you how badly the model is overfitting.`,
      commonMistakes: `## Common Mistakes

### 1. Training Too Long Without Early Stopping
\`\`\`python
# WRONG: training for a fixed 200 epochs without checking
model.fit(X_train, y_train, epochs=200)

# RIGHT: use EarlyStopping to stop when validation loss stops improving
model.fit(X_train, y_train, epochs=200,
          validation_split=0.2,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
\`\`\`
Without early stopping, the model will almost certainly overfit after enough epochs.

### 2. Using Too High a Learning Rate
\`\`\`python
# WRONG: learning rate too large — loss may diverge
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

# RIGHT: start with default or smaller LR
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
\`\`\`
A learning rate of 0.1 is almost always too high for Adam. The default (0.001) is a good starting point.

### 3. Not Using Validation Data
\`\`\`python
# WRONG: no way to detect overfitting
model.fit(X_train, y_train, epochs=50)

# RIGHT: always monitor validation metrics
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
\`\`\`
Without validation data, you have no way of knowing if your model is overfitting. Always set aside validation data.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-088',
        lessonId: lesson30.id,
        prompt:
          'Implement simple SGD: start with w=0.0, learning_rate=0.1. For 5 iterations, compute gradient = 2*(w - 3) (minimizing (w-3) squared). Update w -= lr * gradient. Print w rounded to 4 decimal places after each step.',
        starterCode:
          'w = 0.0\nlr = 0.1\n\nfor i in range(5):\n    # Compute gradient and update w\n    pass\n',
        expectedOutput: '0.6\n1.08\n1.464\n1.7712\n2.017',
        testCode: '',
        hints: JSON.stringify([
          'Gradient of (w-3)^2 is 2*(w-3)',
          'Update rule: w = w - lr * gradient',
          'Print round(w, 4) after each update',
        ]),
        order: 1,
      },
      {
        id: 'exercise-089',
        lessonId: lesson30.id,
        prompt:
          'Simulate training history. Given train_loss=[0.9, 0.6, 0.4, 0.3, 0.25, 0.22, 0.21, 0.20] and val_loss=[0.85, 0.55, 0.42, 0.38, 0.40, 0.45, 0.50, 0.55], find the epoch (1-indexed) where val_loss is minimum (best epoch) and the epoch where overfitting starts (val_loss starts increasing). Print "Best epoch: {n}" and "Overfitting starts: epoch {n}".',
        starterCode:
          'train_loss = [0.9, 0.6, 0.4, 0.3, 0.25, 0.22, 0.21, 0.20]\nval_loss = [0.85, 0.55, 0.42, 0.38, 0.40, 0.45, 0.50, 0.55]\n\n# Find best epoch and overfitting start\n',
        expectedOutput: 'Best epoch: 4\nOverfitting starts: epoch 5',
        testCode: '',
        hints: JSON.stringify([
          'Best epoch is the index of minimum val_loss + 1 (for 1-indexing)',
          'Overfitting starts at the first epoch where val_loss increases from the previous epoch',
          'Use val_loss.index(min(val_loss)) + 1 for best epoch, then loop to find where val_loss[i] > val_loss[i-1]',
        ]),
        order: 2,
      },
      {
        id: 'exercise-090',
        lessonId: lesson30.id,
        prompt:
          'Implement a dropout function: given a numpy array and a dropout rate, randomly zero out elements and scale remaining values by 1/(1-rate). Use np.random.seed(42) for reproducibility. Apply to [1.0, 2.0, 3.0, 4.0, 5.0] with rate=0.4. Print the result as a list with each value rounded to 2 decimal places.',
        starterCode:
          'import numpy as np\n\ndef dropout(arr, rate):\n    np.random.seed(42)\n    # Generate random values, create mask, apply and scale\n    pass\n\nresult = dropout(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.4)\nprint([round(x, 2) for x in result])\n',
        expectedOutput: '[0.0, 3.33, 5.0, 6.67, 0.0]',
        testCode: '',
        hints: JSON.stringify([
          'Generate random values: np.random.random(len(arr))',
          'Create mask: (random_values >= rate) — elements below rate get zeroed',
          'Scale surviving values: result = arr * mask / (1 - rate)',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-088',
        lessonId: lesson30.id,
        question:
          'What is the main advantage of Adam optimizer over basic SGD?',
        type: 'MCQ',
        options: JSON.stringify([
          'It uses less memory',
          'It adapts the learning rate per parameter',
          'It prevents overfitting',
          'It trains for fewer epochs',
        ]),
        correctAnswer: 'It adapts the learning rate per parameter',
        explanation:
          'Adam maintains per-parameter learning rates that adapt based on the first and second moments of the gradients. This means each weight gets its own effective learning rate, leading to faster and more stable convergence.',
        order: 1,
      },
      {
        id: 'quiz-089',
        lessonId: lesson30.id,
        question:
          'Increasing dropout rate always improves model performance.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'False',
        explanation:
          'While dropout helps prevent overfitting, too much dropout can cause underfitting because the model does not have enough capacity to learn the patterns. The dropout rate must be tuned for each specific model and dataset.',
        order: 2,
      },
      {
        id: 'quiz-090',
        lessonId: lesson30.id,
        question:
          'What does EarlyStopping monitor to decide when to stop training?',
        type: 'MCQ',
        options: JSON.stringify([
          'Training accuracy only',
          'A specified metric like validation loss',
          'The number of parameters',
          'GPU memory usage',
        ]),
        correctAnswer: 'A specified metric like validation loss',
        explanation:
          'EarlyStopping monitors a metric you specify (typically val_loss or val_accuracy). When this metric stops improving for a specified number of epochs (patience), training is stopped to prevent overfitting.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 30: Training Deep Networks');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 31 — Image Classification with CNNs
  // ═══════════════════════════════════════════════════════════════════
  const lesson31 = await prisma.lesson.create({
    data: {
      id: 'lesson-031',
      moduleId: mod.id,
      title: 'Image Classification with CNNs',
      slug: 'image-classification-cnns',
      order: 4,
      content: `# Image Classification with CNNs

Convolutional Neural Networks (CNNs) are the backbone of modern computer vision. From recognizing faces to self-driving cars, CNNs have revolutionized how machines understand images. In this lesson you will learn why CNNs are essential for image tasks, how convolutional and pooling layers work, and how to build a CNN that classifies images.

## Why CNNs for Images?

A standard fully connected (Dense) neural network has a serious problem with images. Consider a 28x28 grayscale image like MNIST: that is 784 pixels. A Dense layer with 128 neurons would need 784 * 128 = 100,352 parameters. Now consider a typical color image of 224x224x3 (RGB): that is 150,528 input values. A single Dense layer with 256 neurons would need over 38 million parameters — just for one layer!

CNNs solve this problem with three key ideas:

1. **Local connectivity**: Each neuron only looks at a small region (receptive field) of the input, not the entire image.
2. **Weight sharing**: The same set of weights (a filter/kernel) is applied across the entire image.
3. **Spatial hierarchy**: Multiple layers build increasingly complex features — edges in early layers, shapes in middle layers, and objects in later layers.

## Convolutional Layers

A convolutional layer applies a set of **filters** (also called kernels) to the input. Each filter is a small matrix (typically 3x3 or 5x5) that slides across the input image, computing the dot product at each position.

\`\`\`
Input Image (5x5)          Filter (3x3)          Output (Feature Map)
+---+---+---+---+---+      +---+---+---+         +---+---+---+
| 0 | 1 | 2 | 3 | 4 |      | 1 | 0 |-1 |         | . | . | . |
+---+---+---+---+---+      +---+---+---+         +---+---+---+
| 5 | 6 | 7 | 8 | 9 |      | 1 | 0 |-1 |         | . | . | . |
+---+---+---+---+---+      +---+---+---+         +---+---+---+
|10 |11 |12 |13 |14 |      | 1 | 0 |-1 |         | . | . | . |
+---+---+---+---+---+                             +---+---+---+
|15 |16 |17 |18 |19 |
+---+---+---+---+---+
|20 |21 |22 |23 |24 |
+---+---+---+---+---+
\`\`\`

At each position, the filter is placed on top of the input, element-wise multiplication is performed, and the results are summed to produce one value in the output feature map.

### Convolution Parameters

- **Filters**: The number of different filters (each learns a different feature). More filters = more features detected.
- **Kernel size**: The dimensions of the filter (e.g., 3x3, 5x5). Smaller kernels capture fine details.
- **Stride**: How many pixels the filter moves at each step. Stride 1 means it moves one pixel at a time. Stride 2 moves two pixels, producing a smaller output.
- **Padding**: Whether to add zeros around the border of the input. "same" padding preserves the spatial dimensions; "valid" (no padding) reduces them.

\`\`\`python
import numpy as np

def convolve2d(image, kernel):
    """Simple 2D convolution with no padding, stride 1."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(patch * kernel)

    return output

# Example: edge detection kernel
image = np.arange(25).reshape(5, 5).astype(float)
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

result = convolve2d(image, kernel)
print(result)
\`\`\`

In Keras, you create convolutional layers like this:

\`\`\`python
from tensorflow.keras import layers

# 32 filters, each 3x3, with ReLU activation
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
\`\`\`

## Pooling Layers

Pooling reduces the spatial dimensions of the feature maps, which decreases computation and helps the network become more robust to small translations of the input.

### Max Pooling

Takes the maximum value in each pooling window:

\`\`\`
Input (4x4):                    Max Pool 2x2, stride 2:
+---+---+---+---+               +---+---+
| 1 | 3 | 2 | 4 |               | 6 | 4 |
+---+---+---+---+               +---+---+
| 5 | 6 | 1 | 2 |               | 8 | 6 |
+---+---+---+---+               +---+---+
| 7 | 8 | 3 | 1 |
+---+---+---+---+
| 4 | 2 | 6 | 5 |
+---+---+---+---+
\`\`\`

\`\`\`python
import numpy as np

def max_pool_2d(matrix, pool_size=2, stride=2):
    h, w = matrix.shape
    out_h = h // stride
    out_w = w // stride
    output = np.zeros((out_h, out_w), dtype=matrix.dtype)

    for i in range(out_h):
        for j in range(out_w):
            patch = matrix[i*stride:i*stride+pool_size,
                           j*stride:j*stride+pool_size]
            output[i, j] = np.max(patch)

    return output

matrix = np.array([[1, 3, 2, 4],
                   [5, 6, 1, 2],
                   [7, 8, 3, 1],
                   [4, 2, 6, 5]])

print(max_pool_2d(matrix))
# [[6 4]
#  [8 6]]
\`\`\`

### Average Pooling

Takes the average value instead of the maximum. Less commonly used than max pooling.

In Keras:
\`\`\`python
layers.MaxPooling2D(pool_size=(2, 2))
layers.AveragePooling2D(pool_size=(2, 2))
\`\`\`

## Feature Maps

Each convolutional layer produces a set of **feature maps** — one for each filter. Early layers detect simple features like edges and corners. Deeper layers combine these simple features into more complex patterns like textures, shapes, and eventually entire objects.

\`\`\`
Layer 1: Edges and gradients
Layer 2: Corners and simple shapes
Layer 3: Textures and parts
Layer 4: Object parts
Layer 5: Full objects
\`\`\`

## Building a CNN for MNIST

Here is a complete CNN for classifying handwritten digits:

\`\`\`python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and reshape (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
\`\`\`

## Data Augmentation

Data augmentation artificially increases the training set by applying random transformations to existing images — rotations, flips, zooms, shifts. This helps the model generalize better.

\`\`\`python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)

model.fit(datagen.flow(X_train, y_train, batch_size=64),
          epochs=10, validation_data=(X_test, y_test))
\`\`\`

## Transfer Learning

Instead of training a CNN from scratch, you can use a pre-trained model (like VGG16, ResNet, or MobileNet) that was trained on millions of images. You replace the final classification layer with your own and fine-tune:

\`\`\`python
base_model = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # Freeze pre-trained weights

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])
\`\`\`

Transfer learning lets you achieve excellent results with much less training data and computational resources than training from scratch.`,
      commonMistakes: `## Common Mistakes

### 1. Not Normalizing Pixel Values to 0-1
\`\`\`python
# WRONG: feeding raw pixel values (0-255)
X_train = X_train.reshape(-1, 28, 28, 1)
model.fit(X_train, y_train)

# RIGHT: normalize to [0, 1]
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
\`\`\`
Unnormalized pixel values cause gradient instability and make training much harder.

### 2. Wrong Input Shape
\`\`\`python
# WRONG: forgetting the channel dimension for grayscale images
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28)))

# RIGHT: include the channel dimension
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
\`\`\`
Conv2D expects input shape (height, width, channels). Grayscale images have 1 channel, RGB images have 3.

### 3. Too Many Dense Layers After Conv Layers
\`\`\`python
# WRONG: too many FC layers after flattening (overfitting risk)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))

# RIGHT: typically 1-2 Dense layers after Flatten is enough
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
\`\`\`
Excessive dense layers after convolutions add many parameters and increase overfitting without improving feature extraction.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-091',
        lessonId: lesson31.id,
        prompt:
          'Implement a simple 2D convolution with no padding and stride 1 using numpy. Given a 5x5 input (np.arange(25).reshape(5,5)) and a 3x3 kernel ([[1,0,-1],[1,0,-1],[1,0,-1]]), compute and print the 3x3 output feature map.',
        starterCode:
          'import numpy as np\n\ndef convolve2d(image, kernel):\n    # Implement 2D convolution: no padding, stride 1\n    pass\n\ninput_matrix = np.arange(25).reshape(5, 5)\nkernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])\n\noutput = convolve2d(input_matrix, kernel)\nprint(output)\n',
        expectedOutput: '[[-6 -6 -6]\n [-6 -6 -6]\n [-6 -6 -6]]',
        testCode: '',
        hints: JSON.stringify([
          'Output size = (input_size - kernel_size + 1) for each dimension',
          'For each output position (i,j), extract the patch image[i:i+kh, j:j+kw] and sum element-wise product with kernel',
          'Use np.sum(patch * kernel) to compute each output cell',
        ]),
        order: 1,
      },
      {
        id: 'exercise-092',
        lessonId: lesson31.id,
        prompt:
          'Implement max pooling with pool_size=2 and stride=2 on a 4x4 matrix: [[1,3,2,4],[5,6,1,2],[7,8,3,1],[4,2,6,5]]. Print the 2x2 result.',
        starterCode:
          'import numpy as np\n\ndef max_pool(matrix, pool_size=2, stride=2):\n    # Implement max pooling\n    pass\n\nmatrix = np.array([[1, 3, 2, 4],\n                   [5, 6, 1, 2],\n                   [7, 8, 3, 1],\n                   [4, 2, 6, 5]])\n\nresult = max_pool(matrix)\nprint(result)\n',
        expectedOutput: '[[6 4]\n [8 6]]',
        testCode: '',
        hints: JSON.stringify([
          'Output size = input_size // stride for each dimension',
          'For each output position (i,j), extract the pool_size x pool_size patch',
          'Use np.max(patch) to get the maximum value in each patch',
        ]),
        order: 2,
      },
      {
        id: 'exercise-093',
        lessonId: lesson31.id,
        prompt:
          'Write a function that normalizes image pixel values from 0-255 range to 0-1 range. Test with pixels = [0, 128, 255, 64, 192]. Print the result as a list rounded to 4 decimal places.',
        starterCode:
          'def normalize_pixels(pixels):\n    # Normalize each pixel from 0-255 to 0-1\n    pass\n\npixels = [0, 128, 255, 64, 192]\nresult = normalize_pixels(pixels)\nprint(result)\n',
        expectedOutput: '[0.0, 0.502, 1.0, 0.251, 0.7529]',
        testCode: '',
        hints: JSON.stringify([
          'Divide each pixel value by 255',
          'Round each result to 4 decimal places',
          'Return [round(p / 255, 4) for p in pixels]',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-091',
        lessonId: lesson31.id,
        question: 'What is the primary purpose of a convolutional layer?',
        type: 'MCQ',
        options: JSON.stringify([
          'To reduce image size',
          'To detect local patterns/features',
          'To classify images',
          'To generate new images',
        ]),
        correctAnswer: 'To detect local patterns/features',
        explanation:
          'Convolutional layers apply learned filters to detect local patterns like edges, textures, and shapes. Each filter specializes in detecting a different type of feature across the entire input.',
        order: 1,
      },
      {
        id: 'quiz-092',
        lessonId: lesson31.id,
        question:
          'Max pooling reduces the spatial dimensions of feature maps.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation:
          'Max pooling (typically with pool_size=2 and stride=2) reduces each spatial dimension by half, which decreases computational cost and helps the model become more robust to small spatial translations.',
        order: 2,
      },
      {
        id: 'quiz-093',
        lessonId: lesson31.id,
        question:
          'Why should image pixel values be normalized to 0-1 before training?',
        type: 'MCQ',
        options: JSON.stringify([
          'To save memory',
          'To make images look better',
          'To help gradient descent converge faster',
          'To increase the number of features',
        ]),
        correctAnswer: 'To help gradient descent converge faster',
        explanation:
          'Normalizing inputs to a small range (0-1 or -1 to 1) helps gradient descent converge faster and more reliably. Large input values can cause exploding gradients and unstable training.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 31: Image Classification with CNNs');

  // ═══════════════════════════════════════════════════════════════════
  //  LESSON 32 — Text Classification with RNNs & Embeddings
  // ═══════════════════════════════════════════════════════════════════
  const lesson32 = await prisma.lesson.create({
    data: {
      id: 'lesson-032',
      moduleId: mod.id,
      title: 'Text Classification with RNNs & Embeddings',
      slug: 'text-classification-rnns',
      order: 5,
      content: `# Text Classification with RNNs & Embeddings

While CNNs excel at spatial data like images, Recurrent Neural Networks (RNNs) are designed for **sequential data** — text, time series, audio, and anything where order matters. In this lesson you will learn how RNNs process sequences, how embedding layers represent words as dense vectors, and how to build a sentiment classifier for text.

## Why RNNs for Sequences?

Consider the sentence "The movie was not good." A regular Dense network would treat each word independently, losing the crucial information that "not" modifies "good." RNNs solve this by processing words one at a time while maintaining a **hidden state** that carries information from previous steps.

\`\`\`
Step 1: "The"   → hidden_state_1
Step 2: "movie" → hidden_state_2 (incorporates info from step 1)
Step 3: "was"   → hidden_state_3 (incorporates info from steps 1-2)
Step 4: "not"   → hidden_state_4 (incorporates info from steps 1-3)
Step 5: "good"  → hidden_state_5 (incorporates info from steps 1-4)
                                    ↓
                              Final prediction
\`\`\`

At each time step, the RNN takes two inputs: the current word and the previous hidden state. It produces a new hidden state that encodes all the information seen so far.

## Embedding Layers

Before feeding text to an RNN, we need to convert words into numbers. An **embedding layer** maps each word to a dense vector of a specified dimension. Unlike one-hot encoding (which creates sparse, high-dimensional vectors), embeddings are dense, low-dimensional, and learned during training.

\`\`\`python
import tensorflow as tf
from tensorflow.keras import layers

# Vocabulary size of 10000, embedding dimension of 128
embedding_layer = layers.Embedding(input_dim=10000, output_dim=128)

# Input: integer indices [5, 23, 478, 1]
# Output: 4 vectors, each of dimension 128
\`\`\`

The key insight is that the embedding layer **learns** meaningful representations. Words with similar meanings end up with similar vectors — for example, "good" and "great" would have vectors that are close together in the embedding space.

## Text Preprocessing

Before building the model, raw text needs to be converted into a format the network can process.

### Step 1: Tokenization

Convert text into sequences of integer tokens:

\`\`\`python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)

# "the movie was great" → [1, 45, 12, 287]
print(tokenizer.word_index)  # {'the': 1, 'movie': 45, ...}
\`\`\`

### Step 2: Padding

Neural networks require fixed-length inputs. We pad shorter sequences with zeros:

\`\`\`python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences, maxlen=200, padding='post')
# Short sequences get zeros appended
# Long sequences get truncated to maxlen
\`\`\`

\`\`\`python
# Example:
# [1, 45, 12] with maxlen=5, post-padding → [1, 45, 12, 0, 0]
# [1, 45, 12] with maxlen=5, pre-padding  → [0, 0, 1, 45, 12]
\`\`\`

## SimpleRNN

The basic RNN layer in Keras:

\`\`\`python
model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.SimpleRNN(64),
    layers.Dense(1, activation='sigmoid'),  # Binary classification
])
\`\`\`

However, SimpleRNN has a major limitation: the **vanishing gradient problem**. As sequences get longer, the gradients flowing backward through time become extremely small, making it difficult for the network to learn long-range dependencies.

## LSTM (Long Short-Term Memory)

LSTM was specifically designed to solve the vanishing gradient problem. It uses a system of **gates** (forget gate, input gate, output gate) and a **cell state** to selectively remember and forget information over long sequences.

\`\`\`python
model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid'),
])
\`\`\`

The gates in an LSTM decide:
- **Forget gate**: What information from the cell state should be discarded?
- **Input gate**: What new information should be added to the cell state?
- **Output gate**: What part of the cell state should be output as the hidden state?

## GRU (Gated Recurrent Unit)

GRU is a simplified version of LSTM with fewer parameters. It combines the forget and input gates into a single "update gate" and merges the cell state and hidden state.

\`\`\`python
model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.GRU(64),
    layers.Dense(1, activation='sigmoid'),
])
\`\`\`

GRU typically performs comparably to LSTM but trains faster due to fewer parameters.

### RNN Variant Comparison

| Variant | Parameters | Long Dependencies | Speed |
|---------|-----------|-------------------|-------|
| SimpleRNN | Fewest | Poor (vanishing gradients) | Fastest |
| LSTM | Most | Excellent | Slowest |
| GRU | Medium | Good | Medium |

## Bidirectional RNNs

A standard RNN only reads the sequence from left to right. A **bidirectional** RNN processes the sequence in both directions and concatenates the results, allowing the model to capture context from both past and future tokens.

\`\`\`python
model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(1, activation='sigmoid'),
])
\`\`\`

Bidirectional models often perform better for tasks like sentiment analysis and named entity recognition where the full context of a word matters.

## Building a Sentiment Classifier

Here is a complete example building a sentiment classifier on the IMDB movie reviews dataset:

\`\`\`python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load IMDB dataset (already tokenized)
vocab_size = 10000
max_len = 200
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
    num_words=vocab_size
)

# 2. Pad sequences to fixed length
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# 3. Build the model
model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.Bidirectional(layers.LSTM(64, dropout=0.2)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# 5. Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
\`\`\`

## Limitations of RNNs and Introduction to Transformers

Despite their power, RNNs have limitations:

1. **Sequential processing**: RNNs must process tokens one at a time, making them slow to train and unable to fully leverage GPU parallelism.
2. **Limited long-range memory**: Even LSTMs struggle with very long sequences (thousands of tokens).
3. **No direct attention**: An RNN must compress all information into a fixed-size hidden state, which becomes a bottleneck for long sequences.

**Transformers** (introduced in the 2017 paper "Attention Is All You Need") solve all three problems. They process all tokens in parallel using a mechanism called **self-attention**, which allows every token to directly attend to every other token in the sequence. Models like BERT, GPT, and T5 are all based on the Transformer architecture and have largely replaced RNNs for most NLP tasks.

However, understanding RNNs is still valuable because they provide the conceptual foundation for sequence modeling, and they remain useful for certain tasks where simplicity and low computational cost are priorities.`,
      commonMistakes: `## Common Mistakes

### 1. Not Padding Sequences
\`\`\`python
# WRONG: sequences of different lengths
sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
model.fit(sequences, labels)  # Error!

# RIGHT: pad to fixed length
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(sequences, maxlen=10, padding='post')
model.fit(padded, labels)
\`\`\`
Neural networks require fixed-size inputs. Always pad (or truncate) sequences to a consistent length.

### 2. Wrong Vocabulary Size
\`\`\`python
# WRONG: embedding vocabulary doesn't match tokenizer
tokenizer = Tokenizer(num_words=5000)
model.add(layers.Embedding(10000, 128))  # Mismatch!

# RIGHT: use the same vocabulary size
tokenizer = Tokenizer(num_words=5000)
model.add(layers.Embedding(5000, 128))
\`\`\`
The embedding layer's input_dim must match the number of words in your tokenizer.

### 3. Forgetting to Set Max Sequence Length
\`\`\`python
# WRONG: no maxlen — sequences can be arbitrarily long
padded = pad_sequences(sequences)

# RIGHT: set a reasonable maxlen
padded = pad_sequences(sequences, maxlen=200)
\`\`\`
Without maxlen, the padding defaults to the length of the longest sequence, which can be extremely large and waste memory. Always set an explicit maxlen based on your data.`,
    },
  });

  await prisma.exercise.createMany({
    data: [
      {
        id: 'exercise-094',
        lessonId: lesson32.id,
        prompt:
          'Implement a simple tokenizer. Given a vocabulary {"the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5} and the sentence "the cat sat on the mat", convert the sentence to integer tokens. Print the result as a list.',
        starterCode:
          'vocab = {"the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}\nsentence = "the cat sat on the mat"\n\n# Tokenize the sentence using the vocabulary\n',
        expectedOutput: '[1, 2, 3, 4, 1, 5]',
        testCode: '',
        hints: JSON.stringify([
          'Split the sentence into words with sentence.split()',
          'Look up each word in the vocab dictionary',
          'Use a list comprehension: [vocab[word] for word in sentence.split()]',
        ]),
        order: 1,
      },
      {
        id: 'exercise-095',
        lessonId: lesson32.id,
        prompt:
          'Implement padding for sequences. Given sequences [[1,2,3], [4,5], [6,7,8,9]], pad to maxlen=5 with zeros (post-padding). Print the result as a list of lists.',
        starterCode:
          'def pad_sequences(sequences, maxlen):\n    # Pad each sequence to maxlen with zeros (post-padding)\n    pass\n\nsequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]\nresult = pad_sequences(sequences, maxlen=5)\nprint(result)\n',
        expectedOutput: '[[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0]]',
        testCode: '',
        hints: JSON.stringify([
          'For each sequence, extend it with zeros until it reaches maxlen',
          'Use: seq + [0] * (maxlen - len(seq)) for each sequence',
          'Return [seq + [0] * (maxlen - len(seq)) for seq in sequences]',
        ]),
        order: 2,
      },
      {
        id: 'exercise-096',
        lessonId: lesson32.id,
        prompt:
          'Implement a softmax function using numpy. Apply it to [2.0, 1.0, 0.1]. Print the result formatted so each value shows exactly 4 decimal places.',
        starterCode:
          'import numpy as np\n\ndef softmax(x):\n    # Compute softmax: exp(x) / sum(exp(x))\n    # Subtract max for numerical stability\n    pass\n\nresult = softmax(np.array([2.0, 1.0, 0.1]))\nprint("[" + ", ".join(f"{v:.4f}" for v in result) + "]")\n',
        expectedOutput: '[0.6590, 0.2424, 0.0986]',
        testCode: '',
        hints: JSON.stringify([
          'Subtract max(x) for numerical stability: exp_x = np.exp(x - np.max(x))',
          'Divide by sum: return exp_x / exp_x.sum()',
          'The subtraction of max does not change the result but prevents overflow',
        ]),
        order: 3,
      },
    ],
  });

  await prisma.quizQuestion.createMany({
    data: [
      {
        id: 'quiz-094',
        lessonId: lesson32.id,
        question: 'What is the purpose of an embedding layer in NLP?',
        type: 'MCQ',
        options: JSON.stringify([
          'To tokenize text',
          'To map words to dense vector representations',
          'To pad sequences',
          'To compute attention scores',
        ]),
        correctAnswer: 'To map words to dense vector representations',
        explanation:
          'An embedding layer maps each word (represented as an integer token) to a dense, low-dimensional vector. These vectors are learned during training and capture semantic relationships between words.',
        order: 1,
      },
      {
        id: 'quiz-095',
        lessonId: lesson32.id,
        question:
          'LSTMs were designed to solve the vanishing gradient problem in standard RNNs.',
        type: 'TRUE_FALSE',
        options: JSON.stringify(['True', 'False']),
        correctAnswer: 'True',
        explanation:
          'LSTMs use a cell state with gating mechanisms (forget, input, and output gates) that allow gradients to flow through long sequences without vanishing. This enables LSTMs to learn long-range dependencies that SimpleRNNs cannot.',
        order: 2,
      },
      {
        id: 'quiz-096',
        lessonId: lesson32.id,
        question:
          'What preprocessing step converts variable-length text sequences to fixed-length?',
        type: 'MCQ',
        options: JSON.stringify([
          'Tokenization',
          'Embedding',
          'Padding',
          'Normalization',
        ]),
        correctAnswer: 'Padding',
        explanation:
          'Padding adds zeros to shorter sequences (or truncates longer ones) to ensure all sequences have the same length, which is required for batch processing in neural networks.',
        order: 3,
      },
    ],
  });
  console.log('Seeded Lesson 32: Text Classification with RNNs & Embeddings');

  // ═══════════════════════════════════════════════════════════════════
  //  PROJECTS
  // ═══════════════════════════════════════════════════════════════════
  await prisma.project.create({
    data: {
      id: 'project-009',
      title: 'MNIST Digit Classifier',
      slug: 'mnist-classifier',
      stage: 'DEEP_LEARNING',
      order: 9,
      brief:
        'Build and train a neural network that recognizes handwritten digits from the MNIST dataset with >95% accuracy.',
      requirements: JSON.stringify([
        'Load the MNIST dataset and explore its structure (60,000 training images, 10,000 test images of 28x28 pixels)',
        'Preprocess data: normalize pixel values to 0-1 range and reshape inputs appropriately',
        'Build a neural network model using Keras Sequential API with at least two hidden layers',
        'Train the model with appropriate loss function, optimizer, and validation split',
        'Evaluate on the test set and report accuracy, displaying sample predictions alongside true labels',
      ]),
      stretchGoals: JSON.stringify([
        'Replace the Dense network with a CNN architecture (Conv2D + MaxPooling2D layers) and compare accuracy',
        'Add data augmentation (rotation, shift, zoom) and measure its impact on test accuracy',
        'Generate and display a confusion matrix showing which digits are most commonly confused',
      ]),
      steps: JSON.stringify([
        {
          title: 'Load and Explore the Data',
          description:
            'Use keras.datasets.mnist.load_data() to load the dataset. Print shapes, display a few sample images with their labels using matplotlib.',
        },
        {
          title: 'Preprocess the Data',
          description:
            'Normalize pixel values by dividing by 255.0. Reshape the data for the model input. One-hot encode the labels if using categorical_crossentropy.',
        },
        {
          title: 'Build the Model Architecture',
          description:
            'Create a Sequential model with Dense layers (e.g., 784 -> 128 -> 64 -> 10). Use ReLU for hidden layers and softmax for the output layer.',
        },
        {
          title: 'Train the Model',
          description:
            'Compile with Adam optimizer and categorical_crossentropy loss. Train for 10-20 epochs with batch_size=32 and validation_split=0.2. Use EarlyStopping callback.',
        },
        {
          title: 'Evaluate and Visualize Results',
          description:
            'Evaluate on the test set. Plot training/validation loss and accuracy curves. Display sample predictions with their true labels.',
        },
      ]),
      rubric: JSON.stringify([
        {
          criterion: 'Accuracy',
          description:
            'Model achieves >95% accuracy on the MNIST test set. Stretch: >98% with CNN.',
        },
        {
          criterion: 'Architecture',
          description:
            'Model uses appropriate layer types, activation functions, and output configuration for multi-class classification.',
        },
        {
          criterion: 'Training Process',
          description:
            'Proper use of validation data, callbacks, and training monitoring. Training history is plotted and analyzed.',
        },
        {
          criterion: 'Visualization',
          description:
            'Sample predictions are displayed alongside true labels. Training curves show loss and accuracy over epochs.',
        },
      ]),
      solutionUrl: null,
    },
  });

  await prisma.project.create({
    data: {
      id: 'project-010',
      title: 'Sentiment Analysis on Movie Reviews',
      slug: 'sentiment-analysis',
      stage: 'DEEP_LEARNING',
      order: 10,
      brief:
        'Build a text classifier that determines whether movie reviews are positive or negative using RNNs.',
      requirements: JSON.stringify([
        'Load the IMDB movie reviews dataset (25,000 training and 25,000 test reviews)',
        'Preprocess text data: tokenize, build vocabulary, and pad sequences to a fixed maximum length',
        'Build an RNN-based model using Embedding + LSTM/GRU layers for binary sentiment classification',
        'Train the model with binary_crossentropy loss, tracking training and validation accuracy',
        'Evaluate on the test set and demonstrate predictions on sample reviews with confidence scores',
      ]),
      stretchGoals: JSON.stringify([
        'Compare LSTM vs GRU performance: train both architectures and report accuracy and training time differences',
        'Add a simple attention mechanism or use Bidirectional wrapper to improve classification accuracy',
        'Build a simple prediction pipeline that takes a raw text string and outputs sentiment with confidence',
      ]),
      steps: JSON.stringify([
        {
          title: 'Load and Explore the Data',
          description:
            'Use keras.datasets.imdb.load_data() with a vocabulary size limit. Decode sample reviews back to text to understand the data format.',
        },
        {
          title: 'Preprocess the Text',
          description:
            'Pad all sequences to a fixed maxlen (e.g., 200). Inspect the distribution of review lengths to choose an appropriate maxlen.',
        },
        {
          title: 'Build the RNN Model',
          description:
            'Create a Sequential model with Embedding, LSTM (or GRU), and Dense layers. Add Dropout for regularization.',
        },
        {
          title: 'Train and Monitor',
          description:
            'Compile with Adam and binary_crossentropy. Train with validation split and EarlyStopping. Plot training history.',
        },
        {
          title: 'Evaluate and Analyze',
          description:
            'Report test accuracy. Show predictions on sample reviews. Analyze which types of reviews the model gets wrong.',
        },
      ]),
      rubric: JSON.stringify([
        {
          criterion: 'Accuracy',
          description:
            'Model achieves >85% accuracy on the IMDB test set. Stretch: >88% with bidirectional LSTM.',
        },
        {
          criterion: 'Text Preprocessing',
          description:
            'Proper tokenization, vocabulary management, and sequence padding. Appropriate maxlen chosen and justified.',
        },
        {
          criterion: 'Model Architecture',
          description:
            'Appropriate use of Embedding, RNN layers (LSTM/GRU), Dropout, and Dense output. Architecture decisions are justified.',
        },
        {
          criterion: 'Analysis',
          description:
            'Training curves are plotted and analyzed. Sample predictions demonstrate the model working on real text. Error analysis identifies failure modes.',
        },
      ]),
      solutionUrl: null,
    },
  });
  console.log('Seeded Stage 5 Projects');

  console.log('Module 8 (Deep Learning Foundations) seeding complete!');
}

module.exports = { seedModule8 };
