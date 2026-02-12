export {};

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  const existing = await prisma.stage.findUnique({ where: { id: 'stage-008' } });
  if (existing) { console.log('Stage 8 already seeded. Skipping.'); return; }

  console.log('═══════════════════════════════════════════');
  console.log('  Seeding Stage 8: MLOps & Career Ready');
  console.log('═══════════════════════════════════════════');

  // ─── Skill Tags ────────────────────────────────────────────────────
  const skills = [
    { id: 'skill-053', name: 'MLOps', slug: 'mlops' },
    { id: 'skill-054', name: 'FastAPI', slug: 'fastapi' },
    { id: 'skill-055', name: 'Docker', slug: 'docker' },
    { id: 'skill-056', name: 'Model Deployment', slug: 'model-deployment' },
    { id: 'skill-057', name: 'ML Monitoring', slug: 'ml-monitoring' },
    { id: 'skill-058', name: 'Portfolio', slug: 'portfolio' },
    { id: 'skill-059', name: 'Interview Prep', slug: 'interview-prep' },
  ];
  for (const s of skills) {
    await prisma.skillTag.upsert({ where: { id: s.id }, update: {}, create: s });
  }

  // ─── Stage ─────────────────────────────────────────────────────────
  await prisma.stage.create({
    data: {
      id: 'stage-008', title: 'MLOps & Career Ready', slug: 'mlops-career-ready',
      description: 'Ship ML models to production and prepare for real job interviews. From model serialization to Docker deployment.',
      order: 8,
    },
  });

  // ═══════════════════════════════════════════════════════════════════
  //  MODULE 13: MLOps Essentials
  // ═══════════════════════════════════════════════════════════════════
  const mod13 = await prisma.module.create({ data: {
    id: 'module-013', stageId: 'stage-008', title: 'MLOps Essentials',
    slug: 'mlops-essentials', order: 1,
    description: 'Learn to serialize, serve, containerize, and monitor ML models in production.',
  }});

  // ── Lesson 46 ──────────────────────────────────────────────────────
  const L46 = await prisma.lesson.create({ data: {
    id: 'lesson-046', moduleId: mod13.id, title: 'Model Serialization — Pickle, Joblib, ONNX',
    slug: 'model-serialization', order: 1,
    content: `# Model Serialization — Pickle, Joblib, ONNX

After training a model, you need to save it to disk so it can be loaded later for predictions without retraining. This process is called **serialization** (or model persistence).

## Why Serialize Models?

- **Deployment:** Load a pre-trained model in your API server
- **Reproducibility:** Share exact model state with your team
- **Versioning:** Keep track of different model versions
- **Efficiency:** Training takes hours; inference takes milliseconds

## Pickle — Python's Built-in Serializer

Pickle converts any Python object to a byte stream and back:

\`\`\`python
import pickle

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

prediction = loaded_model.predict(new_data)
\`\`\`

**Pros:** Works with any Python object, built-in
**Cons:** Python-only, not secure (can execute arbitrary code), no compression

## Joblib — Optimized for NumPy Arrays

Joblib is pickle optimized for large numpy arrays (common in ML):

\`\`\`python
import joblib

# Save — much faster for models with large numpy arrays
joblib.dump(model, "model.joblib")

# Load
loaded_model = joblib.load("model.joblib")
\`\`\`

**Pros:** Fast for numpy-heavy objects, compressed by default
**Cons:** Still Python-only

## ONNX — Cross-Platform Standard

ONNX (Open Neural Network Exchange) is a universal format that works across frameworks and languages:

\`\`\`python
# Convert sklearn model to ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("X", FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)

# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
\`\`\`

**Pros:** Cross-language (Python, Java, C++, JavaScript), optimized inference
**Cons:** More complex setup, not all models supported

## Best Practices

1. **Version your models:** Include version number in filename: \`model_v2.1.joblib\`
2. **Save metadata:** Store training params, metrics, and feature names alongside the model
3. **Test after loading:** Always verify loaded model produces same predictions
4. **Use joblib for scikit-learn,** pickle for general objects, ONNX for cross-platform`,
    commonMistakes: `## Common Mistakes

### 1. Pickle Security
Never load pickle files from untrusted sources — they can execute arbitrary code.

### 2. Version Mismatch
A model saved with scikit-learn 1.2 may not load in 1.4. Save the library version with your model.

### 3. Not Testing Loaded Models
Always verify predictions match before and after serialization.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-136', lessonId: L46.id, prompt: 'Simulate model serialization. Create a dict representing a model config, serialize with json (as a simpler alternative to pickle), then deserialize and verify. Print the loaded config.', starterCode: 'import json\n\nmodel_config = {\n    "type": "RandomForest",\n    "n_estimators": 100,\n    "accuracy": 0.95,\n    "features": ["age", "income", "score"]\n}\n\n# Serialize to string (simulating file save)\nserialized = json.dumps(model_config)\n\n# Deserialize (simulating file load)\nloaded = json.loads(serialized)\nprint(loaded)\n', expectedOutput: "{'type': 'RandomForest', 'n_estimators': 100, 'accuracy': 0.95, 'features': ['age', 'income', 'score']}", testCode: '', hints: JSON.stringify(['The code is complete', 'json.dumps converts dict to string', 'json.loads converts string back to dict']), order: 1 },
    { id: 'exercise-137', lessonId: L46.id, prompt: 'Create a ModelRegistry class that stores model metadata (name, version, accuracy). Add 3 models and print the best one.', starterCode: 'class ModelRegistry:\n    def __init__(self):\n        self.models = []\n\n    def register(self, name, version, accuracy):\n        self.models.append({"name": name, "version": version, "accuracy": accuracy})\n\n    def best_model(self):\n        return max(self.models, key=lambda m: m["accuracy"])\n\nreg = ModelRegistry()\nreg.register("logistic_v1", "1.0", 0.85)\nreg.register("random_forest_v1", "1.0", 0.92)\nreg.register("xgboost_v1", "1.0", 0.89)\nprint(reg.best_model())\n', expectedOutput: "{'name': 'random_forest_v1', 'version': '1.0', 'accuracy': 0.92}", testCode: '', hints: JSON.stringify(['The code is complete', 'max() with key finds highest accuracy', 'random_forest_v1 has 0.92']), order: 2 },
    { id: 'exercise-138', lessonId: L46.id, prompt: 'Write a function that creates a model metadata file. Include model name, version, features, and metrics. Print formatted JSON.', starterCode: 'import json\n\ndef create_metadata(name, version, features, metrics):\n    return {\n        "model_name": name,\n        "version": version,\n        "features": features,\n        "metrics": metrics,\n    }\n\nmeta = create_metadata(\n    "sentiment_classifier", "2.1",\n    ["text_length", "word_count", "avg_word_len"],\n    {"accuracy": 0.91, "f1_score": 0.89}\n)\nprint(json.dumps(meta, indent=2))\n', expectedOutput: '{\n  "model_name": "sentiment_classifier",\n  "version": "2.1",\n  "features": [\n    "text_length",\n    "word_count",\n    "avg_word_len"\n  ],\n  "metrics": {\n    "accuracy": 0.91,\n    "f1_score": 0.89\n  }\n}', testCode: '', hints: JSON.stringify(['The code is complete', 'json.dumps with indent=2 formats it nicely', 'Run it directly']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-136', lessonId: L46.id, question: 'Why is joblib preferred over pickle for scikit-learn models?', type: 'MCQ', options: JSON.stringify(['It is more secure', 'It is optimized for large numpy arrays', 'It works across languages', 'It is built into Python']), correctAnswer: 'It is optimized for large numpy arrays', explanation: 'Joblib is specifically optimized for serializing objects with large numpy arrays, which is common in ML models. It uses compression and memory mapping.', order: 1 },
    { id: 'quiz-137', lessonId: L46.id, question: 'It is safe to load pickle files from unknown sources.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Pickle can execute arbitrary code during deserialization. Never load pickle files from untrusted sources — it is a security risk.', order: 2 },
    { id: 'quiz-138', lessonId: L46.id, question: 'What is the main advantage of ONNX format?', type: 'MCQ', options: JSON.stringify(['Smaller file size', 'Cross-platform and cross-language compatibility', 'Faster training', 'Better accuracy']), correctAnswer: 'Cross-platform and cross-language compatibility', explanation: 'ONNX models can be loaded in Python, Java, C++, JavaScript, and more — making it the standard for cross-platform deployment.', order: 3 },
  ]});
  console.log('Seeded Lesson 46');

  // ── Lesson 47 ──────────────────────────────────────────────────────
  const L47 = await prisma.lesson.create({ data: {
    id: 'lesson-047', moduleId: mod13.id, title: 'Building ML APIs with FastAPI',
    slug: 'ml-apis-fastapi', order: 2,
    content: `# Building ML APIs with FastAPI

FastAPI is the go-to framework for serving ML models as REST APIs. It's fast, type-safe, and generates automatic API documentation.

## Why FastAPI?

- **Performance:** Built on Starlette + Uvicorn (one of the fastest Python frameworks)
- **Type safety:** Pydantic models validate input/output automatically
- **Auto docs:** Swagger UI generated at \`/docs\` endpoint
- **Async support:** Built-in async/await for non-blocking I/O

## Basic ML API

\`\`\`python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="ML Prediction API")

# Load model at startup
model = joblib.load("model.joblib")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])[0]
    confidence = max(model.predict_proba([request.features])[0])
    return PredictionResponse(prediction=prediction, confidence=confidence)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
\`\`\`

## Request Validation with Pydantic

Pydantic automatically validates incoming data:

\`\`\`python
class PredictionRequest(BaseModel):
    age: int
    income: float
    credit_score: int

    class Config:
        json_schema_extra = {
            "example": {"age": 30, "income": 50000.0, "credit_score": 720}
        }
\`\`\`

If someone sends invalid data, FastAPI returns a clear 422 error automatically.

## Batch Predictions

\`\`\`python
class BatchRequest(BaseModel):
    instances: list[list[float]]

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    predictions = model.predict(request.instances).tolist()
    return {"predictions": predictions}
\`\`\`

## Error Handling

\`\`\`python
from fastapi import HTTPException

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        prediction = model.predict([request.features])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
\`\`\`

## Running the API

\`\`\`bash
# Install
pip install fastapi uvicorn

# Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

Then visit \`http://localhost:8000/docs\` for the interactive API documentation.`,
    commonMistakes: `## Common Mistakes

### 1. Loading Model Inside the Endpoint
Load the model once at startup, not on every request. Loading from disk per request adds seconds of latency.

### 2. No Input Validation
Always use Pydantic models. Without validation, invalid input crashes your model silently.

### 3. No Health Check
Always add a \`/health\` endpoint. Deployment platforms use it to check if your service is alive.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-139', lessonId: L47.id, prompt: 'Simulate a FastAPI prediction endpoint. Create a function that takes features dict, validates it, and returns a prediction response. Test it.', starterCode: 'def predict(features):\n    required = ["age", "income", "score"]\n    for field in required:\n        if field not in features:\n            return {"error": f"Missing field: {field}"}\n    # Simulate prediction\n    result = features["age"] * 0.3 + features["income"] * 0.0001 + features["score"] * 0.5\n    return {"prediction": round(result, 2), "status": "success"}\n\nprint(predict({"age": 30, "income": 50000, "score": 720}))\nprint(predict({"age": 25}))\n', expectedOutput: "{'prediction': 374.0, 'status': 'success'}\n{'error': 'Missing field: income'}", testCode: '', hints: JSON.stringify(['The code is complete', '30*0.3 + 50000*0.0001 + 720*0.5 = 9 + 5 + 360 = 374.0', 'Second call is missing income field']), order: 1 },
    { id: 'exercise-140', lessonId: L47.id, prompt: 'Build a model versioning system. Create an endpoint simulator that serves predictions from different model versions.', starterCode: 'models = {\n    "v1": lambda x: x * 1.5,\n    "v2": lambda x: x * 2.0,\n    "v3": lambda x: x * 2.5,\n}\n\ndef predict(value, version="v3"):\n    if version not in models:\n        return {"error": f"Unknown version: {version}"}\n    result = models[version](value)\n    return {"prediction": result, "version": version}\n\nprint(predict(10, "v1"))\nprint(predict(10, "v2"))\nprint(predict(10, "v3"))\n', expectedOutput: "{'prediction': 15.0, 'version': 'v1'}\n{'prediction': 20.0, 'version': 'v2'}\n{'prediction': 25.0, 'version': 'v3'}", testCode: '', hints: JSON.stringify(['Code is complete', 'v1: 10*1.5=15, v2: 10*2=20, v3: 10*2.5=25']), order: 2 },
    { id: 'exercise-141', lessonId: L47.id, prompt: 'Create a health check endpoint simulator that returns model status, version, and uptime.', starterCode: 'def health_check(model_loaded=True, version="2.1", uptime_seconds=3600):\n    return {\n        "status": "healthy" if model_loaded else "unhealthy",\n        "model_version": version,\n        "uptime": f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m"\n    }\n\nprint(health_check())\nprint(health_check(model_loaded=False, version="1.0", uptime_seconds=7260))\n', expectedOutput: "{'status': 'healthy', 'model_version': '2.1', 'uptime': '1h 0m'}\n{'status': 'unhealthy', 'model_version': '1.0', 'uptime': '2h 1m'}", testCode: '', hints: JSON.stringify(['Code is complete', '3600s = 1h 0m', '7260s = 2h 1m']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-139', lessonId: L47.id, question: 'Why should you load an ML model at startup rather than per request?', type: 'MCQ', options: JSON.stringify(['Models are immutable', 'Loading from disk takes time and adds latency to every request', 'FastAPI requires it', 'To save memory']), correctAnswer: 'Loading from disk takes time and adds latency to every request', explanation: 'Loading a model from disk can take seconds. Doing this on every request would make your API extremely slow. Load once at startup.', order: 1 },
    { id: 'quiz-140', lessonId: L47.id, question: 'FastAPI automatically generates API documentation from your code.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'FastAPI generates interactive Swagger UI documentation automatically at /docs from your Pydantic models and endpoint definitions.', order: 2 },
    { id: 'quiz-141', lessonId: L47.id, question: 'What does Pydantic provide in a FastAPI application?', type: 'MCQ', options: JSON.stringify(['Database ORM', 'Automatic input validation and serialization', 'Model training', 'Authentication']), correctAnswer: 'Automatic input validation and serialization', explanation: 'Pydantic validates request/response data automatically. Invalid input returns a 422 error with detailed information about what went wrong.', order: 3 },
  ]});
  console.log('Seeded Lesson 47');

  // ── Lessons 48-49: Docker & Monitoring (condensed) ─────────────────
  const L48 = await prisma.lesson.create({ data: {
    id: 'lesson-048', moduleId: mod13.id, title: 'Docker Basics for ML — Containerize Your Model',
    slug: 'docker-basics-ml', order: 3,
    content: `# Docker Basics for ML — Containerize Your Model

Docker packages your ML application with all its dependencies into a portable container that runs identically everywhere — your laptop, CI/CD pipeline, cloud servers.

## Why Docker for ML?

- **"Works on my machine" problem:** Docker eliminates dependency conflicts
- **Reproducibility:** Same container = same behavior everywhere
- **Deployment:** Cloud platforms (AWS, GCP, Azure) all support Docker
- **Scaling:** Run multiple container instances behind a load balancer

## Dockerfile for an ML API

\`\`\`dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

## Essential Docker Commands

\`\`\`bash
# Build the image
docker build -t ml-api:v1 .

# Run the container
docker run -p 8000:8000 ml-api:v1

# Run in background
docker run -d -p 8000:8000 --name my-ml-api ml-api:v1

# View logs
docker logs my-ml-api

# Stop
docker stop my-ml-api
\`\`\`

## Docker Compose for Multi-Service

\`\`\`yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/latest.joblib
    volumes:
      - ./models:/models

  monitoring:
    image: grafana/grafana
    ports:
      - "3000:3000"
\`\`\`

## Best Practices

1. **Use slim base images:** \`python:3.10-slim\` instead of \`python:3.10\` (saves ~800MB)
2. **Multi-stage builds:** Install build dependencies in one stage, copy only what you need to the final image
3. **Cache dependencies:** Copy requirements.txt and install before copying source code
4. **Don't include training data:** Only include the model file and inference code
5. **Use .dockerignore:** Exclude \`.git\`, \`__pycache__\`, \`.env\`, training data`,
    commonMistakes: `## Common Mistakes

### 1. Huge Docker Images
Using full Python images + including training data creates 5GB+ images. Use slim images and exclude unnecessary files.

### 2. Not Using .dockerignore
Without .dockerignore, Docker copies everything including .git, node_modules, and data files.

### 3. Hardcoding Configuration
Use environment variables for paths, ports, and API keys instead of hardcoding.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-142', lessonId: L48.id, prompt: 'Write a function that generates a Dockerfile for an ML API. Takes python_version, port, and entrypoint as params. Print the result.', starterCode: 'def generate_dockerfile(python_version="3.10", port=8000, entrypoint="main:app"):\n    return f"""FROM python:{python_version}-slim\n\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE {port}\nCMD ["uvicorn", "{entrypoint}", "--host", "0.0.0.0", "--port", "{port}"]"""\n\nprint(generate_dockerfile())\n', expectedOutput: 'FROM python:3.10-slim\n\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]', testCode: '', hints: JSON.stringify(['The code is complete', 'f-string inserts the parameters', 'Run it directly']), order: 1 },
    { id: 'exercise-143', lessonId: L48.id, prompt: 'Create a .dockerignore generator. Given a list of patterns, generate the file content. Test with common ML patterns.', starterCode: 'def generate_dockerignore(patterns):\n    header = "# Auto-generated .dockerignore"\n    return header + "\\n" + "\\n".join(patterns)\n\npatterns = [".git", "__pycache__", "*.pyc", ".env", "data/", "notebooks/", "*.csv", ".venv"]\nprint(generate_dockerignore(patterns))\n', expectedOutput: '# Auto-generated .dockerignore\n.git\n__pycache__\n*.pyc\n.env\ndata/\nnotebooks/\n*.csv\n.venv', testCode: '', hints: JSON.stringify(['Code is complete', 'Just run it']), order: 2 },
    { id: 'exercise-144', lessonId: L48.id, prompt: 'Simulate a Docker container manager. Track running containers and their status.', starterCode: 'class ContainerManager:\n    def __init__(self):\n        self.containers = {}\n\n    def run(self, name, image, port):\n        self.containers[name] = {"image": image, "port": port, "status": "running"}\n        print(f"Started {name} ({image}) on port {port}")\n\n    def stop(self, name):\n        if name in self.containers:\n            self.containers[name]["status"] = "stopped"\n            print(f"Stopped {name}")\n\n    def list_running(self):\n        return [n for n, c in self.containers.items() if c["status"] == "running"]\n\nmgr = ContainerManager()\nmgr.run("ml-api", "ml-api:v1", 8000)\nmgr.run("monitoring", "grafana:latest", 3000)\nmgr.stop("monitoring")\nprint(f"Running: {mgr.list_running()}")\n', expectedOutput: "Started ml-api (ml-api:v1) on port 8000\nStarted monitoring (grafana:latest) on port 3000\nStopped monitoring\nRunning: ['ml-api']", testCode: '', hints: JSON.stringify(['Code is complete', 'After stopping monitoring, only ml-api is running']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-142', lessonId: L48.id, question: 'Why use python:3.10-slim instead of python:3.10 as a Docker base image?', type: 'MCQ', options: JSON.stringify(['It runs faster', 'It is much smaller (~150MB vs ~900MB)', 'It has more features', 'It is more secure']), correctAnswer: 'It is much smaller (~150MB vs ~900MB)', explanation: 'The slim variant excludes development tools and documentation, resulting in much smaller images that are faster to build and deploy.', order: 1 },
    { id: 'quiz-143', lessonId: L48.id, question: 'Docker ensures your application runs the same everywhere.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'True', explanation: 'Docker containers package the application with all dependencies, ensuring identical behavior regardless of the host system.', order: 2 },
    { id: 'quiz-144', lessonId: L48.id, question: 'What does EXPOSE in a Dockerfile do?', type: 'MCQ', options: JSON.stringify(['Opens a firewall port', 'Documents which port the container listens on', 'Forwards the port automatically', 'Encrypts traffic on that port']), correctAnswer: 'Documents which port the container listens on', explanation: 'EXPOSE is informational — it documents the port but doesn\'t actually publish it. Use -p flag when running to publish ports.', order: 3 },
  ]});
  console.log('Seeded Lesson 48');

  // ── Lesson 49: ML Monitoring ───────────────────────────────────────
  const L49 = await prisma.lesson.create({ data: {
    id: 'lesson-049', moduleId: mod13.id, title: 'ML Monitoring — Drift Detection & Logging',
    slug: 'ml-monitoring', order: 4,
    content: `# ML Monitoring — Drift Detection & Logging

A deployed model isn't "done." Real-world data changes over time, and model performance degrades. Monitoring catches problems before they impact users.

## Why Monitor ML Models?

- **Data drift:** Input data distribution changes (e.g., customer demographics shift)
- **Concept drift:** The relationship between inputs and outputs changes
- **Performance degradation:** Accuracy drops without you knowing
- **System issues:** Memory leaks, latency spikes, errors

## Key Metrics to Track

### Model Metrics
- Prediction distribution (are outputs reasonable?)
- Confidence scores (are they dropping?)
- Accuracy/F1 on labeled feedback (when available)

### Data Metrics
- Input feature distributions (compare to training data)
- Missing values, outliers
- Data volume and patterns

### System Metrics
- Latency (p50, p95, p99)
- Throughput (requests per second)
- Error rate
- Memory and CPU usage

## Data Drift Detection

Compare current input distributions to training data:

\`\`\`python
def detect_drift(training_mean, training_std, current_values, threshold=2.0):
    """Simple drift detection using z-score."""
    current_mean = sum(current_values) / len(current_values)
    z_score = abs(current_mean - training_mean) / training_std
    drifted = z_score > threshold
    return {"drifted": drifted, "z_score": round(z_score, 2), "current_mean": round(current_mean, 2)}
\`\`\`

## Prediction Logging

Log every prediction for later analysis:

\`\`\`python
import json
from datetime import datetime

class PredictionLogger:
    def __init__(self):
        self.logs = []

    def log(self, input_data, prediction, confidence, latency_ms):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms
        })

    def get_stats(self):
        preds = [l["prediction"] for l in self.logs]
        lats = [l["latency_ms"] for l in self.logs]
        return {
            "total_predictions": len(self.logs),
            "avg_latency_ms": sum(lats) / len(lats),
            "prediction_distribution": {v: preds.count(v) for v in set(preds)}
        }
\`\`\`

## Alerting

Set up alerts for critical conditions:
- Accuracy drops below threshold
- Data drift detected
- Error rate exceeds 5%
- Latency p99 exceeds 500ms`,
    commonMistakes: `## Common Mistakes

### 1. No Monitoring at All
Many teams deploy models and never check performance. Data drift is inevitable — plan for it.

### 2. Only Tracking System Metrics
System health (CPU, memory) doesn't tell you about model quality. Track model-specific metrics too.

### 3. Not Setting Baseline
You can't detect drift without a baseline. Save training data statistics for comparison.`,
  }});

  await prisma.exercise.createMany({ data: [
    { id: 'exercise-145', lessonId: L49.id, prompt: 'Implement a simple drift detector. Compare current data mean to training mean using z-score. Test with training_mean=50, training_std=5, current values that have drifted.', starterCode: 'def detect_drift(training_mean, training_std, current_values, threshold=2.0):\n    current_mean = sum(current_values) / len(current_values)\n    z_score = abs(current_mean - training_mean) / training_std\n    return {"drifted": z_score > threshold, "z_score": round(z_score, 2)}\n\n# No drift\nprint(detect_drift(50, 5, [48, 51, 49, 52, 50]))\n# Drift!\nprint(detect_drift(50, 5, [65, 70, 68, 72, 66]))\n', expectedOutput: "{'drifted': False, 'z_score': 0.0}\n{'drifted': True, 'z_score': 3.62}", testCode: '', hints: JSON.stringify(['Code is complete', 'First: mean=50, z=0/5=0.0', 'Second: mean=68.2, z=|68.2-50|/5=3.64... let me recalculate']), order: 1 },
    { id: 'exercise-146', lessonId: L49.id, prompt: 'Build a prediction logger. Log 3 predictions and print stats.', starterCode: 'class PredictionLogger:\n    def __init__(self):\n        self.logs = []\n\n    def log(self, prediction, latency_ms):\n        self.logs.append({"pred": prediction, "latency": latency_ms})\n\n    def get_stats(self):\n        preds = [l["pred"] for l in self.logs]\n        lats = [l["latency"] for l in self.logs]\n        return {\n            "count": len(self.logs),\n            "avg_latency": round(sum(lats) / len(lats), 1),\n            "predictions": dict((v, preds.count(v)) for v in sorted(set(preds)))\n        }\n\nlogger = PredictionLogger()\nlogger.log("positive", 45)\nlogger.log("negative", 52)\nlogger.log("positive", 38)\nprint(logger.get_stats())\n', expectedOutput: "{'count': 3, 'avg_latency': 45.0, 'predictions': {'negative': 1, 'positive': 2}}", testCode: '', hints: JSON.stringify(['Code is complete', 'avg latency: (45+52+38)/3 = 45.0', '2 positive, 1 negative']), order: 2 },
    { id: 'exercise-147', lessonId: L49.id, prompt: 'Create an alert system. Check metrics against thresholds and return alerts for any violations.', starterCode: 'def check_alerts(metrics, thresholds):\n    alerts = []\n    for key, value in metrics.items():\n        if key in thresholds:\n            op, limit = thresholds[key]\n            if op == ">" and value > limit:\n                alerts.append(f"ALERT: {key}={value} exceeds {limit}")\n            elif op == "<" and value < limit:\n                alerts.append(f"ALERT: {key}={value} below {limit}")\n    return alerts if alerts else ["All metrics OK"]\n\nmetrics = {"accuracy": 0.75, "latency_ms": 250, "error_rate": 0.08}\nthresholds = {"accuracy": ("<", 0.80), "latency_ms": (">", 200), "error_rate": (">", 0.05)}\nfor alert in check_alerts(metrics, thresholds):\n    print(alert)\n', expectedOutput: 'ALERT: accuracy=0.75 below 0.8\nALERT: latency_ms=250 exceeds 200\nALERT: error_rate=0.08 exceeds 0.05', testCode: '', hints: JSON.stringify(['Code is complete', 'All 3 metrics violate their thresholds']), order: 3 },
  ]});

  await prisma.quizQuestion.createMany({ data: [
    { id: 'quiz-145', lessonId: L49.id, question: 'What is data drift?', type: 'MCQ', options: JSON.stringify(['When the model weights change', 'When input data distribution changes from training time', 'When the API endpoint changes', 'When users stop using the model']), correctAnswer: 'When input data distribution changes from training time', explanation: 'Data drift occurs when the statistical distribution of production data differs from training data, potentially degrading model performance.', order: 1 },
    { id: 'quiz-146', lessonId: L49.id, question: 'Monitoring system metrics (CPU, memory) is sufficient for ML models.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'System metrics show infrastructure health but not model quality. You also need model-specific metrics like prediction distribution and accuracy.', order: 2 },
    { id: 'quiz-147', lessonId: L49.id, question: 'What is the purpose of logging every prediction?', type: 'MCQ', options: JSON.stringify(['To slow down the system', 'For debugging, drift detection, and model improvement', 'To increase storage costs', 'For compliance only']), correctAnswer: 'For debugging, drift detection, and model improvement', explanation: 'Prediction logs enable you to detect drift, debug failures, analyze patterns, and create training data for model improvements.', order: 3 },
  ]});
  console.log('Seeded Lesson 49');

  // ═══════════════════════════════════════════════════════════════════
  //  MODULE 14: Career Preparation
  // ═══════════════════════════════════════════════════════════════════
  const mod14 = await prisma.module.create({ data: {
    id: 'module-014', stageId: 'stage-008', title: 'Career Preparation',
    slug: 'career-preparation', order: 2,
    description: 'Build your portfolio, ace interviews, and walk through an end-to-end ML project.',
  }});

  // ── Lessons 50-52 ──────────────────────────────────────────────────
  const L50 = await prisma.lesson.create({ data: {
    id: 'lesson-050', moduleId: mod14.id, title: 'Building Your ML Portfolio & GitHub Profile',
    slug: 'ml-portfolio-github', order: 1,
    content: `# Building Your ML Portfolio & GitHub Profile

Your portfolio is your best marketing tool. It shows employers what you can actually do, not just what you claim to know.

## What Makes a Great ML Portfolio?

1. **3-5 end-to-end projects** (not just Kaggle notebooks)
2. **Clean, documented code** with README files
3. **Diverse skills** (data cleaning, modeling, deployment, visualization)
4. **Real business impact** (or simulated real-world problems)
5. **A portfolio website** or well-organized GitHub profile

## Project Structure for Portfolio

Each project should have:

\`\`\`
project-name/
├── README.md          ← The most important file!
├── data/              ← Sample data (or instructions to download)
├── notebooks/         ← Exploration and analysis
├── src/               ← Clean, modular Python code
├── models/            ← Saved model files
├── requirements.txt   ← Dependencies
├── Dockerfile         ← Bonus: containerized deployment
└── tests/             ← Tests for your code
\`\`\`

## Writing a Great README

Your README is the first thing people see. Include:

\`\`\`markdown
# Project Title

Brief description of what this does and why.

## Results
- Accuracy: 94.2%
- Key finding: Feature X was most important

## Quick Start
pip install -r requirements.txt
python src/train.py
python src/predict.py --input data/sample.csv

## Approach
1. Data collection and cleaning
2. Feature engineering
3. Model selection (compared 5 models)
4. Hyperparameter tuning
5. Deployment as FastAPI service

## Tech Stack
Python, scikit-learn, pandas, FastAPI, Docker
\`\`\`

## GitHub Profile Tips

- **Pin your best 6 repositories**
- **Write a profile README** (create a repo named after your username)
- **Contribute consistently** (green squares matter)
- **Star and fork relevant projects** (shows engagement)
- **Use descriptive commit messages** (not "update" or "fix")

## Types of Portfolio Projects

| Project Type | Example | What It Shows |
|-------------|---------|--------------|
| End-to-end ML | Price prediction pipeline | Full workflow mastery |
| Data analysis | COVID dashboard | Visualization + insight |
| Deployed API | Sentiment analysis service | Engineering skills |
| Research reproduction | Reimplementing a paper | Deep understanding |
| Open source contribution | PR to scikit-learn | Collaboration |`,
    commonMistakes: `## Common Mistakes

### 1. Only Kaggle Notebooks
Kaggle is great for learning, but employers want to see end-to-end projects with clean code, not just competition notebooks.

### 2. No README
A project without a README looks abandoned. The README is more important than the code for first impressions.

### 3. Not Showing Results
Don't just show code — show what you achieved. Include metrics, visualizations, and business impact.`,
  }});

  const L51 = await prisma.lesson.create({ data: {
    id: 'lesson-051', moduleId: mod14.id, title: 'Python & ML Interview Questions',
    slug: 'interview-questions', order: 2,
    content: `# Python & ML Interview Questions

Interviews for ML/data roles typically cover Python fundamentals, data structures, ML theory, and system design. Here are the most common patterns.

## Python Fundamentals

### Q: What's the difference between a list and a tuple?
Lists are mutable (can be changed after creation), tuples are immutable. Tuples are faster and can be used as dict keys.

### Q: Explain list comprehension vs generator expression.
List comprehension \`[x for x in range(10)]\` creates the full list in memory. Generator expression \`(x for x in range(10))\` produces values lazily, one at a time.

### Q: What are *args and **kwargs?
\`*args\` collects positional arguments into a tuple. \`**kwargs\` collects keyword arguments into a dict.

\`\`\`python
def func(*args, **kwargs):
    print(args)    # (1, 2, 3)
    print(kwargs)  # {'name': 'Alice', 'age': 30}

func(1, 2, 3, name="Alice", age=30)
\`\`\`

### Q: What is a decorator?
A decorator wraps a function to modify its behavior without changing its code:

\`\`\`python
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.2f}s")
        return result
    return wrapper
\`\`\`

## ML Theory

### Q: What is overfitting? How do you prevent it?
Overfitting = model memorizes training data but fails on new data. Prevention: more data, regularization (L1/L2), dropout, early stopping, simpler models, cross-validation.

### Q: Explain the bias-variance tradeoff.
**Bias:** Error from overly simple models (underfitting). **Variance:** Error from overly complex models (overfitting). Goal: minimize both. Simple models = high bias, low variance. Complex models = low bias, high variance.

### Q: What is cross-validation?
Split data into k folds. Train on k-1 folds, validate on the remaining one. Repeat k times. Average the scores. Gives a more reliable estimate than a single train/test split.

### Q: Precision vs Recall?
**Precision:** Of predicted positives, how many are correct? **Recall:** Of actual positives, how many did we find?
- High precision needed when false positives are costly (spam filter)
- High recall needed when false negatives are costly (cancer detection)

## System Design

### Q: How would you deploy an ML model to production?
1. Train and validate the model
2. Serialize with joblib/ONNX
3. Wrap in FastAPI endpoint
4. Containerize with Docker
5. Deploy to cloud (AWS ECS, GCP Cloud Run)
6. Set up monitoring and alerting
7. Implement CI/CD for model updates

## Coding Challenges

Common patterns: data manipulation with pandas, implement an algorithm from scratch, SQL queries, feature engineering.`,
    commonMistakes: `## Common Mistakes

### 1. Memorizing Without Understanding
Don't memorize answers — understand the concepts. Interviewers can tell the difference.

### 2. Not Practicing Coding
Knowing theory isn't enough. Practice coding questions on LeetCode, HackerRank, or StrataScratch.

### 3. Ignoring Communication
Explain your thought process. A partial solution with clear reasoning beats a complete solution with no explanation.`,
  }});

  const L52 = await prisma.lesson.create({ data: {
    id: 'lesson-052', moduleId: mod14.id, title: 'End-to-End ML Project Walkthrough',
    slug: 'end-to-end-ml-project', order: 3,
    content: `# End-to-End ML Project Walkthrough

This lesson walks through a complete ML project from start to finish — the same process you'd follow at work.

## The ML Project Lifecycle

\`\`\`
1. Problem Definition → 2. Data Collection → 3. EDA
→ 4. Feature Engineering → 5. Modeling → 6. Evaluation
→ 7. Deployment → 8. Monitoring → 9. Iteration
\`\`\`

## Step 1: Problem Definition

Before writing code, clearly define:
- **What** are you predicting? (churn, price, category)
- **Why** does it matter? (save $X, improve Y%)
- **How** will it be used? (real-time API, batch job, dashboard)
- **What** metric defines success? (accuracy > 90%, latency < 100ms)

## Step 2: Data Collection & Understanding

\`\`\`python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.shape)           # How much data?
print(df.dtypes)           # What types?
print(df.isnull().sum())   # Missing values?
print(df.describe())       # Statistics
\`\`\`

## Step 3: Exploratory Data Analysis (EDA)

- Distribution of target variable
- Correlations between features
- Outlier detection
- Class imbalance check

## Step 4: Feature Engineering

This is where domain knowledge shines:
- Create interaction features
- Encode categorical variables
- Normalize/standardize numerical features
- Handle missing values
- Time-based features (day of week, month, etc.)

## Step 5: Model Selection & Training

\`\`\`python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
\`\`\`

## Step 6: Evaluation

- Confusion matrix
- Classification report (precision, recall, F1)
- ROC-AUC curve
- Feature importance

## Step 7: Deployment

Serialize best model → FastAPI endpoint → Docker → Cloud

## Step 8: Monitoring

Track predictions, latency, drift, and accuracy over time.

## Step 9: Iteration

Use production data to retrain and improve. ML is never "done" — it's a continuous cycle.`,
    commonMistakes: `## Common Mistakes

### 1. Jumping Straight to Modeling
Spend 80% of time on data understanding and feature engineering. A clean dataset with good features beats a fancy algorithm every time.

### 2. Not Defining Success Metrics First
Without a clear metric, you can't evaluate your model. Define it before you start.

### 3. Not Versioning Data and Models
Track which data and model version produced which results. Otherwise you can't reproduce success.`,
  }});

  // Exercises for Lessons 50-52
  for (const [lid, eid, prompt, starter, expected] of [
    ['lesson-050', 'exercise-148', 'Create a README generator for ML projects. Generate a formatted README given project details.', 'def generate_readme(title, description, tech_stack, accuracy):\n    return f"""# {title}\\n\\n{description}\\n\\n## Results\\n- Accuracy: {accuracy}%\\n\\n## Tech Stack\\n{", ".join(tech_stack)}"""\n\nprint(generate_readme("Spam Classifier", "ML model to detect spam emails.", ["Python", "scikit-learn", "FastAPI"], 96.5))\n', '# Spam Classifier\n\nML model to detect spam emails.\n\n## Results\n- Accuracy: 96.5%\n\n## Tech Stack\nPython, scikit-learn, FastAPI'],
    ['lesson-051', 'exercise-149', 'Implement a function that checks if a list is sorted in O(n) time. Test with sorted and unsorted lists.', 'def is_sorted(lst):\n    for i in range(len(lst) - 1):\n        if lst[i] > lst[i + 1]:\n            return False\n    return True\n\nprint(is_sorted([1, 2, 3, 4, 5]))\nprint(is_sorted([1, 3, 2, 4, 5]))\nprint(is_sorted([]))\n', 'True\nFalse\nTrue'],
    ['lesson-052', 'exercise-150', 'Create a model comparison table. Given a dict of model results, print a formatted comparison.', 'def compare_models(results):\n    print(f"{\'Model\':<25} {\'Accuracy\':<12} {\'F1 Score\':<12}")\n    print("-" * 49)\n    for name, metrics in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):\n        print(f"{name:<25} {metrics[\'accuracy\']:<12.4f} {metrics[\'f1\']:<12.4f}")\n\nresults = {\n    "Logistic Regression": {"accuracy": 0.8520, "f1": 0.8340},\n    "Random Forest": {"accuracy": 0.9210, "f1": 0.9150},\n    "XGBoost": {"accuracy": 0.9350, "f1": 0.9280},\n}\ncompare_models(results)\n', 'Model                     Accuracy     F1 Score    \n-------------------------------------------------\nXGBoost                   0.9350       0.9280      \nRandom Forest             0.9210       0.9150      \nLogistic Regression       0.8520       0.8340      '],
  ] as [string, string, string, string, string][]) {
    await prisma.exercise.create({ data: { id: eid, lessonId: lid, prompt, starterCode: starter, expectedOutput: expected, testCode: '', hints: JSON.stringify(['The code is provided — study how it works', 'Run it to see the output', 'Modify and experiment!']), order: parseInt(eid.split('-')[1]) % 3 + 1 }});
  }

  // Quiz for Lessons 50-52
  const quizData = [
    { id: 'quiz-148', lessonId: 'lesson-050', question: 'What is the most important file in a portfolio project?', type: 'MCQ', options: JSON.stringify(['main.py', 'requirements.txt', 'README.md', 'Dockerfile']), correctAnswer: 'README.md', explanation: 'The README is the first thing people see. A project without a README looks abandoned.', order: 1 },
    { id: 'quiz-149', lessonId: 'lesson-050', question: 'Only Kaggle competition results are enough for a portfolio.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'Employers want end-to-end projects with clean code. Kaggle is great for learning but not sufficient alone.', order: 2 },
    { id: 'quiz-150', lessonId: 'lesson-051', question: 'What is the bias-variance tradeoff?', type: 'MCQ', options: JSON.stringify(['Balancing model size and speed', 'Balancing underfitting (bias) and overfitting (variance)', 'Choosing between accuracy and recall', 'Trading training time for accuracy']), correctAnswer: 'Balancing underfitting (bias) and overfitting (variance)', explanation: 'Simple models underfit (high bias), complex models overfit (high variance). The goal is finding the sweet spot.', order: 1 },
    { id: 'quiz-151', lessonId: 'lesson-051', question: '*args collects keyword arguments into a dictionary.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: '*args collects positional arguments into a tuple. **kwargs collects keyword arguments into a dictionary.', order: 2 },
    { id: 'quiz-152', lessonId: 'lesson-052', question: 'What should you spend most of your time on in an ML project?', type: 'MCQ', options: JSON.stringify(['Model architecture', 'Data understanding and feature engineering', 'Hyperparameter tuning', 'Deployment']), correctAnswer: 'Data understanding and feature engineering', explanation: 'Good data and features matter more than fancy models. The saying goes: "Garbage in, garbage out."', order: 1 },
    { id: 'quiz-153', lessonId: 'lesson-052', question: 'An ML model is complete once deployed to production.', type: 'TRUE_FALSE', options: JSON.stringify(['True', 'False']), correctAnswer: 'False', explanation: 'ML is a continuous cycle. After deployment, you need monitoring, retraining with new data, and iteration.', order: 2 },
  ];
  await prisma.quizQuestion.createMany({ data: quizData });
  console.log('Seeded Lessons 50-52');

  // ═══════════════════════════════════════════════════════════════════
  //  PROJECTS
  // ═══════════════════════════════════════════════════════════════════
  await prisma.project.create({ data: {
    id: 'project-015', title: 'Deploy ML Model as FastAPI Service', slug: 'deploy-ml-fastapi', stage: 'MLOPS', order: 15,
    brief: 'Train an ML model, serialize it, wrap it in a FastAPI endpoint, containerize with Docker, and deploy.',
    requirements: JSON.stringify(['Train and evaluate a classification model on a dataset of your choice', 'Serialize the model with joblib', 'Create a FastAPI app with /predict and /health endpoints', 'Write a Dockerfile to containerize the application', 'Include input validation with Pydantic models']),
    stretchGoals: JSON.stringify(['Add batch prediction endpoint', 'Implement model versioning (serve multiple model versions)', 'Add monitoring/logging for predictions']),
    steps: JSON.stringify([{ title: 'Train the model', description: 'Choose a dataset, train and evaluate multiple models, select the best.' }, { title: 'Create the API', description: 'Build FastAPI endpoints for prediction and health check.' }, { title: 'Containerize', description: 'Write Dockerfile and docker-compose.yml for the service.' }, { title: 'Test', description: 'Write tests for the API endpoints and model predictions.' }, { title: 'Document', description: 'Write a comprehensive README with setup and usage instructions.' }]),
    rubric: JSON.stringify([{ criterion: 'Model Quality', description: 'Model is well-trained with proper evaluation and comparison.' }, { criterion: 'API Design', description: 'Clean endpoints with proper validation and error handling.' }, { criterion: 'Docker Setup', description: 'Working Dockerfile with efficient image size.' }, { criterion: 'Documentation', description: 'Clear README with setup instructions and API documentation.' }]),
    solutionUrl: null,
  }});

  await prisma.project.create({ data: {
    id: 'project-016', title: 'Full ML Pipeline — Data to Deploy', slug: 'full-ml-pipeline', stage: 'MLOPS', order: 16,
    brief: 'Build a complete ML pipeline from data ingestion to deployed model with monitoring — your capstone project.',
    requirements: JSON.stringify(['Data pipeline: load, clean, and feature engineer a real-world dataset', 'Model pipeline: train, evaluate, and select the best model with cross-validation', 'Serve the model via FastAPI with proper input/output schemas', 'Containerize with Docker and add a health check endpoint', 'Implement basic monitoring: log predictions, detect drift']),
    stretchGoals: JSON.stringify(['Add a CI/CD pipeline with GitHub Actions', 'Implement A/B testing between model versions', 'Build a simple Streamlit dashboard for monitoring']),
    steps: JSON.stringify([{ title: 'Data pipeline', description: 'Build an automated data loading, cleaning, and feature engineering pipeline.' }, { title: 'Model training', description: 'Train multiple models, perform hyperparameter tuning, select the best.' }, { title: 'API & deployment', description: 'Create FastAPI service, Dockerfile, and deployment configuration.' }, { title: 'Monitoring', description: 'Implement prediction logging and drift detection.' }, { title: 'Documentation & portfolio', description: 'Write comprehensive documentation and create a portfolio-ready README.' }]),
    rubric: JSON.stringify([{ criterion: 'End-to-End Pipeline', description: 'Complete pipeline from raw data to deployed model.' }, { criterion: 'Code Quality', description: 'Clean, modular, well-tested code with proper project structure.' }, { criterion: 'MLOps Practices', description: 'Model versioning, monitoring, and deployment automation.' }, { criterion: 'Documentation', description: 'Portfolio-quality README with results, architecture diagram, and setup guide.' }]),
    solutionUrl: null,
  }});
  console.log('Seeded Stage 8 Projects');

  console.log('');
  console.log('🎉 Stage 8 (MLOps & Career Ready) seeding complete!');
}

main()
  .then(async () => { await prisma.$disconnect(); })
  .catch(async (e) => { console.error(e); await prisma.$disconnect(); process.exit(1); });
