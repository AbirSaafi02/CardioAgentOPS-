# CardioAgent-Ops

Project scaffold for the six-service CardioAgent stack.

```text
.
|-- docker-compose.yml
|-- README.md
|-- api-gateway/
|   |-- Dockerfile
|   |-- main.py
|   `-- requirements.txt
|-- audio-engine/
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- app/
|   |   |-- main.py
|   |   `-- preprocessing.py
|   `-- models/
|       `-- README.md
|-- agent-orchestrator/
|-- vector-db/
|-- llm-server/
`-- mlflow-server/
```

Place the trained `cardio_model.onnx` artifact in `audio-engine/models/`.
