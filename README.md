# Qwen2.5-Coder HumanEval Evaluation

This repository contains scripts and setup instructions for serving the **Qwen/Qwen2.5-Coder-0.5B-Instruct** model with [vLLM](https://github.com/vllm-project/vllm), running inference on the [HumanEval](https://github.com/openai/human-eval) benchmark, and evaluating performance.

---

## 📂 Repository Structure

- `docker/` – Scripts to build and run Docker containers for serving the model.
- `notebooks/` – Jupyter notebooks for inference and evaluation experiments.
- `scripts/` – Python scripts for batch inference and metric calculation.
- `REPORT.md` – Full analysis, results, and prompt/decode sweeps.

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/kartikmehta15/qwen-humaneval.git
cd qwen-humaneval
```

### 2. Build & Run vLLM Docker
```bash
# Build Docker image
docker build -t vllm-qwen .

# Run container exposing API on port 8001
docker run -d --gpus all \
  -p 8001:8000 \
  --name vllm-instruct \
  vllm-qwen \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.5
```

### 3. Check Logs
```bash
docker logs -f vllm-instruct
```

### 4. Stop & Delete Container
```bash
docker stop vllm-instruct
docker rm vllm-instruct
```

---

## 📊 Evaluation

👉 Detailed results, prompt/decoding sweeps, and observations are documented in [`REPORT.md`](./REPORT.md).

---

## 🛠️ Requirements

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/)
- GPU with CUDA support (tested on A100)

---

## 📌 Notes

- Experiments were primarily run with **vLLM 0.10.1.1**.  
- Results may vary slightly depending on GPU type and CUDA version.  
- For compute-intensive runs, techniques like *self-consistency* and *self-refinement* can be explored to improve base model performance.

---

## 👤 Author

Kartik Mehta
