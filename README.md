# 🤖 LLM Evaluator Tool

> Automated LLM Selection & Evaluation — find the best model for *your* task in minutes.

<div align="center">

[![Made by NEO](https://img.shields.io/badge/Made%20by-NEO-6C63FF?style=for-the-badge&logo=sparkles&logoColor=white)](https://heyneo.com)
[![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-FF6B6B?style=for-the-badge)](https://openrouter.ai)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**[Built by NEO](https://heyneo.com)** - Autonomous AI Engineering Agent

</div>

---

## 🎬 Demo

![Demo](data/llm-evaluator-runtime.gif)

> *Watch the LLM Evaluator Tool in action — from task input to ranked results in minutes.*

---

## ✨ What It Does

LLM Evaluator Tool automates the process of selecting and benchmarking the best LLMs for any task you define. It uses **Gemini 3.1 Pro** (via OpenRouter) as a Judge LLM to fairly evaluate candidate models across multiple dimensions.

### Core Workflow

```
Your Task Description
        │
        ▼
┌───────────────────┐
│  1. Test Suite    │  Judge LLM generates tailored test cases
│     Generation    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. Model         │  Discovers top LLMs for your task category
│     Discovery     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. Benchmarking  │  Runs all tests across all candidate models
│     Execution     │  (captures responses + latency)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. Evaluation    │  Judge scores on accuracy, hallucination,
│     (Judge LLM)   │  grounding, tool-calling, clarity
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  5. Ranking &     │  Top 3 models + latency stats +
│     Prompt Opt.   │  optimized system prompt
└───────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gauravvij/llm-evaluator.git
cd llm-evaluator
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Your OpenRouter API Key

The tool requires an [OpenRouter](https://openrouter.ai) API key. **Never hardcode your key** — use one of the two methods below:

#### Option A — Environment Variable (Recommended)

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

Add this to your `~/.bashrc` or `~/.zshrc` to persist across sessions.

#### Option B — `.env` File

Copy the example file and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

> 🔑 **Get your free API key at [openrouter.ai/keys](https://openrouter.ai/keys)**

---

## 🖥️ Usage

### Interactive Mode

```bash
python main.py
```

You'll be prompted to enter your task description.

### CLI Mode

```bash
# Evaluate LLMs for a coding task
python main.py --task "Python software engineering assistant"

# Math tutoring with 3 test cases
python main.py --task "Math tutoring for high school students" --num-tests 3

# Customer support with 4 candidates, no report saved
python main.py --task "Customer support chatbot" --max-candidates 4 --no-save

# Custom output directory
python main.py --task "Creative writing assistant" --output-dir ./results
```

### All CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--task` | `-t` | *(prompted)* | Natural language task description |
| `--num-tests` | `-n` | `5` | Number of test cases to generate |
| `--max-candidates` | `-c` | `6` | Max candidate models to evaluate |
| `--output-dir` | `-o` | `./analysis` | Directory to save JSON report |
| `--no-save` | — | `False` | Skip saving the JSON report |

---

## 📊 Sample Output

```
╔══════════════════════════════════════════════════════╗
║        LLM Evaluator Tool — Evaluation Report        ║
╚══════════════════════════════════════════════════════╝

Task: Python software engineering assistant

── Step 1/5 — Generating Test Suite ──────────────────
✓ Generated 5 test cases

── Step 2/5 — Discovering Candidate Models ───────────
✓ Found 6 candidate models

── Step 3/5 — Running Benchmark ──────────────────────
✓ 30 responses collected

── Step 4/5 — Evaluating Responses ───────────────────
✓ Judge scored all responses

── Step 5/5 — Ranking & Prompt Optimization ──────────

🏆 Top 3 Models for Your Task:

  #1  google/gemini-2.5-pro          Score: 92.4  Latency: 1.8s avg
  #2  openai/gpt-4.1                 Score: 89.1  Latency: 2.1s avg
  #3  anthropic/claude-sonnet-4-5    Score: 87.6  Latency: 1.5s avg

📝 Optimized System Prompt (for gemini-2.5-pro):
  "You are an expert Python software engineer..."
```

---

## 🏗️ Project Structure

```
llm-evaluator/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
└── src/
    ├── config.py            # Configuration & API key loading
    ├── openrouter_client.py # OpenRouter API client
    ├── suite_generator.py   # Test suite generation (Judge LLM)
    ├── model_discovery.py   # Candidate model discovery
    ├── benchmarker.py       # Parallel benchmarking engine
    ├── evaluator.py         # Multi-dimensional evaluation
    ├── prompt_optimizer.py  # Optimized prompt generation
    └── reporter.py          # Rich CLI output & JSON reports
```

---

## 🔒 Security

- **No API keys are ever hardcoded** in this codebase.
- Keys are loaded exclusively from the `OPENROUTER_API_KEY` environment variable or a local `.env` file.
- The `.gitignore` excludes `.env` files and any local config containing secrets.
- **Never commit your `.env` file** or share your API key publicly.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License

---

<div align="center">

### Built with ❤️ by

[![Made by NEO](https://img.shields.io/badge/Made%20by-NEO-6C63FF?style=for-the-badge&logo=sparkles&logoColor=white)](https://heyneo.com)

</div>
