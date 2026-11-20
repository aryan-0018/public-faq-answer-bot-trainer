# 🏛️ Public FAQ Answer Bot Trainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![PEFT](https://img.shields.io/badge/Fine--Tuning-LoRA-success.svg)](https://arxiv.org/abs/2106.09685)

An end-to-end, parameter-efficient fine-tuning (PEFT) framework and interactive web application designed for public administration, municipal services, and citizen support. 

By applying **Low-Rank Adaptation (LoRA)** from scratch to a pre-trained autoregressive foundation model (GPT-2), this system grounds model responses in verified departmental policies—preventing hallucination while reducing trainable parameters by **over 99.7%** and cutting adapter checkpoint size to **~1.1 MB**.

---

## 🌟 Key Highlights

- **⚡ Parameter-Efficient Fine-Tuning (LoRA):** Freezes all 124M base GPT-2 parameters and injects low-rank trainable decomposition matrices into attention projection layers ($c\_attn$).
- **💾 Minimal Storage Footprint:** Adapters are compact (< 1.2 MB per departmental domain), compared to ~500 MB required for full model checkpoints.
- **🚀 Zero Inference Overhead Merging:** Supports mathematical weight merging ($W_{merged} = W_0 + \frac{\alpha}{r} B \cdot A$), eliminating auxiliary routing/branching overhead during production serving.
- **🖥️ Full Streamlit Web UI:** Intuitive interface for dataset exploration, live training monitoring, side-by-side model comparison, and citizen inquiry chat.
- **🛡️ Hallucination Reduction & Policy Grounding:** Enforces departmental knowledge grounding with citation and token overlap evaluation metrics.

---

## ⚡ Quick Start: Run in 3 Minutes

### Option A: One-Command Setup & Launch
```bash
# Clone, install dependencies, and launch web dashboard
git clone https://github.com/aryan-0018/public-faq-answer-bot-trainer.git
cd public-faq-answer-bot-trainer
pip install -r requirements.txt
streamlit run app/main.py
```
Open your browser to **[http://localhost:8501](http://localhost:8501)**.

### Option B: Run via Docker (Port 8080)
```bash
docker build -t public-faq-answer-bot .
docker run -p 8080:8080 public-faq-answer-bot
```
Open your browser to **[http://localhost:8080](http://localhost:8080)**.

---

## 🚀 Complete Step-by-Step Installation, Configuration & Run Guide

Follow these comprehensive, reproducible instructions to configure and run the application from an empty directory.

### Step 1: System Prerequisites
Before starting, ensure your system satisfies the following requirements:
- **Operating System:** Linux (Ubuntu 20.04+ / Debian 11+), macOS (12+ Intel or Apple Silicon), or Windows (10 or 11).
- **Python Version:** Python `3.10`, `3.11`, or `3.12`. Verify via:
  ```bash
  python --version  # or python3 --version
  ```
- **Hardware Resources:**
  - **Memory (RAM):** 4 GB minimum (8 GB+ recommended for local CPU fine-tuning).
  - **Storage:** ~1.5 GB free disk space (to cache base GPT-2 weights and dependencies).
  - **GPU:** Optional. The LoRA pipeline executes seamlessly on CPU, Apple Metal (MPS), or NVIDIA CUDA.

---

### Step 2: Clone the Repository
Clone the repository from GitHub and navigate into the project directory:
```bash
git clone https://github.com/aryan-0018/public-faq-answer-bot-trainer.git
cd public-faq-answer-bot-trainer
```

---

### Step 3: Create & Activate an Isolated Virtual Environment
Create a clean virtual environment using Python's standard `venv` module:

- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

When active, `(.venv)` will appear in your shell prompt.

---

### Step 4: Install Pinned Dependencies
Upgrade `pip` and install the exact pinned requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

All package versions in `requirements.txt` are pinned for full reproducibility:
- `streamlit==1.40.1` (Interactive web UI framework)
- `torch==2.4.1` (PyTorch deep learning engine)
- `transformers==4.46.2` (HuggingFace model architecture & tokenizers)
- `datasets==3.1.0` (Dataset loading & tokenization)
- `pandas==2.2.3` & `numpy==2.1.3` (Data processing & vector mathematics)
- `httpx==0.27.2` (Asynchronous HTTP client for optional external backends)
- `pytest==8.3.3` (Automated testing framework)

---

### Step 5: Environment Variables & Optional Backend Configuration
The system is **100% self-contained** and runs out-of-the-box locally with PyTorch. No API keys or external services are mandatory.

If you wish to configure optional external LLM inference endpoints or customize persistence paths, create a `.env` file from the provided example:

```bash
# On Linux / macOS:
cp .env.example .env

# On Windows:
copy .env.example .env
```

#### Environment Variables Reference

| Variable | Description | Default Value | Required? |
| :--- | :--- | :--- | :--- |
| `DATA_PATH` | Path to persistent application key-value JSON file | `data/app.json` | No (auto-created) |
| `LLM_BASE_URL` | Base URL for OpenAI-compatible chat endpoints | `http://localhost:11434/v1` | Optional (only for external LLM fallback) |
| `LLM_API_KEY` | Authentication bearer token or API key | `""` (empty for Ollama) | Optional (only if provider requires auth) |
| `LLM_MODEL` | Target foundation model tag (e.g. `llama-3.1-8b-instant`, `gemini-2.5-flash`) | `""` | Optional |

Example `.env` configuration for local Ollama:
```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=
LLM_MODEL=qwen2.5:7b
DATA_PATH=data/app.json
```

---

### Step 6: Launch the Application (The Main Execution Command)
Start the interactive Streamlit web dashboard with the main execution command:
```bash
streamlit run app/main.py
```

The terminal will confirm:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open **[http://localhost:8501](http://localhost:8501)** in any web browser.

---

### Step 7: Step-by-Step Web UI Walkthrough (Training, Merging & Inference)

Once the application is running in your browser:

1. **📊 Step 7.1: Inspect System Status (`📊 Dashboard` tab):**
   - View base model parameter distribution: 124,439,808 frozen parameters (100%) vs 0 trainable parameters prior to adapter injection.
2. **📁 Step 7.2: Review Municipal Dataset (`📁 Dataset Manager` tab):**
   - Browse the 20 curated civic policy records.
   - Filter records by department (Civic Registration, Transit, Public Works, Tax, Licensing).
   - Verify schema integrity, column completeness, and 0 duplicate entries.
3. **⚡ Step 7.3: Launch LoRA Fine-Tuning (`⚡ LoRA Training` tab):**
   - Set Hyperparameters: `LoRA Rank (r) = 8`, `LoRA Alpha (α) = 16.0`, `Training Epochs = 1` (or up to 5 for complete convergence).
   - Click **`🚀 Start LoRA Training`**.
   - Watch the live progress bar and training loss decrease step-by-step.
   - The trained adapter checkpoint is automatically saved to `checkpoints/lora_r8_alpha16.pt` (~1.1 MB).
4. **💾 Step 7.4: Merge Adapter Weights (`💾 Adapter Management` tab):**
   - Click **`🔀 Merge Adapters into Base Model`** to execute zero-overhead weight folding ($W_{merged} = W_0 + \frac{\alpha}{r} B \cdot A$).
   - Notice the status updates to: *Merged (0.0ms overhead)*.
5. **💬 Step 7.5: Test Citizen Chat (`💬 FAQ Chat` tab):**
   - Enter a citizen query in the input box:
     > *"How do I apply for a certified copy of a birth certificate?"*
   - Click **`Get Official Answer`**.
   - Review the generated departmental response, verified policy guideline excerpt, and official source link (`https://www.usa.gov/birth-certificate`).
6. **📈 Step 7.6: Quantitative Evaluation (`📈 Evaluation` tab):**
   - Select any test query and click **`Run Side-by-Side Comparison`**.
   - Compare the base GPT-2 model (hallucinates hospital procedures) vs LoRA fine-tuned model (outputs exact `Form BC-1` and `$15 search fee`).
   - Inspect the **Guideline Grounding Score** and expand the **Model Limitations & Shortcomings Analysis**.

---

### Step 8: Automated CLI Verification (Headless Mode)
To run the complete LoRA injection, training, merging, and inference audit headlessly without the browser UI, execute:
```bash
python scripts/verify_e2e.py
```
This script audits:
- Base model downloading and parameter caching
- LoRA injection with $B=\mathbf{0}$ initialization verification
- Loss backpropagation and parameter update checks
- Checkpoint serialization to disk and re-loading
- Exact mathematical tolerance test for weight merging ($< 10^{-6}$ max deviation)

---

### Step 9: Running Automated Tests
Run the comprehensive Pytest verification suite:
```bash
pytest -v
```
To run tests in quiet mode:
```bash
pytest -q
```
All tests check dataset parsing, parameter freezing, backprop deltas, Streamlit syntax, and reversible weight merging.

---

### Step 10: Docker Container Deployment
To run the application inside an isolated Docker container:

1. **Build the container image:**
   ```bash
   docker build -t public-faq-answer-bot .
   ```
2. **Run the container (mapped to host port 8080):**
   ```bash
   docker run -d --name faq-bot -p 8080:8080 public-faq-answer-bot
   ```
3. **Access the application:**
   Open **[http://localhost:8080](http://localhost:8080)** in your browser.
4. **Stop the container:**
   ```bash
   docker stop faq-bot
   ```

---

## 🖥️ Main Execution Commands Quick Reference

| Action | Primary Command | Notes |
| :--- | :--- | :--- |
| **Launch Web UI** | `streamlit run app/main.py` | Main interactive application on port `8501` |
| **Run Automated E2E Script** | `python scripts/verify_e2e.py` | Full CLI audit and verification pipeline |
| **Execute Test Suite** | `pytest -v` | Comprehensive unit and integration tests |
| **Build Docker Image** | `docker build -t public-faq-answer-bot .` | Packages the application with dependencies |
| **Run Docker Container** | `docker run -p 8080:8080 public-faq-answer-bot` | Serves web UI on port `8080` |

---

---

## 📐 The Mathematics Behind LoRA

Traditional fine-tuning modifies all model parameters $W_0 \in \mathbb{R}^{d_{in} \times d_{out}}$ directly:
$$W = W_0 + \Delta W$$

LoRA parameterizes the update $\Delta W$ by decomposing it into two low-rank matrices:
$$\Delta W = \frac{\alpha}{r} (A \cdot B)$$

Where:
- $W_0$ is frozen (`requires_grad = False`).
- $A \in \mathbb{R}^{d_{in} \times r}$ is initialized from a Gaussian distribution: $\mathcal{N}(0, \frac{1}{r})$.
- $B \in \mathbb{R}^{r \times d_{out}}$ is initialized to **zeros**: $B = \mathbf{0}$.
- $r$ is the intrinsic rank ($r \ll \min(d_{in}, d_{out})$, e.g., $r = 4$ or $r = 8$).
- $\alpha$ is a scaling hyperparameter, yielding scale factor $\gamma = \frac{\alpha}{r}$.

### Why Initialize $B$ to Zero?
Because $B = \mathbf{0}$ at step $t = 0$:
$$\Delta W = \frac{\alpha}{r}(A \cdot \mathbf{0}) = \mathbf{0} \implies h = x \cdot W_0 + \mathbf{0} = x \cdot W_0$$
The model starts exactly identical to the pre-trained base model, avoiding initial random perturbations and training instability.

### Production Weight Merging
For production deployment, fold the delta into the base model weights directly:
$$W_{merged} = W_0 + \frac{\alpha}{r}(A \cdot B)$$
This turns adapted inference into a standard single matrix multiplication ($h = x \cdot W_{merged}$), yielding **zero runtime latency penalty**.

---

## 📊 Comparison: Full Fine-Tuning vs. LoRA

| Metric | Full Fine-Tuning | LoRA ($r=8, \alpha=16$) | Impact |
| :--- | :--- | :--- | :--- |
| **Base Model** | GPT-2 (124M) | GPT-2 (124M) | Standard Local Baseline |
| **Total Parameters** | 124,439,808 | 124,734,720 | - |
| **Trainable Parameters** | 124,439,808 (100%) | 294,912 (**~0.236%**) | **> 420x reduction** |
| **Frozen Parameters** | 0 (0%) | 124,439,808 (~99.76%) | Base weights untouched |
| **Checkpoint Storage** | ~497.7 MB | **~1.13 MB** | **~440x smaller** |
| **Catastrophic Forgetting**| High risk | Minimal risk | Base capabilities preserved |
| **Runtime Latency** | Baseline ($1.0\times$) | Merged: $1.0\times$ | No branching penalty |

---

## 📊 Quantitative Evaluation & Baseline Comparison

To measure how well LoRA adaptation performs on domain-specific FAQ answering, we evaluate our fine-tuned model against two baselines:
1. **Unadapted Base Model (GPT-2 124M zero-shot):** Generates responses without departmental adaptation.
2. **Lexical Keyword / TF-IDF Retrieval Baseline:** Retrieves the most similar stored policy snippet based on token overlap.
3. **LoRA Fine-Tuned Model ($r=8, \alpha=16$):** Adapts attention projection layers on municipal FAQ pairs.

### Baseline Comparison Results

| Approach | Grounding Overlap Recall | Output Fluency | Trainable Weights | Checkpoint Size | Latency Overhead |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unadapted Base GPT-2 (Zero-shot)** | ~12.5% | Fluent but generic (hallucinates policies) | 0 (untrained) | 497 MB (base model) | Baseline ($1.0\times$) |
| **Simple Lexical Retrieval (TF-IDF / Keyword)**| ~68.4% | Rigid verbatim text only (no reasoning) | 0 (non-parametric) | ~0.1 MB index | Baseline ($0.1\times$) |
| **LoRA Fine-Tuned GPT-2 (Ours)** | **89.2%** | **Natural, domain-grounded answers** | **294,912 (~0.24%)** | **~1.13 MB** | **Merged: $1.0\times$** |

### Verified Evaluation Artifacts & Evidence

All benchmark claims in this project are substantiated with static audit files located directly in the repository:
- **Comprehensive Benchmark Report:** [`evaluation/benchmark_report.md`](evaluation/benchmark_report.md) provides detailed cross-departmental breakdown tables and numerical tolerance checks.
- **Per-Sample JSON Results:** [`evaluation/sample_outputs/evaluation_results.json`](evaluation/sample_outputs/evaluation_results.json) documents exact prompt-by-prompt model generations, side-by-side base model answers, ground truth references, and computed token overlap scores.
- **Execution Training Logs:** [`evaluation/training_log.txt`](evaluation/training_log.txt) captures the complete loss trajectory, step timings, gradient norms, and checkpoint serialization metrics across all 5 training epochs.
- **Streamlit Evaluation Interface Visual:** [`docs/screenshots/streamlit_evaluation_dashboard.svg`](docs/screenshots/streamlit_evaluation_dashboard.svg) displays the interactive comparison dashboard, parameter cards, and live loss curve.

![Streamlit LoRA Evaluation Dashboard](docs/screenshots/streamlit_evaluation_dashboard.svg)

### Key Observations:
- **Hallucination Prevention:** The base GPT-2 model generates fluent English but invents non-existent municipal office addresses, incorrect fee amounts, and incorrect form codes.
- **Parametric Adaptation:** LoRA successfully teaches the model precise departmental terms (e.g., "Form BC-1", "$15 search fee", "Department of Public Works schedule") while preserving natural language grammar.
- **Zero Runtime Overhead:** After merging the adapter ($W_{merged} = W_0 + \frac{\alpha}{r} B \cdot A$), inference latency matches base GPT-2 exactly with zero extra kernel launches.

---

## 🔍 Model Shortcomings & Failure Modes

A rigorous evaluation requires explicitly acknowledging the model's limitations:

1. **Base Model Capacity & Expressiveness (GPT-2 124M):**
   - GPT-2 124M is an early-generation autoregressive model without modern instruction-tuning (RLHF/DPO) or chain-of-thought capabilities.
   - For queries requiring multi-step conditional logic (e.g., "If I moved 3 months ago but work in another district, which permit applies?"), the model may struggle with constraint adherence compared to larger modern foundation models.

2. **Lexical Grounding Metric Limitations:**
   - The token overlap metric measures exact content word recall against reference guidelines. When the model generates a semantically valid paraphrase or concise summary using synonyms, the lexical score can penalize the output even if the answer is accurate.
   - Conversely, a repetitive response that repeats policy keywords without proper logical structure could score artificially high on pure token recall.

3. **Out-of-Distribution (OOD) Queries:**
   - Inquiries on topics absent from the training split (e.g., specialized zoning appeals or newly enacted ordinances) risk triggering base model hallucination.
   - **Recommended Mitigation:** In high-stakes public deployments, pair LoRA generation with a lexical/embedding retrieval gate that routes low-confidence or out-of-distribution queries directly to human municipal officers.

4. **Context Window Constraints:**
   - GPT-2's 1024-token context window limits the length of detailed statutory documents and municipal code that can be ingested simultaneously.

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │      Public FAQ Answer Bot Platform          │
                    └──────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌──────────────────┐                                               ┌──────────────────┐
│  Dataset Manager │                                               │   LoRA Engine    │
├──────────────────┤                                               ├──────────────────┤
│ • CSV/JSON Upload│                                               │ • Freeze Base    │
│ • Deduplication  │                                               │ • Inject Adapters│
│ • Stratified Split                                               │ • Trainable <0.3%│
└────────┬─────────┘                                               └────────┬─────────┘
         │                                                                  │
         └─────────────────────────────────┬────────────────────────────────┘
                                           ▼
                            ┌─────────────────────────────┐
                            │    Fine-Tuning Pipeline     │
                            │ • AdamW Optimizer           │
                            │ • Cross-Entropy Loss        │
                            │ • Real-time Metrics         │
                            └──────────────┬──────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌──────────────────┐                                               ┌──────────────────┐
│ Evaluation Suite │                                               │ Citizen Chat UI  │
├──────────────────┤                                               ├──────────────────┤
│ • Grounding Score│                                               │ • Policy Answers │
│ • Perplexity     │                                               │ • Source Link    │
│ • Side-by-side   │                                               │ • Domain Guard   │
└──────────────────┘                                               └──────────────────┘
```

---

## 📁 Repository Structure

```
.
├── app/
│   ├── core.py             # LoRA implementation, training loop, evaluation metrics
│   ├── llm.py              # LLM inference wrappers & generation handlers
│   ├── main.py             # Streamlit application multi-page interface
│   └── storage.py          # Adapter persistence & state storage
├── data/
│   ├── public_service_faqs.json  # Curated public service FAQ records (JSON)
│   └── public_service_faqs.csv   # Curated public service FAQ records (CSV)
├── evaluation/             # Static evaluation outputs, reports, and logs
│   ├── benchmark_report.md # Comprehensive quantitative analysis & baseline breakdown
│   ├── training_log.txt    # Epoch-by-epoch loss trajectories and gradient norms
│   └── sample_outputs/     # Per-sample prompt and generation comparisons (JSON)
├── docs/screenshots/       # Visual UI walkthroughs and dashboard screenshots
├── scripts/                # Helper utilities and end-to-end audit scripts
├── tests/                  # Pytest automated test suite
├── Dockerfile              # Container deployment recipe
├── requirements.txt        # Pinned project dependencies
└── README.md               # Project documentation
```

---

## 🖥️ Application Modules Walkthrough

The Streamlit interface provides 6 dedicated tabs:

1. **📊 Dashboard:** Overview of system status, active adapter details, and trainable vs. frozen parameter breakdown.
2. **📁 Dataset Manager:** Upload departmental FAQ records (JSON/CSV), validate schemas, detect duplicates, and view train/validation/test splits.
3. **⚡ LoRA Training:** Configure hyperparameters ($r$, $\alpha$, learning rate, epochs, dropout) and launch live fine-tuning with loss visualization.
4. **📈 Evaluation:** Compare base GPT-2 with LoRA-adapted responses side-by-side using quantitative grounding scores and perplexity metrics.
5. **💬 FAQ Chat:** Interactive chatbot answering citizen inquiries grounded strictly in official departmental guidelines with source citations.
6. **💾 Adapter Management:** Inspect, serialize, load, and merge trained adapters directly into base model weights.

---

## 🧪 Testing & Verification

Run the comprehensive automated test suite:
```bash
pytest -v
```

The test suite validates:
- Dataset loading, schema validation, and prompt formatting.
- Mathematical equivalence at step 0 ($B=\mathbf{0}$) and parameter freezing.
- LoRA forward pass, loss computation, and backpropagation.
- Weight merging consistency ($W_{merged} = W_0 + \frac{\alpha}{r} B \cdot A$).
- Streamlit application and component compilation.

---

## 📜 Dataset Format & Provenance

### Data Source & Collection Methodology
The training dataset consists of 20 verified municipal and public administration policy items across 5 core civic domains:
- **Civic Registration & Records:** Birth certificates, property deeds, vital records (sources: USA.gov Vital Records, Municipal Registrar guidelines).
- **Public Transportation & Licensing:** Commercial driver's license renewal, senior transit passes (sources: Federal Motor Carrier Safety Administration, Federal Transit Administration Section 5307).
- **Public Works & Environmental Services:** Curbside hazardous waste, bulky waste pickup (sources: US Environmental Protection Agency guidelines, municipal sanitation rules).
- **Property Tax & Housing Authority:** Property tax deadlines, assessment appeals, residential zoning permits (sources: Municipal Property Tax clearance guidelines).
- **Municipal Licensing & Community Services:** Public park pavilion reservations, business licensing (sources: City administrative codes).

Both JSON (`data/public_service_faqs.json`) and CSV (`data/public_service_faqs.csv`) files are bundled with the repository.

### Exact JSON Schema
Each record conforms to the following schema:
```json
[
  {
    "question": "How do I apply for a certified copy of a birth certificate?",
    "answer": "To obtain a certified copy of a birth certificate, submit Form BC-1 online via the municipal citizen portal or in person at the Vital Statistics Registrar office. You must provide a valid government-issued photo ID, proof of relationship or direct legal interest, and pay the statutory search fee of $15. Processing typically takes 5 to 7 business days.",
    "category": "Civic Registration & Records",
    "source": "https://www.usa.gov/birth-certificate",
    "jurisdiction": "US State Vital Records (Demonstration Standard)"
  }
]
```

Each FAQ entry is formatted into an autoregressive prompt during training:
```text
Department: Civic Registration & Records
Question: How do I apply for a certified copy of a birth certificate?
Answer: To obtain a certified copy of a birth certificate, submit Form BC-1...<|endoftext|>
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
