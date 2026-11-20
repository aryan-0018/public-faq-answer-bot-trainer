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
│   └── public_service_faqs.json  # Curated public service FAQ records
├── scripts/                # Helper utilities and pipeline scripts
├── tests/                  # Pytest automated test suite
├── Dockerfile              # Container deployment recipe
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Installation

Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/your-username/public-faq-answer-bot-trainer.git
cd public-faq-answer-bot-trainer

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Application

Run the Streamlit web dashboard:
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Application Features & Walkthrough

The Streamlit interface provides 6 dedicated modules:

1. **📊 Dashboard:** Overview of system status, active adapter details, and trainable vs. frozen parameter breakdown.
2. **📁 Dataset Manager:** Upload departmental FAQ records (JSON/CSV), validate schemas, detect duplicates, and view train/validation/test splits.
3. **⚡ LoRA Training:** Configure hyperparameters ($r$, $\alpha$, learning rate, epochs, dropout) and launch live fine-tuning with loss visualization.
4. **📈 Evaluation Suite:** Compare base GPT-2 with LoRA-adapted responses side-by-side using quantitative grounding scores and perplexity metrics.
5. **💬 Citizen FAQ Chat:** Interactive chatbot answering citizen inquiries grounded strictly in official departmental guidelines with source citations.
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

## 📜 Dataset Format

The system ingests structured departmental FAQs in JSON or CSV format:

```json
[
  {
    "category": "Civic Registration & Records",
    "question": "How do I apply for a certified copy of a birth certificate?",
    "answer": "Submit Form BC-1 online via the citizen portal or in person at the Vital Statistics Registrar office with a valid photo ID and statutory fee of $15.",
    "source": "https://gov.services.local/vital-records/birth-certificates"
  }
]
```

Each FAQ entry is formatted into an autoregressive prompt during training:
```text
Department: Civic Registration & Records
Question: How do I apply for a certified copy of a birth certificate?
Answer: Submit Form BC-1 online via the citizen portal...<|endoftext|>
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
