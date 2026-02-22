# 📊 Data Visualisation Project — Group D
**Tama · Omar · Aamir · Khalood**

---

## 📋 Overview

This project explores unsupervised machine learning and dimensionality reduction techniques to produce meaningful, interactive data visualisations. The system clusters high-dimensional data and renders it in low-dimensional space, enabling pattern discovery and exploratory analysis without labelled data.

---

## 🎯 Aims & Objectives

- Clearly define the problem and its significance
- Apply dimensionality reduction to preserve meaningful data properties while reducing computational cost
- Implement and evaluate unsupervised clustering algorithms
- Produce intuitive, high-quality visual outputs from which conclusions can be drawn

---

## 🧠 AI Techniques

### Dimensionality Reduction
We apply at least one of the following techniques to reduce feature space while retaining core data properties:
- **PCA** (Principal Component Analysis)
- **VAE** (Variational Autoencoder)
- **FastICA** (Independent Component Analysis) — under consideration

### Clustering Algorithms
We researched and evaluated four unsupervised learning algorithms:

| Algorithm | Key Strength | Key Weakness |
|-----------|-------------|--------------|
| **K-Means** | Fast, scalable, interpretable | Assumes spherical clusters; sensitive to outliers |
| **SOM** (Self-Organising Maps) | Natural 2D output; topology-preserving | Slow; many hyperparameters |
| **DBSCAN** | No need to specify K; handles arbitrary shapes | Sensitive to ε and minPts; struggles with varying densities |
| **GMM** (Gaussian Mixture Models) | Soft clustering; flexible elliptical shapes | Assumes Gaussian distribution; slower than K-Means |

---

## ⚙️ System Design

### Functional Requirements
- Data ingestion and cleaning pipeline
- Dimensionality reduction module
- Clustering module with autonomous K selection (e.g. elbow method / BIC)
- Interactive visualisation layer

### Non-Functional Requirements
- **Platform Independence:** Runs on Windows, macOS, and Linux
- **Resource Constraints:** Functions on standard hardware (8GB RAM, 4-core CPU)
- **Modular Architecture:** Clear separation between data pipeline, AI models, and visualisation layers
- **Learnability:** Users can understand basic interactions without prior instructions

---

## 📏 Evaluation Metrics

### Quantitative
- **Information Entropy** — Measures how much of the original data's "usefulness" is preserved in the output. Calculated as `information(x) = -log(p(x))`, scored 0–1.0.
- **Silhouette Coefficient** — Validates clustering effectiveness. Score of +1 = well-clustered; -1 = misclassified. Formula: `S = (b - a) / max(a, b)`

### Qualitative
Each output is rated 1–10 across four dimensions by multiple evaluators (peers, lecturers, supervisors):

| Dimension | Questions |
|-----------|-----------|
| **Relevance** | Does the output address the goal? Are there irrelevant points? |
| **Accuracy** | Does it accurately reflect the input data? Any hallucinations? |
| **Coherence** | Is it easy to understand? Is there unnecessary data? |
| **Usefulness** | Can clear conclusions be drawn? Are patterns/biases visible? |

**Scale:** 1–3 (poor) · 4–6 (adequate) · 6–8 (good) · 9–10 (excellent)

---

## 🗂️ Project Structure

```
├── data/               # Raw and processed datasets
├── pipeline/           # Data cleaning and ingestion scripts
├── models/             # Dimensionality reduction & clustering modules
├── visualisation/      # UI and visual output layer
├── evaluation/         # Metrics and evaluation scripts
└── docs/               # Report and documentation
```

---

## 🚀 Getting Started

> Setup instructions will be added as the project develops.

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

---

## 📅 Progress & Reports

Weekly progress reports and retrospective analysis are tracked in the appendices of the project report. Refer to the repository's commit history for a timeline of development activity.

> ⚠️ A consistent commit history is maintained throughout the semester in line with project requirements.

---

## 🤖 Generative AI Usage

Details of any generative AI tools used during this project are documented in Appendix 3 of the full report.

---

