# Robust Federated Learning Against Adversarial Attacks

## Securing Collaborative Medical AI from Malicious Participants

### Team
Transformers

---

## Project Overview

Federated Learning (FL) enables multiple clients to collaboratively train machine learning models while keeping data locally stored, making it highly suitable for privacy-sensitive domains such as healthcare.

However, FL systems are vulnerable to adversarial or Byzantine attacks where malicious clients intentionally manipulate local updates to corrupt the global model. Such attacks may significantly reduce model reliability and compromise medical AI systems.

This project focuses on studying adversarial attacks in Federated Learning and evaluating robust aggregation strategies capable of defending collaborative medical AI systems against malicious participants.

---

## Objectives

The main objectives of this project are:

- Build a baseline Federated Learning pipeline using FedAvg.
- Simulate adversarial attacks:
  - Data poisoning
  - Model poisoning
- Implement robust aggregation methods:
  - Krum
  - Coordinate-wise Median
  - Trimmed Mean
- Compare naive and robust aggregation strategies.
- Evaluate:
  - Accuracy degradation
  - Attack Success Rate (ASR)
  - Aggregation computation time

---

## Dataset

We use a lightweight subset of the CheXpert dataset.

### Dataset Link
https://stanfordmlgroup.github.io/competitions/chexpert/

### Dataset Description
CheXpert is a large-scale chest X-ray dataset developed by Stanford University for thoracic disease classification tasks.

---

## Project Structure

```bash
.
├── data/
├── notebooks/
├── src/
├── reports/
├── figures/
├── models/
├── experiments/
├── README.md
└── requirements.txt
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/khalilboudib/Robust-Federated-Learning-Against-Adversarial-Attacks.git

cd Robust-Federated-Learning-Against-Adversarial-Attacks
```

### 2. Create a virtual environment

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Planned Experiments

| Method | Attack Scenario | Goal |
|---|---|---|
| FedAvg | No Attack | Baseline |
| FedAvg | Under Attack | Vulnerability Analysis |
| Krum | Under Attack | Byzantine Robustness |
| Median | Under Attack | Robust Aggregation |
| Trimmed Mean | Under Attack | Robust Aggregation |

---

## Expected Outcomes

- Understand vulnerabilities of Federated Learning systems.
- Evaluate robustness of aggregation defenses.
- Build a secure collaborative medical AI pipeline.
- Analyze trade-offs between robustness and computational cost.

---

## References

- McMahan et al. — Federated Learning
- Byzantine-Robust Distributed Learning
- CheXpert Dataset — Stanford ML Group