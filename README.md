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

Before running the experiments, make sure the local data is available at:

```bash
data/raw/train.csv
```

The image paths referenced inside `train.csv` must also exist under `data/raw/`.

---

## Project Structure

```bash
.
+-- data/
|   +-- raw/
|       +-- train.csv
+-- notebooks/
+-- src/
|   +-- fl_utils.py
|   +-- plot_utils.py
+-- reports/
+-- figures/
+-- models/
+-- experiments/
|   +-- run_week3_experiments.py
+-- README.md
+-- requirements.txt
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

## Week 3 Experiments

The Week 3 script runs all required adversarial FL scenarios:

| Scenario | Aggregation | Attack |
|---|---|---|
| A_FedAvg_No_Attack | FedAvg | None |
| B_FedAvg_LabelFlip | FedAvg | Label flipping + amplified malicious updates |
| C_Krum_LabelFlip | Krum | Label flipping + amplified malicious updates |
| D_Median_LabelFlip | Coordinate-wise Median | Label flipping + amplified malicious updates |
| E_TrimmedMean_LabelFlip | Trimmed Mean | Label flipping + amplified malicious updates |

Default configuration:

| Parameter | Default |
|---|---:|
| Seed | 42 |
| Sample size | 10,000 |
| Clients | 10 |
| Malicious clients | 2 |
| Communication rounds | 20 |
| Local epochs | 1 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Trim ratio | 0.2 |
| Attack strength | 8.0 |

### Full Experiment Run

From the project root, run:

```bash
python experiments/run_week3_experiments.py
```

This executes all five scenarios and saves the per-round metrics to:

```bash
experiments/week3_results.csv
experiments/week3_results.json
experiments/week3_experiment_config.json
```

The saved metrics include:

- Test accuracy
- Test loss
- Accuracy drop relative to unattacked FedAvg
- Attack success rate
- Aggregation computation time per round

### Quick Smoke Test

Use a smaller sample and fewer rounds to verify that the pipeline works before running the full experiment:

```bash
python experiments/run_week3_experiments.py --sample-size 1000 --rounds 2
```

### Custom Runs

Run for 30 communication rounds:

```bash
python experiments/run_week3_experiments.py --rounds 30
```

Use 15 clients with 3 malicious clients:

```bash
python experiments/run_week3_experiments.py --num-clients 15 --num-malicious 3
```

Run explicitly on CPU:

```bash
python experiments/run_week3_experiments.py --device cpu
```

Enable optional Gaussian model noise on malicious client states:

```bash
python experiments/run_week3_experiments.py --model-noise-std 0.05 --model-noise-scale 1.0
```

Run the original mild label-flipping attack without update amplification:

```bash
python experiments/run_week3_experiments.py --attack-strength 1.0
```

Note: Krum requires `num_clients >= 2 * num_malicious + 3`. For example, 10 clients and 2 malicious clients is valid.

---

## Generate Figures

After running the experiments, generate the Week 3 plots:

```bash
python src/plot_utils.py
```

This creates:

```bash
figures/week3_accuracy_curves.png
figures/week3_aggregation_time.png
```

The first figure compares test accuracy across communication rounds for all five scenarios. The second figure compares average aggregation computation time for FedAvg, Krum, Coordinate-wise Median, and Trimmed Mean.

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

- McMahan et al. - Federated Learning
- Byzantine-Robust Distributed Learning
- CheXpert Dataset - Stanford ML Group
