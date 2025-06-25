
# Federated Learning with Model Editing: Mitigating Client Heterogeneity through Sparsity and Weighted Aggregation

This project presents the development and evaluation of models within a Federated Learning framework using the CIFAR-100 dataset. A centralized model is first implemented, followed by a federated version assessed under both IID and non-IID data distributions. Model editing with varying weight sparsity is applied in both settings to evaluate performance impacts. Additionally, the study examines client heterogeneity by simulating imbalanced data distributions across clients, with variations in class presence and dataset size. This scenario highlights the negative effects of ‘harmful’ clients on the global model. To address these challenges, two corrective actions are applied: sparse fine-tuning to limit disruptive client updates, and weighted aggregation based on client dataset properties.

---

## 📁 Repository Structure

```
federated-learning-project/
├── checkpoints/                          # Directory containing the saved models
├── data/                                 # Script to load the CIFAR-100 dataset
├── dataset/                              # Directory containing the dataset
├── model/                                # Scripts for model operations:
│   ├── federated_averaging.py            # Federated averaging logic
│   ├── model_editing.py                  # Model editing and sparsity management
│   ├── hyperparameter_tuning.py          # Hyperparameter tuning routines
│   ├── prepare_model.py                  # Model loading and layer freezing
│   └── unbalance.py                      # Tools for unbalanced data: weights, discrepancy, severity
├── results/                              # Directory containing the results of the experiments
├── scripts/                              # Jupyter notebooks to reproduce experiments
│   ├── centralized_baseline.ipynb
│   ├── centralized_model_editing.ipynb
│   ├── federated_averaging.ipynb
│   ├── federated_model_editing.ipynb
│   ├── federated_unbalance_head.ipynb    # Contains personal contribution experiments
│   └── federated_unbalance_masked.ipynb  # Contains personal contribution experiments
├── train.py                              # Standalone script to train a model
├── eval.py                               # Standalone script to evaluate a model
├── requirements.txt                      # Python dependencies for reproducibility
└── README.md                             # This documentation
```

---

## 🚀 How to Reproduce the Experiments on Google Colab

All experiments can be executed in Google Colab. Some notebooks allow configuration of experiment parameters in the first cell (e.g., number of classes per client, sparsity levels, etc.).

### Step-by-step Instructions:

1. **Open [Google Colab](https://colab.research.google.com/)**
2. **Clone the repository** inside a new notebook or terminal cell:
   ```bash
   !git clone https://github.com/AlessandroMaini/federated-learning-project.git
   %cd federated-learning-project
   ```
3. **Install dependencies**:
   ```bash
   !pip install -r requirements.txt
   ```
4. **Run one of the notebooks** depending on the desired experiment:
   - `centralized_baseline.ipynb`: Centralized training baseline
   - `centralized_model_editing.ipynb`: Centralized model editing baseline
   - `federated_averaging.ipynb`: Federated learning with IID and non-IID clients
   - `federated_model_editing.ipynb`: Federated learning with model editing (both IID and non-IID cases)
   - `federated_unbalance_head.ipynb`: Federated learning in unbalanced scenario training only the head. Includes 2 modes:
     - Baseline
     - Discrepancy-aware weighted aggregation
   - `federated_unbalance_masked.ipynb`: Federated learning in unbalanced scenario with sparse fine-tuning. Includes multiple modes:
     - Baseline
     - Discrepancy-aware weighted aggregation
     - Severity-aware sparse fine-tuning

🔧 Customize parameters in the **first cell** of the notebooks to adjust:
- Number of classes per client
- Client masks density levels
- Experiment type (baseline, discrepancy-aware, severity-aware)

---

## 📚 Notes

- All experiments are designed to run on **Google Colab**, with GPU acceleration enabled.
- CIFAR-100 is automatically loaded using provided dataset scripts.
- All required packages are listed in `requirements.txt`. Install them with:
  ```bash
  pip install -r requirements.txt
  ```

---
