# 🚀 Fraud Detection using Graph Neural Networks (GAT)

## 📌 Overview

This project implements a **fraud detection system using Graph Neural Networks (GNNs)** on the **Elliptic Bitcoin Dataset**.
Instead of treating transactions independently, we model them as a **graph**, where relationships between transactions help identify fraudulent behavior.

We use a **Graph Attention Network (GAT)** to learn the importance of neighboring transactions and improve fraud detection.

---

## 📊 Dataset

We use the **Elliptic Bitcoin Dataset**, a real-world dataset for financial fraud detection.

* Nodes → Bitcoin transactions
* Edges → Flow of money between transactions
* Features → 166 features per transaction
* Labels:

  * `1` → Fraud (illicit)
  * `2` → Normal (licit)
  * `unknown` → Unlabeled

⚠️ The dataset is **highly imbalanced (~9:1)**, with far fewer fraudulent transactions.

---

## 🧠 Approach

### 1. Graph Construction

* Converted transactions into graph format
* Remapped transaction IDs to continuous indices
* Created `edge_index` for GNN processing

### 2. Preprocessing

* Removed unknown labels from training
* Normalized features using StandardScaler
* Created train/test masks (80/20 split)
* Preserved full graph structure

### 3. Model

We implemented a **Graph Attention Network (GAT)**:

* Multi-head attention (4 heads)
* Learns importance of neighboring nodes
* Output: Binary classification (fraud / non-fraud)

---

## ⚖️ Handling Class Imbalance

Instead of undersampling (which breaks graph structure), we used:

✔ **Class-weighted loss function**

* Higher penalty for misclassifying fraud
* Ensures model focuses on minority class

---

## 🏋️ Training

* Optimizer: Adam
* Learning rate: 0.005
* Epochs: 30
* Loss: CrossEntropy with class weights

---

## 📈 Evaluation

We evaluated using:

* Precision
* Recall
* F1-score
* Confusion Matrix

### 🔥 Key Result:

* Fraud Recall: **97%**
* Fraud Precision: **27%**

👉 The model detects most fraud cases but produces some false positives.

---

## 📊 Interpretation

* High recall → model successfully catches fraud
* Lower precision → some normal transactions flagged as fraud

👉 This trade-off is acceptable in fraud detection where missing fraud is more costly.

---

## 🏗️ Project Structure

```
fraud-gnn/
│
├── data/raw/
├── notebooks/exploration.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🎤 Key Learnings

* Graph-based modeling improves fraud detection
* Class imbalance must be handled carefully
* GNNs capture relational patterns better than traditional ML
* Recall is more important than accuracy in fraud detection

---

## 🚀 Future Improvements

* Threshold tuning to improve precision
* ROC-AUC analysis
* Temporal GNN (time-aware fraud detection)
* Explainability (why a transaction is flagged as fraud)

---

## 📚 Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* Scikit-learn
* Pandas, NumPy

---

## 👨‍💻 Author

Developed as a project to understand **Graph Neural Networks and Fraud Detection Systems**.

---
