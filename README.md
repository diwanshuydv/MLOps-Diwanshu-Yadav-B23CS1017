# DLOPs Assignment 1: Deep Learning & SVM Analysis

**Author:** Diwanshu Yadav 
**Roll Number:** B23CS1017
**Course:** Deep Learning Ops (DLOPs)  
**Submission Deadline:** 24/01/2026

---

## 📄 Project Description
This repository contains the code and analysis for Assignment 1. The goal of this assignment is to compare the performance of Deep Learning models (ResNet-18, ResNet-50) against traditional Machine Learning models (SVM) on the MNIST and FashionMNIST datasets. It also includes a hardware performance analysis (CPU vs. GPU).

### 🔧 Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** `torchvision`, `scikit-learn`, `pandas`, `numpy`, `thop`

---

## 📊 Q1(a): Deep Learning Benchmarks
We trained ResNet-18 and ResNet-50 from scratch (pretrained=False) with a **70-10-20 split** using Automatic Mixed Precision (AMP).

### Table 1: FashionMNIST Test Accuracy
| Model | Batch Size | Optimizer | Learning Rate | Test Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-18 | 16 | SGD | 0.001 | 91.43 |
| ResNet-50 | 16 | SGD | 0.001 | 89.84 |
| ResNet-18 | 16 | SGD | 0.0001 | 85.28 |
| ResNet-50 | 16 | SGD | 0.0001 | 78.17 |
| ResNet-18 | 16 | Adam | 0.001 | 91.61 |
| ResNet-50 | 16 | Adam | 0.001 | 91.09 |
| **ResNet-18** | **16** | **Adam** | **0.0001** | **92.26** |
| ResNet-50 | 16 | Adam | 0.0001 | 91.79 |
| ResNet-18 | 32 | SGD | 0.001 | 90.17 |
| ResNet-50 | 32 | SGD | 0.001 | 88.23 |

### Table 2: MNIST Test Accuracy
| Model | Batch Size | Optimizer | Learning Rate | Test Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-18 | 16 | SGD | 0.001 | 99.01 |
| ResNet-50 | 16 | SGD | 0.001 | 98.89 |
| ResNet-18 | 16 | SGD | 0.0001 | 97.15 |
| ResNet-50 | 16 | SGD | 0.0001 | 96.09 |
| ResNet-18 | 16 | Adam | 0.001 | 98.70 |
| ResNet-50 | 16 | Adam | 0.001 | 97.20 |
| **ResNet-18** | **16** | **Adam** | **0.0001** | **99.29** |
| **ResNet-50** | **16** | **Adam** | **0.0001** | **99.26** |
| ResNet-18 | 32 | SGD | 0.001 | 98.96 |
| ResNet-50 | 32 | SGD | 0.001 | 98.42 |

---

## 📈 Q1(b): SVM Classification
Support Vector Machine (SVM) analysis using `Poly` and `RBF` kernels.

| Dataset | Kernel | Test Accuracy (%) | Training Time (ms) |
| :--- | :---: | :---: | :---: |
| MNIST | Poly | 97.71 | 181,106 |
| MNIST | RBF | **97.92** | 167,134 |
| FashionMNIST | Poly | 86.30 | 283,958 |
| FashionMNIST | RBF | **88.28** | 232,496 |

---

## ⚡ Q2: Hardware & Efficiency Analysis (FashionMNIST)
Comparison of training time and computational cost (FLOPs) between CPU and GPU execution.
*(Settings: Batch Size=16, SGD, LR=0.001)*

| Compute Device | Model | Test Acc (%) | Train Time (ms) | FLOPs |
| :--- | :--- | :---: | :---: | :---: |
| **CPU** | ResNet-18 | ... | 656,216 | ... |
| **CPU** | ResNet-50 | ... | ... | ... |
| **GPU** | ResNet-18 | ... | 73,544 | ... |
| **GPU** | ResNet-50 | ... | 153,237 | ... |

> **Observation:** Training on GPU provided approximately a **9x speedup** compared to CPU for ResNet-18.
