# 🫁 Respiratory Cough Analysis MLOps System
![CI Status](https://github.com/harsh-kevadiya/cough-analysis-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

An advanced end-to-end MLOps pipeline for classifying respiratory sounds using the **COUGHVID** and **COVID-19 Kaggle datasets**. This project focuses on automated preprocessing, deep learning model training (CNN), and containerized deployment.

## 🚀 Project Overview
Developed for the **3rd Year Engineering Lab**, this system processes raw audio to identify respiratory patterns. 

### Key Features:
* **Feature Extraction:** Advanced processing using Mel-spectrograms, MFCCs, and Delta features via `Librosa`.
* **Deep Learning:** CNN-based architectures trained using `Keras` and `TensorFlow`.
* **MLOps Pipeline:** Fully automated CI/CD using **GitHub Actions**.
* **Containerization:** Portable environment via **Docker** and **Docker Compose**.
* **Data Management:** Optimized repository structure using `.gitignore` for 18GB+ datasets.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Libraries:** TensorFlow, Keras, NumPy, Librosa, Scikit-learn
* **DevOps:** Docker, GitHub Actions, Git
* **Database:** Oracle SQL (Metadata management)

---

## 📁 Repository Structure
```text
├── .github/workflows/  # CI/CD Pipeline (GitHub Actions)
├── api/                # FastAPI / Flask inference code
├── models/             # Pre-trained .keras and .h5 models
├── src/                # Modular source code
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   └── train_model.py
├── dockerfile          # Container configuration
└── requirements.txt    # Python dependencies