# 🫁 AI-Driven Respiratory Health Analysis

An end-to-end Machine Learning system designed to classify respiratory health from cough sounds using a **Bottleneck Convolutional Neural Network**. This project integrates signal processing, deep learning, and MLOps best practices.

## 🏗️ Project Architecture
The system follows a modular architecture:
1. **Signal Processing:** Feature extraction using Mel-Frequency Cepstral Coefficients (MFCCs) and 3-channel Spectrograms (Static, Delta, Delta-Delta).
2. **Deep Learning:** A custom Bottleneck CNN with Global Average Pooling to minimize environmental noise overfitting.
3. **API Layer:** FastAPI service for real-time inference.
4. **MLOps:** Containerization with Docker and automated CI/CD via GitHub Actions.

## 📊 Performance & Results

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)
*Our model achieves a balanced sensitivity, prioritizing the recall of symptomatic cases.*

### Feature Visualization
![Feature Visualization](reports/feature_deep_dive.png)
*Visual representation of the 3-channel input: Mel Spectrogram, Delta, and Delta-Delta.*

## 📂 Project Structure
```text
.
├── api/                # FastAPI implementation (main.py)
├── data/               # Processed features and datasets
├── models/             # Saved model artifacts (.keras) and scaling params
├── reports/            # Generated performance plots
├── src/                # Source code (training, evaluation, visualization)
├── .github/workflows/  # CI/CD pipeline definitions
├── Dockerfile          # Container configuration
└── requirements.txt    # Project dependencies