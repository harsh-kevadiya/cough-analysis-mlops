# 🫁 Respiratory Health Analysis System
**A Deep Learning & MLOps Framework for Diagnostic Screening**

This project demonstrates a production-ready Machine Learning pipeline developed on macOS, utilizing **Bottleneck CNN architectures** to classify symptomatic respiratory patterns from raw audio.

## 🚀 Key Engineering Highlights
- **Architecture:** Custom 3-layer CNN with **Global Average Pooling** to mitigate environmental noise bias.
- **Feature Engineering:** 3-Channel Spectrogram extraction (Static + Delta + Delta-Delta) to capture acoustic transients.
- **Production Layer:** Fully functional **FastAPI** backend for real-time inference.
- **Reliability:** Integrated **EarlyStopping** and **ReduceLROnPlateau** callbacks to ensure optimal model convergence.

## 🛠️ System Components
- **Inference Engine:** FastAPI (`/predict` endpoint)
- **Signal Processing:** Librosa-based feature pipeline
- **Model Storage:** Keras H5/SavedModel format with synced scaling parameters
- **CI/CD:** GitHub Actions for automated build verification

## 📊 Visual Analytics

### Model Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)
*Insight: Focused on maximizing Recall for symptomatic cases to serve as an effective screening tool.*

### Multi-Channel Feature Map
![Feature Visualization](reports/feature_deep_dive.png)
*Technical Detail: Uses 64 Mel-bands with a hop length optimized for 1-second audio windows.*

## 📂 Local Workspace Structure
```text
.
├── api/                # FastAPI Production Code
├── src/                # Training & Evaluation Logic
├── models/             # Optimized Weights & Scaling Params
├── reports/            # Performance Visualizations
└── .github/            # Automated MLOps Workflows