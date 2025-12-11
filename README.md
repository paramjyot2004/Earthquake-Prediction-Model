# 🌍🔮 **EARTHQUAKE PREDICTION & RISK ANALYSIS (1990–2023)**

<p align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=wave&color=0:6a11cb,100:2575fc&height=180&section=header&text=Earthquake%20Prediction%20Model&fontSize=35&fontColor=ffffff" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Earthquake%20Prediction-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Machine%20Learning-ANN%20%7C%20Deep%20Learning-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Model-Artificial%20Neural%20Network-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Earthquake%20Data-1990–2023-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge">
</p>

---

# 🌟 Overview

This project uses **historical earthquake data (1990–2023)** to:

- Predict earthquake magnitude categories using a **Deep Learning ANN**
- Visualize global earthquake hotspots using an **interactive Folium heatmap**
- Analyze seismic patterns and risk zones  

A powerful blend of **Machine Learning + Geo-Visualization** 🌋📊

---

# 🚀 Features

## 🗺️ Interactive Earthquake Risk Map
- Folium heatmap showing global earthquake density  
- Top 100 strongest earthquakes  
- Clickable markers with details  
- Output: `earthquake_risk_map.html`

## 🤖 ANN Model for Magnitude Prediction
- Predicts classes: **Low**, **Moderate**, **High**
- Input features: significance, depth, tsunami flag, latitude, longitude
- Trained for 100 epochs  
- Dropout used to reduce overfitting

## 📊 Evaluation Metrics
- Confusion Matrix  
- Classification Report  
- Accuracy & Loss curves  
- Example predictions  

---

# 🔧 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,sklearn,numpy,pandas,matplotlib,seaborn" />
</p>

---

# 📁 Project Structure

```
📦 Earthquake Prediction Project
├── code-1.py                    # Earthquake risk map visualization  
├── final code.py                # ANN model for magnitude prediction  
├── code2.py                     # Additional analysis  
├── earthquake_risk_map.html     # Generated interactive map  
├── Confusion Matrix.png         # Confusion matrix  
├── requirements.txt             # Dependencies  
└── README.md                    # Documentation  
```

---

# 🧠 Model Architecture

| Layer | Details |
|-------|---------|
| Input Layer | 5 features |
| Dense Layer 1 | 64 neurons — ReLU — Dropout(0.3) |
| Dense Layer 2 | 32 neurons — ReLU — Dropout(0.3) |
| Output Layer | 3 neurons — Softmax |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 100 |
| Batch Size | 32 |

---

# 📥 Dataset Information

Dataset: **Earthquakes-1990-2023.csv**

Columns include:

- Longitude  
- Latitude  
- Depth  
- Tsunami Flag  
- Significance  
- Location  
- Date  
- Magnitude (Target)

⚠ Make sure to update dataset path inside the scripts.

---

# 🌋 Visualization Preview

### Heatmap (Example)
<p align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/heatmap/heatmap.png" width="450">
</p>

### Confusion Matrix (Your Output)
<p align="center">
  <img src="Confusion Matrix.png" width="450">
</p>

---

# 📈 Results Summary

- ANN model predicts earthquake categories effectively  
- High precision for **Low** & **Moderate** categories  
- Heatmap reveals patterns in:  
  - Pacific Ring of Fire  
  - Japan  
  - Indonesia  
  - Chile & Peru  
  - Himalayan Belt  

---

# 🏆 Why This Project Stands Out

✨ Combines **ML, visualization, and geoscience**  
✨ Easy-to-run scripts  
✨ Real-world dataset  
✨ Great for portfolios, ML case studies, and academic work  

---
