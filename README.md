# DoS Attack Detection using Machine Learning and Deep Learning

## Overview
This repository demonstrates the process of detecting Denial of Service (DoS) attacks using a combination of machine learning and deep learning techniques. The code is implemented using an HTTP dataset, where different classification models are evaluated to determine their effectiveness in identifying DoS attacks. Various models including ensemble methods, boosting algorithms, and neural networks (RNN, LSTM, GRU, CNN, BiLSTM) are trained and their performances are compared.

## Dataset
The dataset used in this project is an HTTP-based dataset, which contains network traffic data for both normal and DoS attack behavior. The dataset is available on Kaggle, and the project applies several preprocessing steps such as feature selection and engineering before training the models.

## Models Implemented
1. **Traditional Machine Learning Algorithms:**
   - Ensemble Methods
   - Boosting (e.g., XGBoost, LightGBM)
   
2. **Deep Learning Models:**
   - Recurrent Neural Networks (RNN)
   - Long Short-Term Memory (LSTM)
   - Gated Recurrent Unit (GRU)
   - Convolutional Neural Networks (CNN) combined with LSTM, BiLSTM, and GRU

## Model Evaluation
The models are evaluated based on the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC (ROC Curve)
- Confusion Matrix

## Installation
To run the code locally, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/tamerkanak/DoS-Attack-Detection-using-Machine-Learning-and-Deep-Learning.git
cd DoS-Attack-Detection
pip install -r requirements.txt
