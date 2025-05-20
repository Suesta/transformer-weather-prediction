# transformer-weather-prediction

Time series temperature forecasting using a Transformer-based Seq2Seq model with ARIMA residual adjustment.

## 🧠 Model Summary

This project implements a hybrid forecasting pipeline for daily average temperatures across 21 European cities. The core model is a Transformer-based Sequence-to-Sequence (Seq2Seq) architecture trained on sliding windows of historical meteorological data. The final prediction is corrected by an ARIMA model trained on residuals, enhancing the precision of long-term estimates.

## ⚙️ Architecture

- **Encoder-Decoder Transformer** with positional encoding and autoregressive decoding
- Input sequence: 45-day historical window + seasonal lags (weekly/yearly)
- Output sequence: 7-day forward forecast
- **Residual correction** using ARIMA(1,0,1)(1,1,1)[7] on final year of training errors

## 🛠️ Technologies Used

- Python · PyTorch · scikit-learn · statsmodels
- DataLoader + Early Stopping + Learning Rate Scheduling
- Evaluation via MAE, RMSE, and R² on final predictions

## 📊 Results (Test Set)

- **MAE**: 2.36°C  
- **RMSE**: 3.02°C  
- **R²**: 0.88

## 📁 Files

- `modelo_final.pth`: Trained model weights  
- `datos_combinados.csv`: Preprocessed multivariate input  
- `resultados_mejorado.png`: Final evaluation plots  
- Main script includes preprocessing, training, ARIMA adjustment and final evaluation

## 🔒 Repository Status

This repository is currently **private** for academic reasons. It will be made public upon clearance. For inquiries, contact me directly.

---

© Víctor Suesta — 2025
