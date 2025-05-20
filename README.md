# transformer-weather-prediction

Time series temperature forecasting using a Transformer-based Seq2Seq model with ARIMA residual adjustment.

## 🧠 Model Summary

This project implements a hybrid forecasting pipeline for daily average temperatures across 21 European cities.  
The core model is a Transformer-based Sequence-to-Sequence (Seq2Seq) architecture trained on sliding windows of historical meteorological data.  
To enhance long-term accuracy, final predictions are corrected using an ARIMA model fitted on the residuals.

## ⚙️ Architecture

- **Encoder-Decoder Transformer** with positional encoding and autoregressive decoding
- Input sequence: 45-day historical window + seasonal lags (weekly and yearly)
- Output sequence: 7-day forward temperature forecast
- **Residual correction**: ARIMA(1,0,1)(1,1,1)[7] fitted on the last year of training residuals

## 🛠️ Technologies Used

- Python · PyTorch · scikit-learn · statsmodels
- Custom DataLoader, Early Stopping, Learning Rate Scheduling
- Evaluation metrics: MAE, RMSE, and R²

## 📊 Results (Test Set)

- **MAE**: 2.36 °C  
- **RMSE**: 3.02 °C  
- **R²**: 0.88

## 📁 Repository Contents

- `transformer_weather_forecasting.py`: Full pipeline for preprocessing, training, evaluation, and ARIMA correction
- `weather_api.py`: FastAPI-based endpoint for real-time temperature prediction
- `modelo_final.pth`: Trained model weights (see link below)
- `datos_combinados.csv`: Preprocessed input dataset  
- `resultados_mejorado.png`: Final prediction vs actual scatter plot

## 📦 Download Trained Model

🔗 You can download the trained model weights here:  
👉 [modelo_final.pth (Google Drive)](https://drive.google.com/file/d/1ziS0eqpvQCVH8lnDZmApKUwZIavyKQaa/view?usp=sharing)

> This file is hosted externally to comply with GitHub storage guidelines.

## 🔒 Repository Status

This repository is currently **private** for academic purposes. It will be made public upon institutional approval.  
For access or collaboration inquiries, please contact the author directly.

---

© Víctor Suesta — 2025
