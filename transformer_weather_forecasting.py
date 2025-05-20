import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # ARIMA más ligero

#───────────────────────────────────────────────────────────────────────────────
# RUTAS
#───────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "C:/Users/victo/OneDrive/Escritorio/PRACTICAS/datos_combinados.csv"
MODEL_OUT    = "C:/Users/victo/OneDrive/Escritorio/PRACTICAS/modelo_final.pth"
RESULTS_PLOT = "C:/Users/victo/OneDrive/Escritorio/PRACTICAS/resultados_mejorado.png"

#───────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURACIÓN
#───────────────────────────────────────────────────────────────────────────────
class Config:
    L = 91                              # ventana (45 días antes, hueco 7 días)
    K = 7
    seasonal_lags = [7, 14, 365, 730]   # lags semanales y anuales
    batch_size = 128
    learning_rate = 5e-4
    epochs = 150
    patience = 10
    min_lr = 1e-5
    d_model = 192
    nhead = 6
    num_layers = 4
    out_steps = 7                       # predicción a 7 días

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#───────────────────────────────────────────────────────────────────────────────
# 2) CODIFICACIÓN POSICIONAL
#───────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

#───────────────────────────────────────────────────────────────────────────────
# 3) MODELO SEQ2SEQ CON TRANSFORMER
#───────────────────────────────────────────────────────────────────────────────
class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_size, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(input_size, cfg.d_model)
        self.pos_enc    = PositionalEncoding(cfg.d_model, max_len=cfg.L + cfg.out_steps + 1)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,cfg.d_model))
        self.norm      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, 1)

    def forward(self, src):
        B = src.size(0)
        # 3.1) Encoder
        enc = self.input_proj(src)          
        enc = self.pos_enc(enc)
        memory = self.transformer.encoder(enc)
        # 3.2) Decoder autoregresivo
        dec_tokens = self.cls_token.expand(B, -1, -1)
        outputs = []
        for _ in range(self.cfg.out_steps):
            dec_in = self.pos_enc(dec_tokens)
            tgt_mask = self.transformer.generate_square_subsequent_mask(dec_in.size(1)).to(device)
            dec = self.transformer.decoder(tgt=dec_in, memory=memory, tgt_mask=tgt_mask)
            token = dec[:, -1:, :]
            out   = self.head(self.norm(token)).squeeze(-1)
            outputs.append(out)
            dec_tokens = torch.cat([dec_tokens, token], dim=1)
        return torch.stack(outputs, dim=1).squeeze(-1)  # [B, out_steps]

#───────────────────────────────────────────────────────────────────────────────
# 4) PREPROCESAMIENTO
#───────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path_csv):
    print("🚀 Preprocesamiento mejorado...", flush=True)
    df = pd.read_csv(path_csv)
    df = df[df['ciudad']!='combinados'].copy()
    df['time']      = pd.to_datetime(df['time'])
    df['dayofyear'] = df['time'].dt.dayofyear
    df['month']     = df['time'].dt.month
    df['sin_day']   = np.sin(2*np.pi*df['dayofyear']/365)
    df['cos_day']   = np.cos(2*np.pi*df['dayofyear']/365)

    # 4.1) Imputación de nulos por ciudad
    for col in ['tavg','tmin','tmax']:
        df[col] = df.groupby('ciudad')[col].transform(lambda x: x.fillna(x.mean()))

    # 4.2) Medias móviles de temperatura
    df['tavg_roll7']  = df.groupby('ciudad')['tavg'].transform(lambda x: x.rolling(7,1).mean())
    df['tavg_roll14'] = df.groupby('ciudad')['tavg'].transform(lambda x: x.rolling(14,1).mean())

    # 4.3) Escalado independiente
    scaler_tavg   = StandardScaler().fit(df[['tavg']])
    scaler_roll7  = StandardScaler().fit(df[['tavg_roll7']])
    scaler_roll14 = StandardScaler().fit(df[['tavg_roll14']])
    df['tavg_scaled']        = scaler_tavg.transform(df[['tavg']])
    df['tavg_roll7_scaled']  = scaler_roll7.transform(df[['tavg_roll7']])
    df['tavg_roll14_scaled'] = scaler_roll14.transform(df[['tavg_roll14']])
    df['tmin_scaled']        = StandardScaler().fit_transform(df[['tmin']])
    df['tmax_scaled']        = StandardScaler().fit_transform(df[['tmax']])

    # 4.4) One-hot de ciudad
    ohe = OneHotEncoder(sparse_output=False)
    mat = ohe.fit_transform(df[['ciudad']])
    cols_ohe = ohe.get_feature_names_out(['ciudad'])
    df_ohe   = pd.DataFrame(mat, columns=cols_ohe, index=df.index)
    df = pd.concat([df, df_ohe], axis=1)

    daily_feats = [
        'sin_day','cos_day','month',
        'tmin_scaled','tmax_scaled',
        'tavg_roll7_scaled','tavg_roll14_scaled'
    ] + list(cols_ohe)

    print(f"✅ Final: {len(df)} filas, {len(daily_feats)} features.", flush=True)
    return df, daily_feats, (scaler_tavg, scaler_roll7, scaler_roll14), ohe

#───────────────────────────────────────────────────────────────────────────────
# 5) GENERACIÓN DE SECUENCIAS
#───────────────────────────────────────────────────────────────────────────────
def make_sequences(df, feats, cfg: Config):
    print("🚀 Generando secuencias...", flush=True)
    half = cfg.L // 2
    neigh_offs = list(range(-half, -cfg.K))  # solo días anteriores
    X_list, y_list = [], []
    for city, group in df.groupby('ciudad'):
        city_df  = group.sort_values('time').reset_index(drop=True)
        feat_mat = city_df[feats].to_numpy(np.float32)
        feat_mean= feat_mat.mean(0)
        tavg_s   = city_df['tavg_scaled'].to_numpy(np.float32)
        n = len(city_df)
        for i in range(half, n-half):
            # 5.1) contexto histórico
            seq = [feat_mat[i+o] for o in neigh_offs]
            # 5.2) lags estacionales
            for lag in cfg.seasonal_lags:
                day = city_df.loc[i,'time'] - pd.Timedelta(days=lag)
                idx = np.where(city_df['time']==day)[0]
                seq.append(feat_mat[idx[0]] if idx.size>0 else feat_mean)
            X_list.append(np.vstack(seq))
            # 5.3) etiqueta: próximos 7 días
            y_list.append(tavg_s[i : i+cfg.out_steps])
    X = np.stack(X_list)
    y = np.stack(y_list).astype(np.float32)
    print(f"✅ Secuencias: X={X.shape}, y={y.shape}", flush=True)
    return X, y

#───────────────────────────────────────────────────────────────────────────────
# 6) ENTRENAR, EVALUAR, ENSAMBLAR ARIMA Y GRAFICAR
#───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 6.1) Preprocesar y secuencias
    df, feats, scalers, ohe = load_and_preprocess(DATA_PATH)
    X, y = make_sequences(df, feats, Config())

    # 6.2) División cronológica 70/15/15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_te, y_val, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)

    def mkldr(Xa, ya):
        return DataLoader(TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
                          batch_size=Config.batch_size, shuffle=True)
    loaders = {
        'train': mkldr(X_tr, y_tr),
        'val':   mkldr(X_val, y_val),
        'test':  mkldr(X_te,  y_te)
    }

    # 6.3) Modelo, optimizador, scheduler, pérdida
    model     = TransformerSeq2Seq(input_size=X.shape[2], cfg=Config()).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, min_lr=Config.min_lr)
    criterion = nn.MSELoss()

    # 6.4) Entrenamiento con early stopping
    history, best, wait = {'tr':[], 'vl':[], 'lr':[]}, float('inf'), 0
    print("🚀 Entrenando...", flush=True)
    for ep in range(Config.epochs):
        # entrenamiento
        model.train(); tr_loss = 0.0
        for xb, yb in loaders['train']:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
        # validación
        model.eval(); vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in loaders['val']:
                xb, yb = xb.to(device), yb.to(device)
                vl_loss += criterion(model(xb), yb).item()
        avg_tr = tr_loss/len(loaders['train'])
        avg_vl = vl_loss/len(loaders['val'])
        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step(avg_vl)
        history['tr'].append(avg_tr); history['vl'].append(avg_vl); history['lr'].append(lr_now)

        if avg_vl < best:
            best, wait = avg_vl, 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            wait += 1
            if wait >= Config.patience:
                print(f"⚠️ Early stopping en época {ep+1}", flush=True)
                break

        print(f"Ep{ep+1}/{Config.epochs} • Train={avg_tr:.4f} • Val={avg_vl:.4f} • LR={lr_now:.5f}", flush=True)

    # 6.5) Evaluación final del Transformer
    sc_tavg, _, _ = scalers
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loaders['test']:
            xb = xb.to(device)
            p  = model(xb).cpu().numpy()
            p  = sc_tavg.inverse_transform(p)
            t  = sc_tavg.inverse_transform(yb.numpy())
            preds.extend(p.reshape(-1)); trues.extend(t.reshape(-1))

    # 6.6) Ajustar ARIMA sobre último año de residuos
    resid        = np.array(trues) - np.array(preds)
    resid_sample = resid[-365:]  # último año
    print("🚀 Ajustando ARIMA sobre residuos (1 año)...", flush=True)
    arima = ARIMA(resid_sample, order=(1,0,1), seasonal_order=(1,1,1,7))
    res_arima = arima.fit()
    adj = res_arima.predict(start=0, end=len(resid_sample)-1)
    final_adj = np.concatenate([np.zeros(len(resid) - len(adj)), adj])
    final_preds = np.array(preds) + final_adj

    # 6.7) Métricas finales y gráficos
    mae  = mean_absolute_error(trues, final_preds)
    rmse = mean_squared_error(trues, final_preds, squared=False)
    r2   = r2_score(trues, final_preds)
    print(f"\n✅ Test MAE={mae:.2f} • RMSE={rmse:.2f} • R²={r2:.2f}", flush=True)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['tr'], label='Train'); plt.plot(history['vl'], label='Val')
    plt.title("Evolución de la Pérdida"); plt.xlabel("Época"); plt.ylabel("MSE"); plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(trues, final_preds, alpha=0.3)
    mn, mx = min(trues), max(trues)
    plt.plot([mn,mx],[mn,mx],'r--')
    plt.title(f"True vs Pred (R²={r2:.2f})")
    plt.xlabel("Temp real (°C)"); plt.ylabel("Temp predicha (°C)")

    plt.tight_layout()
    plt.savefig(RESULTS_PLOT)
    plt.show()
