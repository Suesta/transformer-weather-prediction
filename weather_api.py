# api_predict_fixed.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIG y DEVICE
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    L = 91
    K = 7
    seasonal_lags = [7, 14, 365, 730]
    d_model = 192
    nhead = 6
    num_layers = 4
    out_steps = 7

DATA_PATH  = "C:/Users/victo/OneDrive/Escritorio/PRACTICAS/datos_combinados.csv"
MODEL_PATH = "C:/Users/victo/OneDrive/Escritorio/PRACTICAS/modelo_final.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 2) TRANSFORMER SEQ2SEQ (igual que en entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────
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
        # <-- aquí volvemos a 1 salida (tal y como entrenaste)
        self.head      = nn.Linear(cfg.d_model, 1)

    def forward(self, src):
        B = src.size(0)
        # 1) Encoder
        enc = self.input_proj(src)
        enc = self.pos_enc(enc)
        memory = self.transformer.encoder(enc)
        # 2) Decoder paso a paso
        dec_tokens = self.cls_token.expand(B, -1, -1)
        outputs = []
        for _ in range(self.cfg.out_steps):
            dec_in = self.pos_enc(dec_tokens)
            tgt_mask = self.transformer.generate_square_subsequent_mask(dec_in.size(1)).to(DEVICE)
            dec = self.transformer.decoder(tgt=dec_in, memory=memory, tgt_mask=tgt_mask)
            token = dec[:, -1:, :]                      # [B,1,d_model]
            out   = self.head(self.norm(token))         # [B,1,1]
            out   = out.squeeze(-1)                     # [B,1]
            outputs.append(out)
            dec_tokens = torch.cat([dec_tokens, token], dim=1)
        # stack -> [B, out_steps, 1] -> squeeze -> [B, out_steps]
        return torch.stack(outputs, dim=1).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# 3) PREPROCESAMIENTO y SECUENCIAS (idéntico al training)
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path_csv):
    df = pd.read_csv(path_csv)
    df = df[df['ciudad']!='combinados'].copy()
    df['time']      = pd.to_datetime(df['time'])
    df['dayofyear'] = df['time'].dt.dayofyear
    df['month']     = df['time'].dt.month
    df['sin_day']   = np.sin(2*np.pi*df['dayofyear']/365)
    df['cos_day']   = np.cos(2*np.pi*df['dayofyear']/365)

    for col in ['tavg','tmin','tmax']:
        df[col] = df.groupby('ciudad')[col].transform(lambda x: x.fillna(x.mean()))

    df['tavg_roll7']  = df.groupby('ciudad')['tavg'].transform(lambda x: x.rolling(7,1).mean())
    df['tavg_roll14'] = df.groupby('ciudad')['tavg'].transform(lambda x: x.rolling(14,1).mean())

    scaler_tavg   = StandardScaler().fit(df[['tavg']])
    scaler_roll7  = StandardScaler().fit(df[['tavg_roll7']])
    scaler_roll14 = StandardScaler().fit(df[['tavg_roll14']])
    df['tavg_scaled']        = scaler_tavg.transform(df[['tavg']])
    df['tavg_roll7_scaled']  = scaler_roll7.transform(df[['tavg_roll7']])
    df['tavg_roll14_scaled'] = scaler_roll14.transform(df[['tavg_roll14']])
    df['tmin_scaled']        = StandardScaler().fit_transform(df[['tmin']])
    df['tmax_scaled']        = StandardScaler().fit_transform(df[['tmax']])

    ohe = OneHotEncoder(sparse_output=False)
    mat = ohe.fit_transform(df[['ciudad']])
    cols_ohe = ohe.get_feature_names_out(['ciudad'])
    df_ohe   = pd.DataFrame(mat, columns=cols_ohe, index=df.index)
    df = pd.concat([df, df_ohe], axis=1)

    feats = [
        'sin_day','cos_day','month',
        'tmin_scaled','tmax_scaled',
        'tavg_roll7_scaled','tavg_roll14_scaled'
    ] + list(cols_ohe)

    return df, feats, scaler_tavg, ohe

def build_sequence_for_date(df, feats, ciudad, fecha, cfg: Config):
    city_df = df[df['ciudad']==ciudad].sort_values('time').reset_index(drop=True)
    idxs = city_df.index[city_df['time']==fecha]
    if len(idxs)==0:
        raise ValueError(f"No hay datos para {ciudad} en {fecha.date()}")
    i = idxs[0]
    half = cfg.L//2
    if i<half:
        raise ValueError("Fecha demasiado cercana al inicio")

    offsets = list(range(-half, -cfg.K))
    parts = []
    feat_mat = city_df[feats].to_numpy(np.float32)
    feat_mean= feat_mat.mean(0)

    for o in offsets:
        parts.append(feat_mat[i+o])
    for lag in cfg.seasonal_lags:
        d = fecha - pd.Timedelta(days=lag)
        j = city_df.index[city_df['time']==d]
        parts.append(feat_mat[j[0]] if len(j)>0 else feat_mean)

    seq = np.stack(parts).astype(np.float32)
    return torch.from_numpy(seq).unsqueeze(0)

# ─────────────────────────────────────────────────────────────────────────────
# 4) APP FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="API Predicción Meteorológica")

DF, FEATS, SCALER_TAVG, OHE = load_and_preprocess(DATA_PATH)
MODEL = TransformerSeq2Seq(input_size=len(FEATS), cfg=Config()).to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.eval()

class PredInput(BaseModel):
    ciudad: str
    fecha:  str  # "YYYY-MM-DD"

class PredOutput(BaseModel):
    ciudad:        str
    fecha_inicio:  str
    fechas:        List[str]
    predicciones:  List[float]

@app.post("/predecir", response_model=PredOutput)
def predecir(inp: PredInput):
    try:
        fecha = datetime.strptime(inp.fecha, "%Y-%m-%d")
        if inp.ciudad not in DF['ciudad'].unique():
            raise HTTPException(404, f"Ciudad no conocida: {inp.ciudad}")

        x = build_sequence_for_date(DF, FEATS, inp.ciudad, fecha, Config()).to(DEVICE)
        with torch.no_grad():
            y_hat = MODEL(x).cpu().numpy().reshape(-1)

        preds = SCALER_TAVG.inverse_transform(y_hat.reshape(-1,1)).flatten().tolist()
        fechas = [(fecha + timedelta(days=i)).strftime("%Y-%m-%d")
                  for i in range(Config.out_steps)]

        return PredOutput(
            ciudad=inp.ciudad,
            fecha_inicio=inp.fecha,
            fechas=fechas,
            predicciones=[round(v,2) for v in preds]
        )

    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, f"Error interno: {e}")
