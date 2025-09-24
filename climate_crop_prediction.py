"""
Climate and Crop Yield Prediction Pipeline
=========================================
Single-file example pipeline that demonstrates how to build
an AI model to predict local climate variables and crop yield
using 50 years of historical data and accounting for global
warming trends.

This is a template and proof-of-concept. Replace the data-loading
placeholders with your real datasets (local weather station data,
satellite-derived NDVI, soil maps, historical yields, CO2/GHG time
series, etc.).

How the pipeline is organized:
- Data schema expectations and synthetic-data generator (for testing)
- Feature engineering that builds spatio-temporal features and
  encodes global-warming trends (via CO2 and temperature anomaly series)
- Two-model ensemble:
    * XGBoost for static/tabular features + aggregated time features
    * LSTM (PyTorch) for sequential/time-series modeling
  We blend their predictions to get final climate & yield forecasts.
- Training, evaluation, saving, and inference example.

Required packages:
- numpy, pandas, scikit-learn, xgboost, torch

Note: This script focuses on clarity and modularity. For production,
consider: better hyperparameter search, spatial models (GCN),
transformer time-series, uncertainty quantification, bias checks,
and deployment considerations.
"""

import os
import math
import json
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Optional imports; guard in case not installed
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "history_years": 50,
    "forecast_years": 1,  # how many years ahead to predict
    "seq_length": 12,     # months of input for LSTM
    "batch_size": 64,
    "xgb_params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    "device": "cuda" if torch and torch.cuda.is_available() else "cpu",
}

# -----------------------------
# Synthetic data generator (for testing/demo only)
# -----------------------------

def generate_synthetic_dataset(n_locations=200, years=50, seed=1):
    """Generate a synthetic dataset with monthly climate records and crop yields.

    Returns a tuple: (climate_df, yield_df, global_trend_df)

    climate_df columns: [loc_id, date, temp_c, precip_mm, humidity, radiation, ndvi]
    yield_df columns: [loc_id, year, crop_yield_ton_per_ha]
    global_trend_df columns: [year, co2_ppm, temp_anomaly_c]
    """
    rng = np.random.RandomState(seed)

    months = years * 12
    start_year = 1970
    dates = pd.date_range(f"{start_year}-01-01", periods=months, freq="MS")

    rows = []
    for loc in range(n_locations):
        # base climate for location
        base_temp = rng.uniform(5, 25)  # mean annual temp
        base_precip = rng.uniform(300, 2000) / 12.0  # monthly average
        base_hum = rng.uniform(40, 80)
        base_rad = rng.uniform(80, 250)

        # location sensitivity to warming
        temp_sensitivity = rng.uniform(0.4, 1.6)
        precip_trend_dir = rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

        for i, dt in enumerate(dates):
            year_frac = (dt.year + (dt.month - 1) / 12.0) - start_year
            # global warming effect (synthetic)
            gw_effect = 0.02 * year_frac  # degrees per year

            # seasonal cycle
            month_angle = 2 * math.pi * (dt.month - 1) / 12.0
            seasonal_temp = 10 * math.sin(month_angle)

            temp = base_temp + seasonal_temp + temp_sensitivity * gw_effect + rng.normal(0, 0.8)
            precip = base_precip * (1 + 0.01 * precip_trend_dir * year_frac) * (1 + 0.2 * np.sin(month_angle + 0.5))
            precip += rng.normal(0, precip * 0.2)
            humidity = np.clip(base_hum + 5 * np.sin(month_angle) + rng.normal(0, 5), 5, 100)
            rad = base_rad * (1 + 0.05 * np.cos(month_angle)) + rng.normal(0, 15)
            ndvi = np.clip(0.3 + 0.4 * np.sin(month_angle) + 0.001 * year_frac + rng.normal(0, 0.05), 0, 1)

            rows.append({
                "loc_id": loc,
                "date": dt,
                "temp_c": temp,
                "precip_mm": precip,
                "humidity": humidity,
                "radiation": rad,
                "ndvi": ndvi,
            })

    climate_df = pd.DataFrame(rows)

    # yearly yields (aggregate from monthly features + global trend + noise)
    yield_rows = []
    for loc in range(n_locations):
        for year in range(start_year, start_year + years):
            annual = climate_df[(climate_df["loc_id"] == loc) & (climate_df["date"].dt.year == year)]
            # a simple yield function: depends on mean growing season temp, total precip, avg ndvi
            gseason = annual[(annual["date"].dt.month >= 4) & (annual["date"].dt.month <= 9)]
            mean_temp = gseason["temp_c"].mean()
            total_precip = gseason["precip_mm"].sum()
            mean_ndvi = gseason["ndvi"].mean()

            # synthetic CO2 and temp anomaly effect
            year_idx = year - start_year
            co2 = 320 + 1.5 * year_idx + rng.normal(0, 1)
            temp_anom = 0.02 * year_idx + rng.normal(0, 0.05)

            # crop-specific parameters
            opt_temp = rng.uniform(18, 24)
            temp_penalty = -0.02 * (mean_temp - opt_temp) ** 2
            precip_effect = 0.0008 * (total_precip - 400)

            base_yield = 2.5 + mean_ndvi * 3.0
            yield_val = base_yield + temp_penalty + precip_effect + 0.001 * (co2 - 300) + 0.05 * temp_anom
            yield_val += rng.normal(0, 0.3)
            yield_rows.append({"loc_id": loc, "year": year, "crop_yield_ton_per_ha": max(0.1, yield_val)})

    yield_df = pd.DataFrame(yield_rows)

    # global trend series
    years_list = list(range(start_year, start_year + years))
    co2 = [320 + 1.5 * i + rng.normal(0, 1) for i in range(years)]
    temp_anom = [0.02 * i + rng.normal(0, 0.05) for i in range(years)]
    global_trend_df = pd.DataFrame({"year": years_list, "co2_ppm": co2, "temp_anomaly_c": temp_anom})

    return climate_df, yield_df, global_trend_df


# -----------------------------
# Feature engineering
# -----------------------------

def build_features(climate_df: pd.DataFrame, yield_df: pd.DataFrame, global_trend_df: pd.DataFrame):
    """Produce per-location, per-year features suitable for tabular and sequence models.

    Returns: X_tabular, y, seq_data (dict mapping loc->time-series matrix), meta
    """
    # Ensure datetime
    climate = climate_df.copy()
    climate["year"] = climate["date"].dt.year
    climate["month"] = climate["date"].dt.month

    # Aggregate monthly to yearly features
    agg_funcs = {
        "temp_c": ["mean", "max", "min", "std"],
        "precip_mm": ["sum", "mean", "std"],
        "humidity": ["mean"],
        "radiation": ["mean"],
        "ndvi": ["mean", "max"],
    }
    yearly = climate.groupby(["loc_id", "year"]).agg(agg_funcs)
    yearly.columns = ["_".join(col).strip() for col in yearly.columns.values]
    yearly = yearly.reset_index()

    # merge with yields
    df = pd.merge(yearly, yield_df, on=["loc_id", "year"], how="left")

    # merge global trend by year
    df = pd.merge(df, global_trend_df, on="year", how="left")

    # Add engineered features: anomalies relative to location baseline (first 10 years)
    baselines = df[df["year"] < (df["year"].min() + 10)].groupby("loc_id").median().reset_index()
    bcols = [c for c in baselines.columns if c not in ["loc_id", "year", "crop_yield_ton_per_ha"]]
    baseline_map = baselines.set_index("loc_id")[bcols].to_dict(orient="index")

    def attach_baseline(row):
        loc = row["loc_id"]
        base = baseline_map.get(loc, {})
        out = {}
        for c in bcols:
            out[f"{c}_anom"] = row.get(c, np.nan) - base.get(c, np.nan)
        return pd.Series(out)

    anoms = df.apply(attach_baseline, axis=1)
    df = pd.concat([df, anoms], axis=1)

    # Prepare tabular matrix (per-year)
    target = "crop_yield_ton_per_ha"
    drop_cols = ["loc_id", "year", target]
    X_tab = df.drop(columns=drop_cols)
    y = df[target].values

    # Scale numeric columns
    numeric_cols = X_tab.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_tab[numeric_cols] = scaler.fit_transform(X_tab[numeric_cols].fillna(0.0))

    # Sequence data: for each location, create monthly sequences of length seq_length months (for LSTM)
    seq_length = CONFIG["seq_length"]
    seq_map = {}
    grouped = climate.sort_values(["loc_id", "date"]).groupby("loc_id")
    for loc, g in grouped:
        # features per month
        feat = g[["temp_c", "precip_mm", "humidity", "radiation", "ndvi"]].values
        dates = g["date"].values
        # create rolling windows aligned to month-end and pick last month of each year as target index
        seqs = []
        seq_years = []
        for i in range(seq_length - 1, len(feat)):
            window = feat[i - seq_length + 1:i + 1]
            seqs.append(window)
            seq_years.append(g.iloc[i]["year"])
        seq_map[loc] = {"seqs": np.array(seqs), "years": np.array(seq_years)}

    meta = {"scaler": scaler, "tab_columns": X_tab.columns.tolist()}
    return X_tab, y, df[["loc_id", "year"]], seq_map, meta


# -----------------------------
# XGBoost model (tabular)
# -----------------------------

def train_xgboost(X, y, params=None):
    if xgb is None:
        raise RuntimeError("xgboost is required to run XGBoost model. Please install xgboost.")
    params = params or CONFIG["xgb_params"]
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


# -----------------------------
# LSTM model (PyTorch)
# -----------------------------

class SeqDataset(Dataset):
    def __init__(self, seqs, years, targets_map):
        # seqs: (n_samples, seq_len, n_features)
        # years: (n_samples,) corresponding year for each seq
        # targets_map: {(loc, year) -> target}
        self.seqs = []
        self.targets = []
        self.years = years
        for i, s in enumerate(seqs):
            year = int(years[i])
            self.seqs.append(s.astype(np.float32))
            self.targets.append(float(targets_map.get(year, np.nan)))
        self.seqs = np.array(self.seqs)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        # take last timestep
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)


def train_lstm_for_location(seq_map, target_df, epochs=10):
    if torch is None:
        raise RuntimeError("PyTorch is required to run LSTM. Please install torch.")

    # Build a flattened dataset across all locations
    all_seqs = []
    all_years = []
    targets_map = {}
    # target_df is DataFrame with loc_id, year, crop_yield_ton_per_ha
    for _, r in target_df.iterrows():
        targets_map[(r.loc_id, r.year)] = r.crop_yield_ton_per_ha

    for loc, v in seq_map.items():
        seqs = v["seqs"]  # (n, seq_len, f)
        yrs = v["years"]  # (n,)
        # map years to targets per location
        for i in range(len(seqs)):
            all_seqs.append(seqs[i])
            # store (loc, year) to allow lookup
            all_years.append((loc, int(yrs[i])))

    # create dataset where target for item i is target[(loc, year)]
    targets_map_flat = {}
    for (loc, year), val in targets_map.items():
        targets_map_flat[(loc, year)] = val

    # Filter out items without target
    seqs_f = []
    years_f = []
    targets_f = []
    for i, (loc, year) in enumerate(all_years):
        t = targets_map_flat.get((loc, year), None)
        if t is None or np.isnan(t):
            continue
        seqs_f.append(all_seqs[i])
        years_f.append(year)
        targets_f.append(t)

    X = np.array(seqs_f, dtype=np.float32)
    y = np.array(targets_f, dtype=np.float32)

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    loader_train = DataLoader(dataset_train, batch_size=CONFIG["batch_size"], shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=CONFIG["batch_size"], shuffle=False)

    model = LSTMModel(input_size=X.shape[-1]).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in loader_train:
            xb = xb.to(CONFIG["device"])
            yb = yb.to(CONFIG["device"])
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        preds_val = []
        ys = []
        with torch.no_grad():
            for xb, yb in loader_val:
                xb = xb.to(CONFIG["device"])
                yb = yb.to(CONFIG["device"])
                pv = model(xb).cpu().numpy()
                preds_val.extend(pv.tolist())
                ys.extend(yb.cpu().numpy().tolist())
        if epoch % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, val_mse={mean_squared_error(ys, preds_val):.4f}")

    return model


# -----------------------------
# Simple blending and evaluation
# -----------------------------

def blend_and_evaluate(xgb_model, lstm_model, X_tab, seq_map, meta, loc_year_df, target_df):
    """Make predictions per-row in loc_year_df and evaluate against target_df."""
    # X_tab rows correspond to loc_year_df order assumptions in build_features
    # For simplicity, align by index
    X_tab_np = X_tab.values
    xgb_preds = xgb_model.predict(X_tab_np) if xgb is not None else np.zeros(len(X_tab_np))

    # For LSTM, find the last sequence for (loc, year) — we take the last monthly window whose year matches
    lstm_preds = []
    for _, row in loc_year_df.iterrows():
        loc = int(row.loc_id)
        year = int(row.year)
        seqs = seq_map.get(loc, {}).get("seqs", None)
        yrs = seq_map.get(loc, {}).get("years", None)
        if seqs is None:
            lstm_preds.append(np.nan)
            continue
        # find indices where yrs == year
        idxs = np.where(yrs == year)[0]
        if len(idxs) == 0:
            lstm_preds.append(np.nan)
            continue
        # take mean of lstm model predictions across those indices
        inputs = torch.from_numpy(seqs[idxs].astype(np.float32)).to(CONFIG["device"])
        with torch.no_grad():
            out = lstm_model(inputs).cpu().numpy()
        lstm_preds.append(float(np.nanmean(out)))

    lstm_preds = np.array([p if not np.isnan(p) else np.nanmean([v for v in lstm_preds if not np.isnan(v)]) for p in lstm_preds])

    # simple weighted blend (weights tuned or learned)
    w_xgb = 0.6
    w_lstm = 0.4
    blended = w_xgb * xgb_preds + w_lstm * lstm_preds

    # merge targets
    merged = loc_year_df.copy()
    merged["pred"] = blended
    merged = pd.merge(merged, target_df, on=["loc_id", "year"], how="left")

    mse = mean_squared_error(merged["crop_yield_ton_per_ha"], merged["pred"])
    r2 = r2_score(merged["crop_yield_ton_per_ha"], merged["pred"])
    print(f"Blend evaluation: MSE={mse:.4f}, R2={r2:.4f}")
    return merged


# -----------------------------
# End-to-end runner (example)
# -----------------------------

def run_demo():
    print("Generating synthetic data...")
    climate_df, yield_df, global_trend_df = generate_synthetic_dataset(n_locations=120, years=CONFIG["history_years"]) 
    print("Building features...")
    X_tab, y, loc_year_df, seq_map, meta = build_features(climate_df, yield_df, global_trend_df)

    # train-test split on tabular rows
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_tab, y, loc_year_df.index.values, test_size=0.2, random_state=42
    )

    # XGBoost
    print("Training XGBoost (tabular)...")
    xgb_model = train_xgboost(X_train, y_train)

    # LSTM: train on all available sequences and associated targets
    print("Training LSTM (sequential)...")
    lstm_model = train_lstm_for_location(seq_map, yield_df, epochs=6)

    # Evaluate on test rows
    test_loc_year = loc_year_df.iloc[idx_test].reset_index(drop=True)
    X_test_df = X_tab.iloc[idx_test].reset_index(drop=True)

    merged = blend_and_evaluate(xgb_model, lstm_model, X_test_df, seq_map, meta, test_loc_year, yield_df)

    # Save models
    if xgb is not None:
        xgb_model.save_model("xgb_crop_model.json")
    if torch is not None:
        torch.save(lstm_model.state_dict(), "lstm_crop_model.pt")
    print("Demo complete. Models saved: xgb_crop_model.json, lstm_crop_model.pt")


if __name__ == '__main__':
    run_demo()