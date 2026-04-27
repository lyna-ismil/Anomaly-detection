"""
M4 - Anomaly Detection Pipeline
Dataset    : carOBD (github.com/eron93br/carOBD) - 27 OBD-II PIDs at 1Hz
Algorithms : LSTM Autoencoder (TensorFlow/Keras) + Isolation Forest (scikit-learn)
Sections   : Load -> Preprocess -> Sequence -> Train -> Evaluate -> Feature Importance -> Export

Directory layout expected:
    data/
        drive/   *.csv   <- car in motion
        idle/    *.csv   <- engine on, car stopped
        live/    *.csv   <- live driving sessions
        long/    *.csv   <- long-duration drives
        ufpe/    *.csv   <- UFPE campus routes
"""

import os, json, glob, gc, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
)
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import joblib

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
CFG = {
    "data_dir"              : "data",
    "output_dir"            : "outputs/m4",
    "target_hz"             : 1,        # resample target (1 reading/sec)
    "iqr_multiplier"        : 3.0,      # IQR clipping fence
    "rolling_window"        : 5,        # seconds for rolling mean/std
    "corr_threshold"        : 0.95,     # drop features correlated above this
    "sequence_len"          : 30,       # sliding window (seconds)
    "test_ratio"            : 0.2,
    "ae_epochs"             : 80,
    "ae_batch"              : 64,
    "ae_latent_dim"         : 8,
    "ae_lr"                 : 1e-3,
    "if_n_estimators"       : 200,
    "if_contamination"      : 0.05,
    "anomaly_pct_threshold" : 95,       # percentile -> AE threshold
    "perm_n_repeats"        : 15,       # IF permutation importance repeats
    "random_seed"           : 42,
}

FEATURES = [
    "ENGINE_RPM", "VEHICLE_SPEED", "THROTTLE", "ENGINE_LOAD",
    "COOLANT_TEMP", "LONG_TERM_FUEL_TRIM", "SHORT_TERM_FUEL_TRIM",
    "INTAKE_MANIFOLD_PRESSURE", "FUEL_TANK_LEVEL", "ABSOLUTE_THROTTLE_B",
    "ACCELERATOR_POS_D", "ACCELERATOR_POS_E", "COMMANDED_THROTTLE_ACTUATOR",
    "FUEL_AIR_EQUIV_RATIO", "RELATIVE_THROTTLE_POS",
    "INTAKE_AIR_TEMP", "TIMING_ADVANCE",
    "CATALYST_TEMPERATURE_BANK1_SENSOR1", "CATALYST_TEMPERATURE_BANK1_SENSOR2",
    "CONTROL_MODULE_VOLTAGE",
]

OUT = Path(CFG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(CFG["random_seed"])
tf.random.set_seed(CFG["random_seed"])


# -------------------------------------------------------------
# SECTION 1 - DATA LOADING
# -------------------------------------------------------------
def load_csv_folder(folder: str, label: int) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
        # Strip the "_()" suffix left by OBD-II headers like "ENGINE_RPM ()"
        df.columns = df.columns.str.replace(r"_\(\)$", "", regex=True)
        # Map dataset-specific names to the canonical FEATURES names
        col_map = {
            "COOLANT_TEMPERATURE": "COOLANT_TEMP",
            "SHORT_TERM_FUEL_TRIM_BANK_1": "SHORT_TERM_FUEL_TRIM",
            "LONG_TERM_FUEL_TRIM_BANK_1": "LONG_TERM_FUEL_TRIM",
            "PEDAL_D": "ACCELERATOR_POS_D",
            "PEDAL_E": "ACCELERATOR_POS_E",
            "FUEL_TANK": "FUEL_TANK_LEVEL",
            "FUEL_AIR_COMMANDED_EQUIV_RATIO": "FUEL_AIR_EQUIV_RATIO",
            "RELATIVE_THROTTLE_POSITION": "RELATIVE_THROTTLE_POS",
            "CATALYST_TEMPERATURE_BANK1_SENSOR1": "CATALYST_TEMPERATURE_BANK1_SENSOR1",
            "CATALYST_TEMPERATURE_BANK1_SENSOR2": "CATALYST_TEMPERATURE_BANK1_SENSOR2",
            "COMMANDED_THROTTLE_ACTUATOR": "COMMANDED_THROTTLE_ACTUATOR",
        }
        df.rename(columns=col_map, inplace=True)
        # Use float32 to halve DataFrame memory (304K rows × 60 cols)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].astype(np.float32)
        df["label"] = label
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# Columns with zero or near-zero variance (useless for anomaly detection)
USELESS_COLS = [
    "WARM_UPS_SINCE_CODES_CLEARED",
    "TIME_RUN_WITH_MIL_ON",
    "DISTANCE_TRAVELED_WITH_MIL_ON",
    "COMMANDED_EVAPORATIVE_PURGE",
    "ABSOLUTE_BAROMETRIC_PRESSURE",
    "ENGINE_RUN_TINE",
    "TIME_SINCE_TROUBLE_CODES_CLEARED",
]


def load_dataset() -> pd.DataFrame:
    folders = ["drive", "idle", "live", "long", "ufpe"]
    parts = {}
    for name in folders:
        parts[name] = load_csv_folder(os.path.join(CFG["data_dir"], name), label=0)

    normal = pd.concat(parts.values(), ignore_index=True)

    # Drop useless columns (zero / near-zero variance)
    to_drop = [c for c in USELESS_COLS if c in normal.columns]
    if to_drop:
        normal.drop(columns=to_drop, inplace=True)
        print(f"[load] dropped {len(to_drop)} useless columns: {to_drop}")

    counts = "  |  ".join(f"{k}: {len(v):,}" for k, v in parts.items())
    print(f"[load] rows: {len(normal):,}  |  {counts}")
    return normal


def resolve_features(df: pd.DataFrame) -> list[str]:
    """
    Three-tier column resolution:
      1. Exact match against FEATURES list (after uppercasing + normalising)
      2. Fuzzy match - strips common OBD prefixes/suffixes to find partial matches
      3. Fallback - uses ALL numeric columns (excluding meta columns)

    Always prints actual CSV columns so you can update FEATURES if needed.
    """
    META_COLS = {"label", "source_file", "timestamp", "time", "index"}

    # Normalise actual columns once
    actual_cols   = list(df.columns)
    actual_upper  = {c.upper().replace(" ", "_").replace("-", "_"): c for c in actual_cols}

    print(f"\n[load] actual CSV columns ({len(actual_cols)}):")
    for c in actual_cols:
        print(f"         {c}")

    # Tier 1 - exact match (normalised)
    exact = []
    for feat in FEATURES:
        norm = feat.upper().replace(" ", "_").replace("-", "_")
        if norm in actual_upper:
            exact.append(actual_upper[norm])

    if exact:
        print(f"\n[load] exact matches: {len(exact)} features -> {exact}")
        return exact

    # Tier 2 - fuzzy: any FEATURES keyword appears in column name or vice-versa
    fuzzy = []
    for col_norm, col_orig in actual_upper.items():
        if col_orig.lower() in META_COLS:
            continue
        for feat in FEATURES:
            feat_norm = feat.replace("_", "")
            col_stripped = col_norm.replace("_", "")
            if feat_norm in col_stripped or col_stripped in feat_norm:
                if col_orig not in fuzzy:
                    fuzzy.append(col_orig)
                    print(f"[load] ~ fuzzy match: '{col_orig}' <- '{feat}'")
                break

    if fuzzy:
        print(f"[load] fuzzy matched {len(fuzzy)} features")
        return fuzzy

    # Tier 3 - fallback: all numeric columns minus meta
    numeric = [
        c for c in actual_cols
        if c.lower() not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"\n[load] WARNING: no feature name matches found.")
    print(f"[load] fallback -> using all {len(numeric)} numeric columns: {numeric}")
    print("[load] -> update the FEATURES list at the top of the file to be explicit.\n")
    return numeric


# -------------------------------------------------------------
# SECTION 2 - PREPROCESSING
# Steps (in order): resample -> IQR clip -> rolling stats -> correlation filter
# -------------------------------------------------------------

def pp_resample(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Step 1 - Resample to fixed Hz.
    Requires a TIMESTAMP column (seconds). If absent, data is assumed already
    at 1 Hz and only NaN cleaning is applied.
    """
    df = df.copy()
    if "TIMESTAMP" in df.columns:
        df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
        df = df.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")
        df = df.set_index(pd.to_timedelta(df["TIMESTAMP"], unit="s"))
        rule = f"{int(1 / CFG.get('target_hz', 1) * 1000)}ms"
        df = df[features + ["label"]].resample(rule).mean().interpolate("linear")
        df = df.reset_index(drop=True)
    else:
        df = df[features + ["label"]].reset_index(drop=True)

    # Coerce any non-numeric entries (e.g. "NODATA", dashes) to NaN
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[features] = df[features].ffill().bfill()
    df.dropna(inplace=True)
    print(f"[preprocess] 1/4 resample   -> {len(df):,} rows  (dropped {before - len(df):,} NaN)")
    return df.reset_index(drop=True)


def pp_iqr_clip(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Step 2 - IQR clipping.
    Clips extreme sensor spikes to [Q1 - k*IQR, Q3 + k*IQR].
    Preserves the distribution shape while removing hardware noise artifacts.
    """
    k = CFG.get("iqr_multiplier", 1.5)
    clip_stats, total_clipped = {}, 0
    for col in features:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        clip_stats[col] = {"lo": round(lo, 4), "hi": round(hi, 4), "clipped": int(n_clipped)}
        total_clipped += n_clipped

    print(f"[preprocess] 2/4 IQR clip   -> {total_clipped:,} values clipped (k={k})")
    with open(OUT / "clip_stats.json", "w") as f:
        json.dump(clip_stats, f, indent=2)
    return df


def pp_rolling_statistics(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Step 3 - Rolling statistics.
    Appends a rolling mean and rolling std column for every feature.
    Captures temporal dynamics (e.g. sustained high RPM vs a momentary spike)
    that raw instantaneous readings miss.
    """
    w, new_cols = CFG.get("rolling_window", 5), []
    for col in features:
        df[f"{col}_RMEAN"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_RSTD"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
        new_cols.extend([f"{col}_RMEAN", f"{col}_RSTD"])
    extended = features + new_cols
    print(f"[preprocess] 3/4 rolling    -> +{len(new_cols)} columns -> {len(extended)} total (w={w}s)")
    return df, extended


def pp_correlation_filter(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Step 4 - Drop highly correlated features (above CFG threshold).
    Keeps one representative from each correlated cluster.
    Reduces redundancy so both models learn from independent signals.
    """
    threshold = CFG.get("corr_threshold", 0.95)
    corr  = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop  = [col for col in upper.columns if any(upper[col] > threshold)]
    retained = [f for f in features if f not in to_drop]
    print(f"[preprocess] 4/4 corr filter -> dropped {len(to_drop)}, retained {len(retained)}")
    if to_drop:
        print(f"             dropped: {to_drop}")

    fig, ax = plt.subplots(figsize=(max(10, len(retained)//2), max(8, len(retained)//2)))
    sns.heatmap(df[retained].corr(), ax=ax, cmap="coolwarm", center=0,
                annot=False, linewidths=0.4, cbar_kws={"shrink": 0.6})
    ax.set_title("Feature Correlation Matrix (post-filter)")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    return df, retained


def preprocess(df: pd.DataFrame, raw_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    print("\n-- Preprocessing --------------------------------------")
    df = pp_resample(df, raw_features)
    df = pp_iqr_clip(df, raw_features)
    df, extended = pp_rolling_statistics(df, raw_features)
    gc.collect()
    df, final    = pp_correlation_filter(df, extended)
    print(f"[preprocess] final feature count: {len(final)}")
    print("-- Preprocessing complete ------------------------------\n")
    return df, final


# -------------------------------------------------------------
# SECTION 3 - SEQUENCE CONSTRUCTION
# -------------------------------------------------------------
def split_normal(X: np.ndarray, test_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:]


def make_sequences(data: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    """Build sliding-window sequences with configurable stride."""
    indices = list(range(0, len(data) - seq_len + 1, stride))
    out = np.empty((len(indices), seq_len, data.shape[1]), dtype=data.dtype)
    for j, i in enumerate(indices):
        out[j] = data[i : i + seq_len]
    return out


def inject_synthetic_anomalies(data: np.ndarray, features: list[str], ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Injects realistic OBD-II fault patterns into a subset of test sequences.

    Five fault types based on real failure signatures:
      1. Engine Misfire          – RPM oscillation + ENGINE_LOAD bump
      2. Coolant System Failure  – COOLANT_TEMP linear ramp
      3. O2 Sensor / Fuel Trim   – STFT & LTFT drift + FUEL_AIR_EQUIV_RATIO perturbation
      4. Throttle Position Fault  – THROTTLE stuck constant + ENGINE_LOAD drop
      5. Intake Manifold Anomaly – INTAKE_MANIFOLD_PRESSURE spike + TIMING_ADVANCE retard

    Parameters
    ----------
    data     : (N, seq_len, n_features) array of normal sequences
    features : list of feature names matching the column order in *data*
    ratio    : fraction of *data* to convert into anomaly samples

    Returns
    -------
    (combined_data, labels) where normal=0 and anomaly=1
    """
    def _idx(name: str) -> int | None:
        try:
            return features.index(name)
        except ValueError:
            return None

    n = int(len(data) * ratio)
    idx = np.random.choice(len(data), n, replace=False)
    anomalies = data[idx].copy()
    seq_len = data.shape[1]
    t = np.arange(seq_len, dtype=np.float32)          # time axis for ramps / sinusoids

    # Split anomaly samples roughly equally across the 5 fault types
    groups = np.array_split(np.arange(n), 5)
    fault_names = [
        "Engine Misfire",
        "Coolant System Failure",
        "O2 Sensor / Fuel Trim Drift",
        "Throttle Position Sensor Fault",
        "Intake Manifold Pressure Anomaly",
    ]
    counts = {name: 0 for name in fault_names}

    # ---- Fault 1: Engine Misfire ----
    i_rpm  = _idx("ENGINE_RPM")
    i_load = _idx("ENGINE_LOAD")
    if i_rpm is not None:
        for j in groups[0]:
            freq = np.random.uniform(0.5, 2.0)
            amp  = np.random.uniform(2.0, 4.0)
            anomalies[j, :, i_rpm] += amp * np.sin(2 * np.pi * freq * t / seq_len)
            if i_load is not None:
                anomalies[j, :, i_load] += np.random.uniform(1.5, 2.5)
        counts[fault_names[0]] = len(groups[0])
    else:
        print("[anomaly] WARNING: ENGINE_RPM not in features – skipping Engine Misfire fault")

    # ---- Fault 2: Coolant System Failure ----
    i_cool = _idx("COOLANT_TEMP")
    if i_cool is not None:
        for j in groups[1]:
            ramp_end = np.random.uniform(3.0, 5.0)
            ramp = np.linspace(0, ramp_end, seq_len).astype(np.float32)
            noise = np.random.uniform(-0.2, 0.2, size=seq_len).astype(np.float32)
            anomalies[j, :, i_cool] += ramp + noise
        counts[fault_names[1]] = len(groups[1])
    else:
        print("[anomaly] WARNING: COOLANT_TEMP not in features – skipping Coolant System Failure fault")

    # ---- Fault 3: O2 Sensor / Fuel Trim Drift ----
    i_stft = _idx("SHORT_TERM_FUEL_TRIM")
    i_ltft = _idx("LONG_TERM_FUEL_TRIM")
    i_fa   = _idx("FUEL_AIR_EQUIV_RATIO")
    if i_stft is not None and i_ltft is not None:
        for j in groups[2]:
            direction = np.random.choice([-1.0, 1.0])
            drift_mag = np.random.uniform(2.5, 4.5)
            drift = np.linspace(0, direction * drift_mag, seq_len).astype(np.float32)
            anomalies[j, :, i_stft] += drift
            anomalies[j, :, i_ltft] += drift
            if i_fa is not None:
                anomalies[j, :, i_fa] += np.random.uniform(-1.5, 1.5) if direction > 0 else np.random.uniform(-1.5, 1.5)
        counts[fault_names[2]] = len(groups[2])
    else:
        missing = [n for n, i in [("SHORT_TERM_FUEL_TRIM", i_stft), ("LONG_TERM_FUEL_TRIM", i_ltft)] if i is None]
        print(f"[anomaly] WARNING: {missing} not in features – skipping O2 Sensor / Fuel Trim Drift fault")

    # ---- Fault 4: Throttle Position Sensor Fault ----
    i_thr   = _idx("THROTTLE")
    i_speed = _idx("VEHICLE_SPEED")
    i_load4 = _idx("ENGINE_LOAD")
    if i_thr is not None:
        for j in groups[3]:
            mean_thr = anomalies[j, :, i_thr].mean()
            stuck_val = mean_thr + np.random.uniform(2.0, 3.5)
            anomalies[j, :, i_thr] = stuck_val
            if i_load4 is not None:
                anomalies[j, :, i_load4] -= np.random.uniform(1.5, 2.5)
            # VEHICLE_SPEED left untouched — the mismatch is the anomaly signal
        counts[fault_names[3]] = len(groups[3])
    else:
        print("[anomaly] WARNING: THROTTLE not in features – skipping Throttle Position Sensor Fault")

    # ---- Fault 5: Intake Manifold Pressure Anomaly ----
    i_map = _idx("INTAKE_MANIFOLD_PRESSURE")
    i_ta  = _idx("TIMING_ADVANCE")
    if i_map is not None:
        for j in groups[4]:
            spike_point = np.random.randint(0, max(1, seq_len // 2))
            spike_mag   = np.random.uniform(3.0, 5.0)
            anomalies[j, spike_point:, i_map] += spike_mag
            if i_ta is not None:
                retard_mag = np.random.uniform(2.0, 3.5)
                anomalies[j, spike_point:, i_ta] -= retard_mag
        counts[fault_names[4]] = len(groups[4])
    else:
        print("[anomaly] WARNING: INTAKE_MANIFOLD_PRESSURE not in features – skipping Intake Manifold Pressure Anomaly")

    # Clip to realistic scaled bounds
    anomalies = np.clip(anomalies, -10.0, 10.0)

    # Summary
    print(f"\n[anomaly] Injected {n} synthetic anomalies across 5 fault types:")
    for name, cnt in counts.items():
        print(f"  {name:<40s} {cnt:>5d} samples")

    return (
        np.concatenate([data, anomalies]),
        np.concatenate([np.zeros(len(data)), np.ones(n)]),
    )


# -------------------------------------------------------------
# SECTION 4 - AUTOENCODER
# -------------------------------------------------------------
def build_autoencoder(seq_len: int, n_features: int, latent_dim: int) -> Model:
    inp     = layers.Input(shape=(seq_len, n_features))
    x       = layers.LSTM(64, return_sequences=True)(inp)
    x       = layers.Dropout(0.2)(x)
    x       = layers.LSTM(32, return_sequences=False)(x)
    encoded = layers.Dense(latent_dim, activation="relu")(x)
    x       = layers.RepeatVector(seq_len)(encoded)
    x       = layers.LSTM(32, return_sequences=True)(x)
    x       = layers.Dropout(0.2)(x)
    x       = layers.LSTM(64, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(n_features))(x)
    model   = Model(inp, decoded, name="lstm_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(CFG["ae_lr"]), loss="mse")
    return model


def _batched_recon_error(model: Model, X: np.ndarray, batch: int = 512) -> np.ndarray:
    """Compute per-sample MSE without allocating a full prediction copy."""
    errors = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch):
        b = X[i:i+batch]
        r = model.predict(b, verbose=0)
        errors[i:i+batch] = np.mean(np.mean((b - r) ** 2, axis=2), axis=1)
    return errors


def train_autoencoder(X_train: np.ndarray) -> tuple[Model, float]:
    model = build_autoencoder(X_train.shape[1], X_train.shape[2], CFG["ae_latent_dim"])
    model.summary()
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(str(OUT / "ae_best.keras"), save_best_only=True),
    ]
    history = model.fit(
        X_train, X_train,
        epochs=CFG["ae_epochs"], batch_size=CFG["ae_batch"],
        validation_split=0.1, callbacks=cb, verbose=1,
    )
    errors    = _batched_recon_error(model, X_train)
    threshold = float(np.percentile(errors, CFG["anomaly_pct_threshold"]))
    _plot_ae_loss(history)
    print(f"[ae] threshold (p{CFG['anomaly_pct_threshold']}): {threshold:.6f}")
    return model, threshold


def ae_reconstruction_error(model: Model, X: np.ndarray) -> np.ndarray:
    return _batched_recon_error(model, X)


# -------------------------------------------------------------
# SECTION 5 - ISOLATION FOREST
# -------------------------------------------------------------
def train_isolation_forest(X_flat: np.ndarray) -> IsolationForest:
    clf = IsolationForest(
        n_estimators=CFG["if_n_estimators"],
        contamination=CFG["if_contamination"],
        random_state=CFG["random_seed"],
        n_jobs=-1,
    )
    clf.fit(X_flat)
    print("[if] Isolation Forest trained.")
    return clf


# -------------------------------------------------------------
# SECTION 6 - EVALUATION
# -------------------------------------------------------------
def evaluate(scores: np.ndarray, labels: np.ndarray,
             model_name: str, threshold=None) -> dict:
    auc_roc = roc_auc_score(labels, scores)
    auc_pr  = average_precision_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1s      = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    thr      = threshold if threshold is not None else best_thr
    preds    = (scores >= thr).astype(int)
    cm       = confusion_matrix(labels, preds)

    metrics = {
        "model"    : model_name,
        "AUC-ROC"  : round(float(auc_roc), 4),
        "AUC-PR"   : round(float(auc_pr), 4),
        "Best-F1"  : round(float(f1s[best_idx]), 4),
        "Threshold": round(float(thr), 6),
    }
    print(f"\n[eval] {model_name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    _plot_pr_curve(precision, recall, auc_pr, model_name)
    _plot_confusion_matrix(cm, model_name)
    return metrics


# -------------------------------------------------------------
# SECTION 7 - FEATURE IMPORTANCE
# -------------------------------------------------------------

def fi_autoencoder(model: Model, X_normal: np.ndarray, features: list[str]) -> pd.Series:
    """
    Per-feature mean reconstruction error on normal training data.
    High error = model struggled to reconstruct that feature = it carries
    complex signal that matters for anomaly detection.
    Shape: (N, seq_len, n_features) -> mean over N and seq_len -> (n_features,)
    """
    per_feature_err = np.zeros(len(features), dtype=np.float64)
    batch = 512
    for i in range(0, len(X_normal), batch):
        b = X_normal[i:i+batch]
        r = model.predict(b, verbose=0)
        per_feature_err += np.sum((b - r) ** 2, axis=(0, 1))
    per_feature_err /= (len(X_normal) * X_normal.shape[1])
    return pd.Series(per_feature_err, index=features).sort_values(ascending=False)


def fi_isolation_forest(clf: IsolationForest, X_flat: np.ndarray,
                        features: list[str], seq_len: int) -> pd.Series:
    """
    Permutation importance for Isolation Forest.
    For each feature, shuffle its values across all time-step columns and
    measure the increase in mean anomaly score. Larger increase = more important.
    """
    n_features = len(features)
    baseline   = -clf.decision_function(X_flat).mean()
    importances = np.zeros(n_features)

    for fi in range(n_features):
        scores = []
        for _ in range(CFG["perm_n_repeats"]):
            feat_cols = [fi + f * n_features for f in range(seq_len)
                         if fi + f * n_features < X_flat.shape[1]]
            
            # Save original columns
            orig_cols = X_flat[:, feat_cols].copy()
            
            # Permute in place
            perm_idx  = np.random.permutation(len(X_flat))
            X_flat[:, feat_cols] = X_flat[perm_idx][:, feat_cols]
            
            # Score
            scores.append(-clf.decision_function(X_flat).mean())
            
            # Restore original columns
            X_flat[:, feat_cols] = orig_cols
        importances[fi] = np.mean(scores) - baseline

    return pd.Series(importances, index=features).sort_values(ascending=False)


def plot_feature_importance(ae_imp: pd.Series, if_imp: pd.Series):
    """
    Side-by-side horizontal bar charts for AE and IF importance.
    Both normalised 0-1 so scales are directly comparable.
    """
    ae_norm = (ae_imp - ae_imp.min()) / (ae_imp.max() - ae_imp.min() + 1e-8)
    if_norm = (if_imp - if_imp.min()) / (if_imp.max() - if_imp.min() + 1e-8)
    if_norm = if_norm.reindex(ae_norm.index).fillna(0)   # align on AE order

    n   = len(ae_norm)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n * 0.38)))

    for ax, imp, cmap, title in zip(
        axes,
        [ae_norm, if_norm],
        ["RdYlGn_r", "RdYlBu_r"],
        [
            "Autoencoder\nPer-feature reconstruction error (normalised)",
            "Isolation Forest\nPermutation importance (normalised)",
        ],
    ):
        colors = plt.colormaps[cmap](np.linspace(0.15, 0.85, n))
        bars   = ax.barh(range(n), imp.values, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(n))
        ax.set_yticklabels(imp.index, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (0-1)", fontsize=10)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlim(0, 1.18)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, imp.values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)

    fig.suptitle("Feature Importance - M4 Anomaly Detection", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[importance] saved -> {OUT}/feature_importance.png")

    pd.DataFrame({
        "feature"       : ae_norm.index,
        "ae_importance" : ae_norm.values,
        "if_importance" : if_norm.values,
    }).to_csv(OUT / "feature_importance.csv", index=False)
    print(f"[importance] ranked table -> {OUT}/feature_importance.csv")

    print("\n[importance] Top 5 - Autoencoder:")
    for feat, val in ae_norm.head(5).items():
        print(f"  {feat:<38} {val:.4f}")
    print("\n[importance] Top 5 - Isolation Forest:")
    for feat, val in if_norm.head(5).items():
        print(f"  {feat:<38} {val:.4f}")


# -------------------------------------------------------------
# SECTION 8 - EXPORT
# -------------------------------------------------------------
def export_artifacts(ae_model, ae_threshold, if_model, scaler, features, results):
    ae_model.save(OUT / "autoencoder.keras")
    joblib.dump(if_model, OUT / "isolation_forest.pkl")
    joblib.dump(scaler,   OUT / "scaler.pkl")
    meta = {
        "features"        : features,
        "sequence_len"    : CFG["sequence_len"],
        "ae_threshold"    : float(ae_threshold),
        "if_contamination": CFG["if_contamination"],
        "results"         : results,
    }
    with open(OUT / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[export] all artifacts -> {OUT}/")
    for name in ["autoencoder.keras", "isolation_forest.pkl", "scaler.pkl", "model_meta.json"]:
        print(f"  {name}")


# -------------------------------------------------------------
# PLOT HELPERS
# -------------------------------------------------------------
def _plot_ae_loss(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_title("Autoencoder Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "ae_loss.png", dpi=150); plt.close(fig)


def _plot_pr_curve(precision, recall, auc_pr, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(recall, precision, where="post", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.15, step="post")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve - {name}\nAUC-PR={auc_pr:.4f}"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"pr_curve_{name.replace(' ','_')}.png", dpi=150); plt.close(fig)


def _plot_confusion_matrix(cm, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout(); fig.savefig(OUT / f"cm_{name.replace(' ','_')}.png", dpi=150); plt.close(fig)


def _plot_score_distributions(ae_scores, if_scores, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scores, title in zip(
        axes,
        [ae_scores, if_scores],
        ["Autoencoder - Reconstruction Error", "Isolation Forest - Anomaly Score"],
    ):
        ax.hist(scores[labels==0], bins=60, alpha=0.6, label="Normal",  density=True)
        ax.hist(scores[labels==1], bins=60, alpha=0.6, label="Anomaly", density=True)
        ax.set_title(title); ax.set_xlabel("Score"); ax.set_ylabel("Density")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Score Distributions - M4", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "score_distributions.png", dpi=150); plt.close(fig)


def _plot_model_comparison(results: list[dict]):
    metrics = ["AUC-ROC", "AUC-PR", "Best-F1"]
    x, width = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        bars = ax.bar(x + i * width, [r[m] for m in metrics], width, label=r["model"])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Model Comparison - M4 Anomaly Detection")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=150); plt.close(fig)


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    print("=" * 65)
    print("M4 - Anomaly Detection Pipeline")
    print("=" * 65)

    # 1. Load
    raw_df       = load_dataset()
    raw_features = resolve_features(raw_df)

    # 2. Preprocess
    df, features = preprocess(raw_df, raw_features)
    del raw_df; gc.collect()  # free ~50 MB

    # 3. Split + Scale (prevent data leakage)
    X_all_unscaled = df[features].values
    del df; gc.collect()  # free DataFrame, keep only numpy array
    X_train_unscaled, X_test_unscaled = split_normal(X_all_unscaled, CFG["test_ratio"])

    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_unscaled).astype(np.float32)
    X_test_raw  = scaler.transform(X_test_unscaled).astype(np.float32)

    # 4. Sequences
    seq_len  = CFG["sequence_len"]
    X_train  = make_sequences(X_train_raw, seq_len, stride=3)
    X_test_n = make_sequences(X_test_raw,  seq_len)
    X_test, y_test = inject_synthetic_anomalies(X_test_n, features, ratio=0.2)
    print(f"[seq] train: {len(X_train):,}  |  test normal: {int((y_test==0).sum()):,}  anomaly: {int((y_test==1).sum()):,}")

    # 5. Autoencoder
    print("\n-- Autoencoder -----------------------------------------")
    ae_model, ae_threshold = train_autoencoder(X_train)
    ae_scores  = ae_reconstruction_error(ae_model, X_test)
    ae_results = evaluate(ae_scores, y_test, "Autoencoder", threshold=ae_threshold)

    # 6. Isolation Forest
    print("\n-- Isolation Forest ------------------------------------")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test),  -1)
    if_model   = train_isolation_forest(X_train_flat)
    if_scores  = -if_model.decision_function(X_test_flat)
    if_results = evaluate(if_scores, y_test, "Isolation Forest")

    # 7. Feature Importance
    print("\n-- Feature Importance ----------------------------------")
    print("[importance] AE - computing per-feature reconstruction error...")
    ae_imp = fi_autoencoder(ae_model, X_train, features)

    print(f"[importance] IF - permutation importance ({CFG['perm_n_repeats']} repeats)...")
    if_imp = fi_isolation_forest(if_model, X_train_flat, features, seq_len)

    plot_feature_importance(ae_imp, if_imp)

    # 8. Summary plots
    all_results = [ae_results, if_results]
    _plot_score_distributions(ae_scores, if_scores, y_test)
    _plot_model_comparison(all_results)

    # 9. Export
    export_artifacts(ae_model, ae_threshold, if_model, scaler, features, all_results)

    # 10. Final print
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Model':<22} {'AUC-ROC':>9} {'AUC-PR':>9} {'Best-F1':>9}"
    print(header); print("-" * len(header))
    for r in all_results:
        print(f"{r['model']:<22} {r['AUC-ROC']:>9} {r['AUC-PR']:>9} {r['Best-F1']:>9}")
    print("=" * 65)
    print(f"\nAll outputs -> {OUT.resolve()}")
    print("\nKey output files:")
    print("  feature_importance.png    <- which features drive anomaly detection")
    print("  feature_importance.csv    <- full ranked table (AE + IF side by side)")
    print("  correlation_matrix.png    <- feature correlation after filtering")
    print("  model_comparison.png      <- AE vs IF metrics bar chart")
    print("  clip_stats.json           <- IQR clipping bounds per feature")
    print("  autoencoder.keras         <- deploy to Raspberry Pi")
    print("  isolation_forest.pkl      <- deploy to Raspberry Pi")
    print("  model_meta.json           <- threshold + feature list for inference")


if __name__ == "__main__":
    main()