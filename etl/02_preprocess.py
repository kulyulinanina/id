import argparse, yaml, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml", help="YAML-config file")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    raw_path    = Path(cfg["paths"]["raw"])
    clean_path  = Path(cfg["paths"]["clean"])
    scaler_path = clean_path.with_suffix(".scaler.pkl")

    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run 01_load_data.py first.")

    logging.info("📖  reading raw data")
    df = pd.read_csv(raw_path)

    # базовая очистка
    df = df.drop(columns=["id", "diagnosis"])
    if df.isna().any().any():
        logging.info("🔧  filling NaNs with column means")
        df = df.fillna(df.mean(numeric_only=True))

    # масштабирование признаков
    X = df.drop(columns=["target"])
    scaler = StandardScaler()
    df[X.columns] = scaler.fit_transform(X)

    # сохранение результатов
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    joblib.dump(scaler, scaler_path)

    logging.info(f"✅ cleaned data  → {clean_path}")
    logging.info(f"✅ scaler model → {scaler_path}")

if __name__ == "__main__":
    main()