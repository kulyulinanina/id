import argparse, yaml, pandas as pd, joblib, json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(p.parse_args().config, encoding="utf-8"))

    df = pd.read_csv(cfg["paths"]["clean"])
    X, y = df.drop(columns=["target"]), df["target"]

    model = joblib.load(cfg["paths"]["model"])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size   = cfg["training"]["test_size"],
        random_state= cfg["training"]["random_state"],
        stratify    = y
    )

    y_pred = model.predict(Xte)
    m = dict(
        accuracy = round(accuracy_score (yte, y_pred), 4),
        precision= round(precision_score(yte, y_pred), 4),
        recall   = round(recall_score   (yte, y_pred), 4),
        f1       = round(f1_score       (yte, y_pred), 4)
    )

    Path(cfg["paths"]["metrics"]).write_text(json.dumps(m, indent=2))
    print("📊 metrics →", cfg["paths"]["metrics"], "\n", m)

if __name__ == "__main__":
    main()