import argparse, yaml, pandas as pd, joblib, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    clean_path  = Path(cfg["paths"]["clean"])
    model_path  = Path(cfg["paths"]["model"])
    metrics_path = Path(cfg["paths"]["metrics"])

    df = pd.read_csv(clean_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --- оценки ---
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }

    # --- сохранение ---
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("✔ model  →", model_path)
    print("✔ metrics→", metrics_path)
    print("📊", metrics)

if __name__ == "__main__":
    main()