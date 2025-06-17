import argparse, yaml, pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "breast-cancer-wisconsin/wdbc.data")

HEADERS = [
    "id", "diagnosis",
    *[f"{m}_{s}" for m in (
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry",
        "fractal_dimension") for s in ("mean", "se", "worst")]
]

def load(cfg):
    Path(cfg["paths"]["raw"]).parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(UCI_URL, cfg["paths"]["raw"])

    df = pd.read_csv(cfg["paths"]["raw"], header=None, names=HEADERS)
    df["target"] = (df["diagnosis"] == "M").astype(int)   # M → 1, B → 0
    df.to_csv(cfg["paths"]["raw"], index=False)
    print(f"✔ Dataset сохранён в {cfg['paths']['raw']}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(p.parse_args().config, encoding="utf-8"))
    load(cfg)

if __name__ == "__main__":
    main()