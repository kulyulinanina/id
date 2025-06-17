import argparse, yaml, pandas as pd
from pathlib import Path

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(argp.parse_args().config, encoding="utf-8"))

    raw = Path(cfg["paths"]["raw"])
    if not raw.exists():
        raise FileNotFoundError(f"{raw} not found. Сначала запусти 01_load_data.py")

    df = pd.read_csv(raw)

    out = Path("results/eda.txt")
    out.parent.mkdir(exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("=== df.info() ===\n")
        df.info(buf=f)
        f.write("\n\n=== df.describe() ===\n")
        f.write(df.describe().to_string())
    print("✔ EDA report →", out)

if __name__ == "__main__":
    main()