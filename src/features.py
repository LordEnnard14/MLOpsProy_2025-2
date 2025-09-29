import pandas as pd
import argparse, os

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Normaliza nombres de columnas (quita espacios)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # 2) Quita identificador que no aporta al modelado
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 3) Convierte TotalCharges a numérico (hay filas vacías)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Elimina filas donde quedó NaN tras la conversión
        df = df.dropna(subset=["TotalCharges"])

    # 4) Convierte el target Churn a binario 0/1
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int64")

    return df

def main(infile: str, outfile: str) -> None:
    df = pd.read_csv(infile)
    df = clean_telco(df)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)

    # Info rápida en consola
    print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
    if "Churn" in df.columns:
        dist = df["Churn"].value_counts(normalize=True).round(3).to_dict()
        print(f"Distribución de Churn: {dist}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/raw/Telco-Customer-Churn.csv")
    ap.add_argument("--outfile", default="data/processed/telco_clean.csv")
    args = ap.parse_args()
    print("Cambio de prueba")
    main(args.infile, args.outfile)
