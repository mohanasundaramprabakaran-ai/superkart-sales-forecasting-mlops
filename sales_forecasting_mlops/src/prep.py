from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("sales_forecasting_mlops/data/SuperKart.csv")
OUTPUT_DIR = Path("sales_forecasting_mlops/artifacts")
TARGET = "Product_Store_Sales_Total"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    # Remove identifier columns that are not used as model features.
    df = df.drop(
        columns=["Product_Id", "Store_Id"],
        errors="ignore"
    )

    # Fill missing feature values safely.
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(exclude="number").columns:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    Xtrain.to_csv(OUTPUT_DIR / "Xtrain.csv", index=False)
    Xtest.to_csv(OUTPUT_DIR / "Xtest.csv", index=False)
    ytrain.to_frame(name=TARGET).to_csv(
        OUTPUT_DIR / "ytrain.csv", index=False
    )
    ytest.to_frame(name=TARGET).to_csv(
        OUTPUT_DIR / "ytest.csv", index=False
    )

    print("Data preparation completed.")
    print("Training rows:", len(Xtrain))
    print("Testing rows:", len(Xtest))
    print("Features:", Xtrain.shape[1])
    print("Artifacts:", sorted(p.name for p in OUTPUT_DIR.glob("*.csv")))

if __name__ == "__main__":
    main()
