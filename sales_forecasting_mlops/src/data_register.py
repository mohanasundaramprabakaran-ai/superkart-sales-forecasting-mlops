from pathlib import Path
import pandas as pd

DATA_PATH = Path("sales_forecasting_mlops/data/SuperKart.csv")

EXPECTED_COLUMNS = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Store_Sales_Total",
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Commit SuperKart.csv to the repository first."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Product_Store_Sales_Total"].isna().any():
        raise ValueError("Target column contains missing values.")

    print("Dataset registration and validation successful.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate rows:", df.duplicated().sum())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSales target summary:")
    print(df["Product_Store_Sales_Total"].describe())

if __name__ == "__main__":
    main()
