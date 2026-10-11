import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH = "/content/drive/MyDrive/Data/SuperKart (1).csv"
OUTPUT_DIR = Path("sales_forecasting_mlops/artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

target = "Product_Store_Sales_Total"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found.")

df.drop(
    columns=["Product_Id", "Store_Id"],
    inplace=True,
    errors="ignore"
)

X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv(OUTPUT_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(OUTPUT_DIR / "Xtest.csv", index=False)
ytrain.to_frame(name=target).to_csv(
    OUTPUT_DIR / "ytrain.csv", index=False
)
ytest.to_frame(name=target).to_csv(
    OUTPUT_DIR / "ytest.csv", index=False
)

print("Data preparation completed successfully!")
print("Training rows:", len(Xtrain))
print("Testing rows:", len(Xtest))
print("Features:", Xtrain.shape[1])
print("Saved files:", list(OUTPUT_DIR.iterdir()))
