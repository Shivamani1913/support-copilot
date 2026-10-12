from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
csv_path=next((ROOT/"data").glob("*.csv"))
df=pd.read_csv(csv_path)

print("File:",csv_path.name)
print("Shape:",df.shape)
print("Columns:",df.columns.tolist())
print()
print(df.head(3).to_string())
print()
print(df["category"].value_counts())
