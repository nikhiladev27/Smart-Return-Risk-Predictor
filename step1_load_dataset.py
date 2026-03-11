import pandas as pd

# Load the dataset
file_path = "order_dataset.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
except Exception as e:
    print("Error loading dataset:", e)
