# step2_data_cleaning.py
import pandas as pd

# Step 1: Load raw dataset
df = pd.read_csv("order_dataset.csv")
print("Dataset loaded for cleaning!")
print("Shape before cleaning:", df.shape)

# Step 2: Handle missing values
print("\nMissing values per column before cleaning:")
print(df.isnull().sum())

# Example: Drop rows with too many missing values (adjust as needed)
df = df.dropna(thresh=len(df.columns) - 3)

# fill remaining missing values
df = df.fillna({
    'Refunded Item Count': 0,
    'Purchased Item Count': 0
})

# Step 3: Remove duplicates
df = df.drop_duplicates()

# Step 4: Remove invalid numeric values
df = df[(df['Purchased Item Count'] >= 0) & (df['Refunded Item Count'] >= 0)]

# Step 5: Save cleaned dataset
df.to_csv("cleaned_order_dataset.csv", index=False)

print("\nData cleaning complete!")
print("Shape after cleaning:", df.shape)
print("Cleaned dataset saved as 'cleaned_order_dataset.csv'")
