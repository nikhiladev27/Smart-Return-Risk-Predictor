# step3_feature_engineering.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load cleaned dataset
df = pd.read_csv("cleaned_order_dataset.csv")
print("Cleaned dataset loaded!")
print("Shape before feature engineering:", df.shape)

# Step 2: Encode categorical features
label_encoder = LabelEncoder()

categorical_cols = ['Item Name', 'Category', 'Courier Status', 'Fulfilment',
                    'B2B', 'Customer City', 'Customer State', 'Customer Pin Code']

for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# Step 3: Create new derived features (optional)
# Example: Return ratio feature
df['Return_Ratio'] = df['Refunded Item Count'] / (df['Purchased Item Count'] + 1)

# Step 4: Handle any infinite values
df.replace([float('inf'), -float('inf')], 0, inplace=True)

# Step 5: Save engineered dataset
df.to_csv("engineered_order_dataset.csv", index=False)

print("\ngit initFeature engineering complete!")
print("Shape after feature engineering:", df.shape)
print("Engineered dataset saved as 'engineered_order_dataset.csv'")
