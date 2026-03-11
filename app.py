import os
import joblib
import pickle
from pickle import UnpicklingError
from typing import List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Smart Product Return Predictor",  layout="wide")
st.title(" Smart Product Return Predictor")
st.markdown("A simple, PowerBI-style dashboard + prediction UI for product return risk. "
            "Upload your engineered_order_dataset.csv, and ensure return_risk_model.pkl "
            "and model_features.pkl are in the same folder.")

DATA_PATH = "engineered_order_dataset.csv"
MODEL_PATH = "return_risk_model.pkl"
FEATURES_PATH = "model_features.pkl"   

@st.cache_data
def safe_load_data(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str)  
    for c in df.columns:
        col = df[c].str.strip()
        if col.replace('-', '').str.match(r'^\d+(\.\d+)?$').all():
            df[c] = pd.to_numeric(col, errors='coerce')
    # parse Date if present (common format dd/mm/yyyy)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    return df

@st.cache_resource
def load_model_and_features(model_path: str, feat_path: str):
    model = None
    features = None
    # load model (prefer joblib)
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            except UnpicklingError:
                st.warning("Model exists but could not be unpickled (version mismatch).")
            except Exception as e:
                st.warning(f"Error loading model via pickle: {e}")
    else:
        st.info("No model file found; prediction will be disabled until you train and save 'return_risk_model.pkl'.")

    # load features list if present
    if os.path.exists(feat_path):
        try:
            features = joblib.load(feat_path)
            # sometimes it's just a pickle list
            if not isinstance(features, list):
                # try reading via pickle
                with open(feat_path, "rb") as f:
                    features = pickle.load(f)
        except Exception:
            try:
                with open(feat_path, "rb") as f:
                    features = pickle.load(f)
            except Exception:
                features = None
    else:
        # fallback: if model has attribute feature_names_in_
        if hasattr(model, "feature_names_in_"):
            try:
                features = list(model.feature_names_in_)
            except Exception:
                features = None

    return model, features

def find_revenue_column(df: pd.DataFrame) -> Optional[str]:
    possible = ["Final Revenue", "FinalRevenue", "Total Revenue", "TotalRevenue", "Overall Revenue", "OverallRevenue"]
    for p in possible:
        if p in df.columns:
            return p
    # fallback to first numeric
    num_cols = df.select_dtypes(include=[np.number]).columns
    return num_cols[0] if len(num_cols) > 0 else None

def safe_median_or_zero(series):
    try:
        return float(series.median())
    except Exception:
        return 0.0

def build_input_row_for_model(user_inputs: dict, model_features: Optional[List[str]], df: pd.DataFrame):
    """
    Build a DataFrame row aligned with model_features.
    If model_features is None, return a row using only user_inputs columns.
    Handles one-hot style features for Category if features have Category_<value>.
    """
    if model_features is None:
        return pd.DataFrame([user_inputs])

    # start with zeros
    row = {f: 0 for f in model_features}

    # direct mapping for numeric features present in model_features
    for k, v in user_inputs.items():
        if k in row:
            row[k] = v
        else:
            # handle one-hot like Category_<val> patterns
            if k == "Category":
                # try to map Category to one-hot columns
                cat_val = str(v)
                # look for exact one-hot column name matches
                found = False
                for feat in model_features:
                    # common encodings: 'Category_Value', 'Category Value', 'Category_Value_X'
                    if feat.lower().startswith("category") and cat_val.lower() in feat.lower():
                        row[feat] = 1
                        found = True
                # if not found, try mapping Category -> numeric index if 'Category' itself is feature
                if not found and "Category" in row:
                    # create mapping based on unique values in df
                    unique = list(df["Category"].dropna().unique()) if "Category" in df.columns else []
                    if cat_val in unique:
                        row["Category"] = unique.index(cat_val)
                    else:
                        row["Category"] = 0
            else:
                # if model expects 'Price Reductions' but user provided 'Total Revenue' etc.
                # attempt to map by substring
                for feat in model_features:
                    if k.lower() in feat.lower():
                        row[feat] = v
    # for any numeric features present in df but missing in row, use median
    for feat in model_features:
        if feat in row:
            continue
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            row[feat] = safe_median_or_zero(df[feat])
        # else keep 0
    return pd.DataFrame([row])

# ---------------- Load resources ----------------
df = safe_load_data(DATA_PATH)
model, model_features = load_model_and_features(MODEL_PATH, FEATURES_PATH)

# If dataset absent - stop
if df is None:
    st.error(f"Dataset not found: {DATA_PATH}. Please place 'engineered_order_dataset.csv' in the app folder.")
    st.stop()

# ---------------- Sidebar filters (simple) ----------------
st.sidebar.header("Filters")
revenue_col = find_revenue_column(df)
if "Date" in df.columns:
    min_date, max_date = df["Date"].min(), df["Date"].max()
    start_date, end_date = st.sidebar.date_input("Date range", value=[min_date.date(), max_date.date()])
    # filter df copy
    df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
else:
    df_filtered = df.copy()

if "Category" in df_filtered.columns:
    cats = sorted(df_filtered["Category"].dropna().unique())
    chosen = st.sidebar.multiselect("Category", options=cats, default=cats[:5] if len(cats) > 5 else cats)
    if chosen:
        df_filtered = df_filtered[df_filtered["Category"].isin(chosen)]

# ---------------- KPI cards ----------------
st.subheader("Dataset Overview — Key Metrics")
col1, col2, col3 = st.columns(3)
total_revenue = df_filtered[revenue_col].sum() if revenue_col else 0
total_orders = df_filtered["Transaction ID"].nunique() if "Transaction ID" in df_filtered.columns else len(df_filtered)
if "Return_Ratio" in df_filtered.columns:
    avg_return_ratio = df_filtered["Return_Ratio"].astype(float).mean()
else:
    # try compute from refunded/purchased if available
    if ("Refunded Item Count" in df_filtered.columns) and ("Purchased Item Count" in df_filtered.columns):
        avg_return_ratio = (df_filtered["Refunded Item Count"].astype(float) / (df_filtered["Purchased Item Count"].astype(float) + 1e-6)).mean()
    else:
        avg_return_ratio = 0.0

col1.metric("💰 Total Revenue", f"{total_revenue:,.2f}")
col2.metric("🧾 Total Orders", f"{total_orders:,}")
col3.metric("📦 Avg Return Ratio", f"{avg_return_ratio:.3f}")

st.markdown("---")

# ---------------- Visuals ----------------
st.subheader("Visual Insights")
# Revenue by Category
if "Category" in df_filtered.columns and revenue_col:
    rev_by_cat = df_filtered.groupby("Category", as_index=False)[revenue_col].sum().sort_values(by=revenue_col, ascending=False)
    fig = px.bar(rev_by_cat, x="Category", y=revenue_col, title="Revenue by Category", labels={revenue_col: "Revenue"})
    st.plotly_chart(fig, width='stretch')
else:
    st.info("Category or revenue column missing — cannot show revenue-by-category.")

# Return ratio by category
if "Category" in df_filtered.columns and "Return_Ratio" in df_filtered.columns:
    rr = df_filtered.groupby("Category", as_index=False)["Return_Ratio"].mean().sort_values(by="Return_Ratio", ascending=False)
    fig2 = px.bar(rr, x="Category", y="Return_Ratio", title="Avg Return Ratio by Category")
    st.plotly_chart(fig2, width='stretch')
else:
    st.info("Return_Ratio not found — ensure engineered dataset has Return_Ratio column.")

# Refund distribution
# ------------------- Refund Distribution Pie (FORCED DISPLAY) -------------------

st.subheader("Refund Distribution by Category (Pie Chart)")

# Step 1: Try to intelligently detect a refund-related column
possible_refund_cols = [
    "Refunds",
    "Refund Amount",
    "Refund_Amount",
    "Total Refund",
    "Refund Value",
    "Refunded Item Count"
]

refund_col = None
for col in possible_refund_cols:
    if col in df_filtered.columns:
        refund_col = col
        break

# Step 2: If still not found, create a dummy refund column so pie always shows
if refund_col is None:
    st.warning("⚠ No refund column found in dataset. Creating demo refund values so pie chart always displays.")
    df_filtered["Demo_Refunds"] = np.random.randint(10, 100, size=len(df_filtered))
    refund_col = "Demo_Refunds"

# Step 3: Ensure category exists
if "Category" in df_filtered.columns:
    refunds = df_filtered.groupby("Category", as_index=False)[refund_col].sum()

    # Step 4: Force minimum non-zero values if everything is zero
    if refunds[refund_col].sum() == 0:
        st.warning("⚠ Refund values are all zero. Showing small values to keep pie chart visible.")
        refunds[refund_col] = np.random.randint(10, 50, size=len(refunds))

    # Step 5: Display Pie Chart (Guaranteed)
    fig3 = px.pie(
        refunds,
        names="Category",
        values=refund_col,
        title="Refund Distribution by Category"
    )

    st.plotly_chart(fig3, width='stretch')

else:
    st.error("❌ Category column missing. Pie chart cannot be displayed.")

st.markdown("---")

# ---------------- Prediction Input ----------------
st.subheader("Predict Return Risk — Simple Input")
st.write("Provide minimal order details and click Predict. The app will use the trained model (if present).")

# Defaults from dataset medians
default_category = df_filtered["Category"].dropna().unique()[0] if "Category" in df_filtered.columns else ""
default_qty = int(df_filtered["Final Quantity"].dropna().median()) if "Final Quantity" in df_filtered.columns else 1
default_revenue = float(df_filtered[revenue_col].dropna().median()) if revenue_col and revenue_col in df_filtered.columns else 0.0

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    if "Category" in df_filtered.columns:
        category_input = c1.selectbox("Category", options=sorted(df_filtered["Category"].dropna().unique()), index=0)
    else:
        category_input = c1.text_input("Category (free text)", value=str(default_category))
    final_qty_input = c2.number_input("Final Quantity", min_value=1, value=int(default_qty))
    total_revenue_input = c3.number_input("Total Revenue", min_value=0.0, value=float(default_revenue), format="%.2f")
    submitted = st.form_submit_button("🔮 Predict Return Risk")

if submitted:
    # ---------------- Build user input ----------------
    user_inputs = {
        "Category": category_input,
        "Final Quantity": final_qty_input,

        # Use realistic values instead of extreme ones
        "Total Revenue": total_revenue_input,
        "Final Revenue": total_revenue_input,        
        "Price Reductions": 0,
        "Refunds": 0,
        "Return_Ratio": 0.01
    }

    # ---------------- Align with model features ----------------
    input_df = build_input_row_for_model(user_inputs, model_features, df)

    if model is None:
        st.warning("Model not loaded. Please run training script to generate 'return_risk_model.pkl'.")
    else:
        try:
            input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # ✅ ✅ ✅ REAL PROBABILITY-BASED PREDICTION
            probs = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            high_index = classes.index("High Risk") if "High Risk" in classes else 1
            low_index = classes.index("Low Risk") if "Low Risk" in classes else 0

            high_risk_prob = probs[high_index]
            low_risk_prob = probs[low_index]

            st.caption(f"Confidence → High Risk: {high_risk_prob:.2f}, Low Risk: {low_risk_prob:.2f}")

            # ✅ BUSINESS THRESHOLD
            if high_risk_prob >= 0.70:
                risk = "High Risk"
            elif high_risk_prob <= 0.45:
                risk = "Low Risk"
            else:
                risk = "Medium Risk"

            # ---------------- Prediction Result ----------------
            st.markdown("### 🔮 Prediction Result")

            if risk == "High Risk":
                st.error(f"🚨 Predicted: {risk}")
            else:
                st.success(f"✅ Predicted: {risk}")

            # ---------------- ✅ AUTOMATED RECOMMENDATIONS ----------------
            st.markdown("### 💡 Seller Recommendations (Automated)")

            if risk == "High Risk":
                st.warning("High return risk detected. Immediate seller actions recommended:")
                st.markdown("""
                ✅ Improve product quality checks before shipping  
                ✅ Add clearer product images & descriptions  
                ✅ Improve packaging to avoid transit damage  
                ✅ Reduce delivery delays  
                ✅ Tighten refund & return policy abuse  
                """)
            else:
                st.success("Low return risk detected. Maintain and scale these good practices:")
                st.markdown("""
                ✅ Maintain current quality standards  
                ✅ Replicate this category's strategy in other products  
                ✅ Introduce loyalty offers & bundle sales  
                ✅ Scale inventory confidently for this category  
                """)

            # ---------------- ✅ CATEGORY-BASED HISTORICAL INSIGHT ----------------
            if "Category" in df.columns:
                cat = category_input
                cat_df = df[df["Category"] == cat]

                if len(cat_df) > 0:
                    st.markdown("---")
                    st.markdown(f"### 📊 Historic Stats for Category: *{cat}*")

                    # Return ratio insight
                    if "Return_Ratio" in df.columns:
                        cat_rr = cat_df["Return_Ratio"].astype(float).mean()
                        dataset_rr = df["Return_Ratio"].astype(float).mean()

                        st.write(f"• Average return ratio: *{cat_rr:.3f}*")
                        st.write(f"• Dataset average: *{dataset_rr:.3f}*")

                        if cat_rr > dataset_rr:
                            st.error("⚠ This category performs WORSE than average in returns.")
                        else:
                            st.success("✅ This category performs BETTER than average in returns.")

                    # Revenue insight
                    if revenue_col in df.columns:
                        cat_revenue = cat_df[revenue_col].sum()
                        st.write(f"• Total revenue (historical): *{cat_revenue:,.2f}*")

                else:
                    st.info("No past data available for this category.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Developed by Nikhila D — Streamlit dashboard for Smart Product Return Risk Predictor")