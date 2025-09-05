from flask import Flask, request, send_from_directory, render_template_string
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Paths
APP_ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(APP_ROOT, "car price.xlsx")   # your dataset
MODEL_PATH = os.path.join(APP_ROOT, "model.joblib")

app = Flask(__name__, static_folder='.', template_folder='.')

# ----------------- Data Loading & Cleaning -----------------
def load_and_clean_data(path):
    df = pd.read_excel(path, engine="openpyxl")

    # Map possible column names
    col_map = {}
    for c in df.columns:
        name = c.lower()
        if "brand" in name or "make" in name:
            col_map[c] = "brand"
        elif "model" in name or "name" in name:
            col_map[c] = "model"
        elif "year" in name:
            col_map[c] = "year"
        elif "engine" in name or "cc" in name:
            col_map[c] = "engine"
        elif "trans" in name:
            col_map[c] = "transmission"
        elif "fuel" in name:
            col_map[c] = "fuel_type"
        elif "price" in name or "selling" in name:
            col_map[c] = "price"

    df = df.rename(columns=col_map)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Debug: print final column names
    print("✅ Columns after cleaning:", df.columns.tolist())

    # Ensure required columns exist
    for col in ["brand","model","year","engine","transmission","fuel_type","price"]:
        if col not in df.columns:
            df[col] = None

    # Convert numeric
    df["engine"] = pd.to_numeric(df["engine"], errors="coerce")
    if df["engine"].notna().any() and df["engine"].max(skipna=True) <= 20:
        df["engine"] = df["engine"] * 1000  # liters → cc
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Debug shape before dropna
    print("🔍 Shape before dropna:", df.shape)

    # Drop rows only if price is missing
    df = df.dropna(subset=["price"], how="any")

    # Debug shape after dropna
    print("🔍 Shape after dropna:", df.shape)
    print(df.head())

    # Clean text columns safely
    for c in ["brand","model","transmission","fuel_type"]:
        if c in df.columns and pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    return df

# ----------------- Model Training -----------------
def train_model(df):
    X = df[["brand","model","year","engine","transmission","fuel_type"]]  # mileage removed
    y = df["price"]

    cat_cols = ["brand","model","transmission","fuel_type"]
    num_cols = ["year","engine"]

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])
    model = Pipeline([
        ("pre", preproc),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

# ----------------- Load or Train -----------------
if os.path.exists(MODEL_PATH):
    print("✅ Loaded saved model.")
    model_pipe = joblib.load(MODEL_PATH)
else:
    print("⚙️ Training model from Excel data...")
    df_data = load_and_clean_data(DATA_PATH)
    model_pipe = train_model(df_data)
    print("✅ Model trained and saved.")

# ----------------- Routes -----------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        brand = request.form.get("brand")
        model_name = request.form.get("model")
        year = int(request.form.get("year"))
        engine = float(request.form.get("engine"))
        transmission = request.form.get("transmission")
        fuel = request.form.get("fuel_type")

        X_new = pd.DataFrame([{
            "brand": brand,
            "model": model_name,
            "year": year,
            "engine": engine,
            "transmission": transmission,
            "fuel_type": fuel
        }])

        price = model_pipe.predict(X_new)[0]
        return render_template_string(f"""
            <h2>Predicted Used Car Price: ₹{price:,.0f}</h2>
            <p><a href="/">Go Back</a></p>
        """)
    except Exception as e:
        return f"<h3>Error: {e}</h3>"

# ----------------- Run -----------------
if __name__ == "__main__":
    print("🚀 Flask app running → http://127.0.0.1:5000/")
    app.run(debug=True)
