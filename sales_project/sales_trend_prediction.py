"""
Sales Trend Prediction Using Machine Learning
Tech: Python, pandas, NumPy, scikit-learn, Matplotlib

Workflow:
  1. Load data (your own CSV, or auto-generated sample data)
  2. Data cleaning
  3. Feature engineering
  4. Exploratory data analysis (EDA)
  5. Train Linear Regression + Decision Tree
  6. Evaluate with RMSE and R^2, plot predicted vs actual

Usage:
  python sales_trend_prediction.py                 # uses sample data
  python sales_trend_prediction.py --csv mydata.csv
     CSV columns needed: date, sales, temperature, rainfall, humidity
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
def generate_sample_data(days=1095, seed=RANDOM_STATE):
    """Create realistic daily sales + weather data (3 years)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    day_of_year = dates.dayofyear.values

    # Weather with seasonality (Indian-style climate: hot summers, monsoon)
    temperature = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365) + rng.normal(0, 2, days)
    rainfall = np.clip(
        8 * np.sin(2 * np.pi * (day_of_year - 160) / 365) + rng.normal(0, 4, days), 0, None
    )
    humidity = np.clip(55 + 0.8 * rainfall + rng.normal(0, 5, days), 20, 100)

    # Sales = base + growth trend + weekend boost + weather effects + noise
    trend = np.linspace(0, 400, days)
    weekend = (dates.dayofweek.values >= 5) * 150
    sales = (
        1000 + trend + weekend
        + 18 * temperature      # hotter days -> more sales
        - 12 * rainfall         # rainy days -> fewer sales
        + rng.normal(0, 60, days)
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales.round(2),
            "temperature": temperature.round(1),
            "rainfall": rainfall.round(1),
            "humidity": humidity.round(1),
        }
    )

    # Add real-world messiness so the cleaning step is meaningful
    for col in ["temperature", "rainfall", "humidity"]:
        df.loc[rng.choice(days, 25, replace=False), col] = np.nan
    df.loc[rng.choice(days, 8, replace=False), "sales"] = np.nan
    df = pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data(csv_path=None):
    if csv_path:
        return pd.read_csv(csv_path)
    print("No CSV given -> using generated sample data.\n")
    return generate_sample_data()


# ----------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------
def clean_data(df):
    print("=== DATA CLEANING ===")
    print(f"Raw shape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum()}\n")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates()
    df = df.sort_values("date").reset_index(drop=True)

    # Rows without target are useless
    df = df.dropna(subset=["sales"])

    # Fill weather gaps by time interpolation, then median as a fallback
    weather_cols = ["temperature", "rainfall", "humidity"]
    df[weather_cols] = df[weather_cols].interpolate(method="linear", limit_direction="both")
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())

    # Clip extreme outliers in sales (1st-99th percentile)
    low, high = df["sales"].quantile([0.01, 0.99])
    df["sales"] = df["sales"].clip(low, high)

    print(f"Clean shape: {df.shape}\n")
    return df


# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_index"] = np.arange(len(df))  # captures the long-term trend

    # Lag features (only past values -> no data leakage)
    df["sales_lag_1"] = df["sales"].shift(1)
    df["sales_lag_7"] = df["sales"].shift(7)
    df["sales_roll_7"] = df["sales"].shift(1).rolling(7).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 4. EDA
# ----------------------------------------------------------------------
def run_eda(df):
    print("=== EDA ===")
    print(df[["sales", "temperature", "rainfall", "humidity"]].describe().round(2), "\n")

    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()["sales"].drop("sales").sort_values(key=abs, ascending=False)
    print("Correlation of each feature with sales:")
    print(corr.round(3), "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Exploratory Data Analysis", fontsize=14, fontweight="bold")

    axes[0, 0].plot(df["date"], df["sales"], lw=0.7, color="tab:blue")
    axes[0, 0].plot(df["date"], df["sales"].rolling(30).mean(), color="red", lw=1.8, label="30-day avg")
    axes[0, 0].set_title("Sales over time")
    axes[0, 0].legend()

    axes[0, 1].scatter(df["temperature"], df["sales"], s=6, alpha=0.5, color="tab:orange")
    axes[0, 1].set_title("Temperature vs Sales")
    axes[0, 1].set_xlabel("Temperature (C)")

    axes[1, 0].scatter(df["rainfall"], df["sales"], s=6, alpha=0.5, color="tab:green")
    axes[1, 0].set_title("Rainfall vs Sales")
    axes[1, 0].set_xlabel("Rainfall (mm)")

    df.groupby("month")["sales"].mean().plot(kind="bar", ax=axes[1, 1], color="tab:purple")
    axes[1, 1].set_title("Average sales by month")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=130)
    plt.close()
    print("Saved: eda_plots.png\n")


# ----------------------------------------------------------------------
# 5 & 6. MODELING + EVALUATION
# ----------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<20} RMSE = {rmse:8.2f} | R2 = {r2:.4f}")
    return rmse, r2


def train_and_compare(df):
    features = [
        "temperature", "rainfall", "humidity",
        "month", "day_of_week", "is_weekend", "day_index",
        "sales_lag_1", "sales_lag_7", "sales_roll_7",
    ]
    X, y = df[features], df["sales"]

    # Time-based split: train on the past, test on the most recent 20%
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_dates = df["date"].iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE),
    }

    print("=== MODEL EVALUATION (test set) ===")
    results, preds = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
        results[name] = evaluate(name, y_test, preds[name])

    best = min(results, key=lambda k: results[k][0])
    print(f"\nBest model (lowest RMSE): {best}\n")

    # Feature importance / coefficients
    lr = models["Linear Regression"]
    dt = models["Decision Tree"]
    print("Decision Tree feature importance:")
    print(pd.Series(dt.feature_importances_, index=features).sort_values(ascending=False).round(3), "\n")

    plot_results(test_dates, y_test, preds, results, best)
    return models, results


def plot_results(dates, y_test, preds, results, best):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Sales Trend Prediction: Predicted vs Actual", fontsize=14, fontweight="bold")

    colors = {"Linear Regression": "tab:red", "Decision Tree": "tab:green"}
    for ax, (name, y_pred) in zip(axes[0], preds.items()):
        ax.plot(dates, y_test.values, label="Actual", color="black", lw=1.2)
        ax.plot(dates, y_pred, label="Predicted", color=colors[name], lw=1.2, alpha=0.85)
        rmse, r2 = results[name]
        ax.set_title(f"{name}\nRMSE={rmse:.1f}, R2={r2:.3f}")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)

    for ax, (name, y_pred) in zip(axes[1], preds.items()):
        ax.scatter(y_test, y_pred, s=10, alpha=0.6, color=colors[name])
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "k--", lw=1, label="Perfect prediction")
        ax.set_xlabel("Actual sales")
        ax.set_ylabel("Predicted sales")
        ax.set_title(f"{name}: Actual vs Predicted")
        ax.legend()

    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=130)
    plt.close()
    print("Saved: predicted_vs_actual.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Trend Prediction")
    parser.add_argument("--csv", help="Path to CSV with date, sales, temperature, rainfall, humidity")
    args = parser.parse_args()

    raw = load_data(args.csv)
    clean = clean_data(raw)
    data = engineer_features(clean)
    run_eda(data)
    train_and_compare(data)
