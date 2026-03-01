
retail-analytics-system

retail-analytics-system/
│
├── data/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_regression.py
│   ├── train_clustering.py
│
├── models/
│
├── app/
│   ├── app.py
│
├── requirements.txt
└── README.md


SRC-Preprocess.py

import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.dropna()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df


def save_data(df, path):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = load_data("data/raw_data.csv")
    df = clean_data(df)
    save_data(df, "data/cleaned_data.csv")
 
src/feature_engineering.py

import pandas as pd


def create_time_features(df):
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Day"] = df["Order Date"].dt.day
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = create_time_features(df)
    df.to_csv("data/cleaned_data.csv", index=False)
    
src/train_regression.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(df):
    X = df[["Month", "Quantity"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model Performance:")
    print("MAE:", mae)
    print("R2 Score:", r2)

    with open("models/sales_model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    train_model(df)

    
 src/train_clustering.py
 
import pandas as pd
import pickle
from sklearn.cluster import KMeans


def train_clustering(df):
    customer_data = df.groupby("Customer ID").agg(
        {
            "Sales": "sum",
            "Quantity": "count"
        }
    ).reset_index()

    model = KMeans(n_clusters=3, random_state=42)
    customer_data["Cluster"] = model.fit_predict(
        customer_data[["Sales", "Quantity"]]
    )

    with open("models/clustering_model.pkl", "wb") as f:
        pickle.dump(model, f)

    customer_data.to_csv("data/customer_segments.csv", index=False)

    print("Clustering completed and saved.")


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    train_clustering(df)

    
 app/app.py
 
import streamlit as st
import pickle
import numpy as np

with open("../models/sales_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Retail Sales Forecasting Dashboard")

st.write("Enter details to predict sales:")

month = st.number_input("Month (1-12)", min_value=1, max_value=12)
quantity = st.number_input("Quantity", min_value=1)

if st.button("Predict Sales"):
    input_data = np.array([[month, quantity]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales: ₹{prediction[0]:.2f}")
