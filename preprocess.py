#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Added import statement
from scipy.stats import zscore

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def remove_outliers(df, zscore_threshold=3):
    df["price_zscore"] = zscore(df["price"])
    outliers = df[df["price_zscore"].abs() > zscore_threshold].copy()
    return outliers

def analyze_data(df):
    # Descriptive statistics
    print("Descriptive Statistics:")
    print(df.describe())

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Unique values in the 'year' column
    print("\nUnique Years:")
    print(df["year"].unique())

    # Drop rows with year == 2060
    df.drop(df[df["year"] == 2060].index, inplace=True)

    # Maximum price for each model
    print("\nMaximum Prices for Each Model:")
    print(df.groupby("model")["price"].max().sort_values(ascending=False))

    # Unique values in the 'fuelType' column
    print("\nUnique Fuel Types:")
    print(df["fuelType"].unique())

    # Average price for each fuel type and model
    print("\nAverage Prices for Each Fuel Type and Model:")
    g = df.groupby(["fuelType", "model"])["price"].mean()
    for (fuel_type, model), avg_price in g.items():
        print(f"\nFuel Type: {fuel_type}, Model: {model}")
        print(f"Average Price: {avg_price:.2f}")

    # Show prices that are considered outliers using zscore
    print("\nOutliers:")
    outliers = remove_outliers(df)
    print(outliers[["model", "price"]])


 

if __name__ == "__main__":
    # Load the dataset
    dataset_path = "/home/bd-a1/ford.csv"
    df = load_data(dataset_path)

    # Analyze the data
    analyze_data(df)

    # Save the resulting dataframe as a new CSV file
    df.to_csv("ford_res.csv", index=False)

    print("Data preprocessing completed!")

    # Invoke the next Python file
    exec(open("/home/bd-a1/knn.py").read())


