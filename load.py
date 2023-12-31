# load.py
import pandas as pd

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    dataset_path = "/home/bd-a1/ford.csv"
    df = load_dataset(dataset_path)
    print("Dataset loaded successfully!")
    # Print the loaded dataset
    print(df)
