import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ab_test_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

def load_data():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape} rows, columns: {list(df.columns)}")
    return df

def preprocess_data(df):
    print("Preprocessing data...")

    categorical_cols = ['test group', 'most ads day']

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_arr = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(categorical_cols))

    encoded_df.columns = [col.replace(' ', '_') for col in encoded_df.columns]

    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df.reset_index(drop=True)], axis=1)

    if df['converted'].dtype == 'bool':
        df['converted'] = df['converted'].astype(int)

    print(f"After encoding, data shape: {df.shape}")
    return df

def split_data(df, test_size=0.3, random_state=42):
    print(f"Splitting data into train/test with test_size={test_size}")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['converted'])
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def save_split(train_df, test_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train and test datasets saved to:\n - {train_path}\n - {test_path}")

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    train_df, test_df = split_data(df)
    save_split(train_df, test_df)