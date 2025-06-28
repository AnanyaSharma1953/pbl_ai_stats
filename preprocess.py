import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_metadata(file_path):
    # Step 1: Load the dataset
    df = pd.read_csv(file_path, header=None)

    # Step 2: Replace '?' with NaN and convert to numeric
    df.replace('?', pd.NA, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Step 3: Drop fully empty columns (NOTE: you can comment this out if you want to *keep* that column!)
    # df.dropna(axis=1, how='all', inplace=True)

    # 🧠 NEW: Fill fully empty columns manually if needed
    for col in df.columns:
        if df[col].isna().all():
            df[col] = 0  # or df[col].fillna(df[col].mean()), but mean will still be NaN

    # Step 4: Separate features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Step 5: Impute missing values in features using the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Step 6: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("Shape of features:", X_scaled.shape)
    print("Shape of labels:", y.shape)

    return X_scaled, y
