from preprocess import load_and_preprocess_metadata

X, y = load_and_preprocess_metadata('data/dermatology.csv')
print("Shape of features:", X.shape)
print("Shape of labels:", y.shape)
