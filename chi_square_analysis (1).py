from app.preprocess import load_and_preprocess_metadata
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# ✅ Load data
X, y = load_and_preprocess_metadata('data/dermatology.csv')

# ✅ Feature names (same as in train_model.py)
feature_names = [
    "erythema", "scaling", "definite borders", "itching", "koebner phenomenon", 
    "polygonal papules", "follicular papules", "oral mucosal involvement", 
    "knee and elbow involvement", "scalp involvement", "family history", 
    "melanin incontinence", "eosinophils in the infiltrate", "PNL infiltrate", 
    "fibrosis of the papillary dermis", "exocytosis", "acanthosis", "hyperkeratosis", 
    "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges", 
    "thinning of the suprapapillary epidermis", "spongiform pustule", 
    "munro microabcess", "focal hypergranulosis", "disappearance of the granular layer", 
    "vacuolisation and damage of basal layer", "spongiosis", 
    "saw-tooth appearance of retes", "follicular horn plug", 
    "perifollicular parakeratosis", "inflammatory monoluclear infiltrate"
]

# ✅ Scale features to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# ✅ Chi-Square Test
chi_scores, p_values = chi2(X_scaled_df, y)

# ✅ Create DataFrame
chi_square_results = pd.DataFrame({
    'Feature': feature_names,
    'Chi2 Score': chi_scores,
    'p-value': p_values
})

# ✅ Sort by importance
chi_square_results.sort_values(by='Chi2 Score', ascending=False, inplace=True)

# ✅ Display Top Features
print("\n📊 Top Features by Chi-Square Test:")
print(chi_square_results.head(10))
import matplotlib.pyplot as plt

# Plot top 10 features
top_features = chi_square_results.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'][::-1], top_features['Chi2 Score'][::-1], color='skyblue')
plt.xlabel("Chi-Square Score")
plt.title("Top 10 Features by Chi-Square Test")
plt.tight_layout()
plt.savefig("chi_square_top10_features.png")
plt.show()
