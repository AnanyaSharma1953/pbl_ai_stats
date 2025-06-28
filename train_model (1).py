import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

from app.preprocess import load_and_preprocess_metadata

# ✅ Feature names
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

# ✅ Load and preprocess
X, y = load_and_preprocess_metadata('data/dermatology.csv')
print("Shape of features:", X.shape)
print("Shape of labels:", y.shape)

# ✅ Balance
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
print("Balanced shape:", X.shape, y.shape)

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# ✅ Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_df, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test_df)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# ✅ Save accuracy to file
accuracy = accuracy_score(y_test, y_pred)
os.makedirs("models", exist_ok=True)
with open("models/rfc_accuracy.txt", "w") as f:
    f.write(str(accuracy))


# ✅ Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")
print("✅ Model saved to models/rf_model.pkl")

# ✅ SHAP
print("\n🔍 Computing SHAP values...")
explainer = shap.TreeExplainer(model, X_train_df)
shap_values = explainer.shap_values(X_test_df, check_additivity=False)
shap.initjs()

print("SHAP values type:", type(shap_values))
print("SHAP values shape:", shap_values.shape)  # Should be (232, 32, 4)
print("X_test_df shape:", X_test_df.shape)

# ✅ SHAP 3D Fix
os.makedirs("outputs", exist_ok=True)
num_classes = shap_values.shape[2] if len(shap_values.shape) == 3 else 0
unique_classes = np.unique(y_train)

for class_idx in unique_classes:
    print(f"\n📊 Generating SHAP plots for class {class_idx}...")
    try:
        shap_vals_class = shap_values[:, :, class_idx]

        if shap_vals_class.shape != X_test_df.shape:
            print(f"❌ Shape mismatch for class {class_idx}: {shap_vals_class.shape} vs {X_test_df.shape}")
            continue

        # Summary plot
        plt.figure()
        shap.summary_plot(shap_vals_class, X_test_df, plot_type="bar", show=False)
        plt.title(f"SHAP Summary - Class {class_idx}")
        plt.tight_layout()
        plt.savefig(f"outputs/shap_summary_class_{class_idx}.png")
        plt.close()

        

        print(f"✅ Saved SHAP plots for class {class_idx}")

    except Exception as e:
        print(f"❌ Failed for class {class_idx}: {e}")
