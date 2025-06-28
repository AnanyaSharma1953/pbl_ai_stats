import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import io

# Load models
symptom_model = joblib.load("models/rf_model.pkl")
image_model = load_model("models/image_model_mobilenetv2.keras")

# Disease class mappings (for image model - adjust as needed)
image_class_labels = {
    0: "Acne", 
    1: "Hyperpigmentation", 
    2: "Nail Psoriasis", 
    3: "SJS-TEN", 
    4: "Vitiligo", 
    5: "Other"
}

# Feature names with tooltips
feature_info = {
    "erythema": "Redness of the skin",
    "scaling": "Skin flaking or shedding",
    "definite borders": "Clear edge around the lesion",
    "itching": "Sensation that causes a desire to scratch",
    "koebner phenomenon": "Skin lesions appearing after trauma",
    "polygonal papules": "Small, raised, flat-topped bumps with many sides",
    "follicular papules": "Bumps related to hair follicles",
    "oral mucosal involvement": "Lesions inside the mouth",
    "knee and elbow involvement": "Symptoms appear on knees and elbows",
    "scalp involvement": "Symptoms on the scalp",
    "family history": "Any family member has similar skin issues",
    "melanin incontinence": "Pigment leakage into the dermis",
    "eosinophils in the infiltrate": "Presence of immune cells in skin biopsy",
    "PNL infiltrate": "Polymorphonuclear leukocyte infiltration",
    "fibrosis of the papillary dermis": "Thickening of the upper skin layer",
    "exocytosis": "Movement of cells to the skin surface",
    "acanthosis": "Thickening of the skin",
    "hyperkeratosis": "Overproduction of keratin",
    "parakeratosis": "Retention of nuclei in skin's outer layer",
    "clubbing of the rete ridges": "Bulbous projections of epidermis",
    "elongation of the rete ridges": "Lengthened skin projections",
    "thinning of the suprapapillary epidermis": "Thinned layer above dermis",
    "spongiform pustule": "Pus-filled blister in upper skin",
    "munro microabcess": "Small immune cell clusters in skin",
    "focal hypergranulosis": "Increased granular layer in spots",
    "disappearance of the granular layer": "Loss of skin granules",
    "vacuolisation and damage of basal layer": "Damage to skin base cells",
    "spongiosis": "Intercellular skin swelling",
    "saw-tooth appearance of retes": "Notched pattern in skin layers",
    "follicular horn plug": "Keratin plug in hair follicles",
    "perifollicular parakeratosis": "Parakeratosis around follicles",
    "inflammatory monoluclear infiltrate": "Inflammatory cell presence"
}

# Disease mapping for metadata model
symptom_class_map = {
    0: "Psoriasis",
    1: "Seborrheic Dermatitis",
    2: "Lichen Planus",
    3: "Pityriasis Rosea",
    4: "Chronic Dermatitis",
    5: "Pityriasis Rubra Pilaris"
}

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("🌟 SkinSavvy: Know Your Skin with AI!")
st.markdown("**Choose how you want to get diagnosed:**")

tabs = st.tabs(["📋 Symptom-Based Entry", "🖼️ Image Upload (AI Diagnosis)"])

# --- Tab 1: Symptoms ---
with tabs[0]:
    st.subheader("🧾 Enter your symptoms below:")

    user_input = []
    for feature, tooltip in feature_info.items():
        val = st.slider(f"{feature}", 0, 3, 0, help=tooltip)
        user_input.append(val)

    input_df = pd.DataFrame([user_input], columns=feature_info.keys())

    if st.button("🔍 Predict from Symptoms"):
        prediction = symptom_model.predict(input_df)[0]
        predicted_disease = symptom_class_map.get(prediction, "Unknown")

        st.success(f"🎯 Predicted Skin Condition: {predicted_disease} (Class {prediction})")

        # SHAP plots
        st.subheader("📊 SHAP Explanation")
        try:
       
            st.image(f"outputs/shap_summary_class_{prediction}.png", caption="SHAP Summary Plot", width=600)

            with open(f"outputs/shap_force_plot_class_{prediction}_instance_0.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=400, scrolling=True)
        except Exception as e:
            st.warning("⚠️ SHAP explanation not available.")

# --- Tab 2: Image Upload ---
with tabs[1]:
    st.subheader("🖼️ Upload a Skin Image:")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=False, width=300)

        # Preprocess
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = image_model.predict(img_array)[0]
        class_index = np.argmax(pred)
        confidence = np.max(pred) * 100
        predicted_label = image_class_labels.get(class_index, "Unknown")

        st.success(f"🎯 Predicted Condition: **{predicted_label}**")
        st.info(f"📊 Confidence: {confidence:.2f}%")

