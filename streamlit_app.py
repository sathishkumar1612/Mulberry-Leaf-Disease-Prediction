import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="Mulberry Leaf Disease Detector",
    page_icon="🍃",
    layout="centered"
)

# --- Load Model (only once) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mulberry_leaf_disease_Identification.keras")
    return model

model = load_model()

# --- Class Info ---
CLASS_NAMES = ["Disease Free", "Potential Leaf Rust", "Potential Leaf Spot"]

RECOMMENDATIONS = {
    "Disease Free": [
        "✅ Leaf is healthy. Continue regular monitoring.",
        "💧 Maintain proper irrigation, avoid waterlogging.",
        "🌿 Apply balanced NPK fertilizer every 2–3 weeks.",
    ],
    "Potential Leaf Rust": [
        "⚠️ Remove and destroy infected leaves immediately.",
        "🧪 Apply copper-based fungicide every 7–10 days.",
        "💨 Improve air circulation by pruning dense foliage.",
        "🚫 Avoid overhead watering.",
    ],
    "Potential Leaf Spot": [
        "⚠️ Remove heavily infected leaves.",
        "🧪 Spray with Chlorothalonil or Carbendazim fungicide.",
        "🌱 Ensure proper plant spacing for airflow.",
        "💧 Water early morning so leaves dry during the day.",
    ],
}

# --- UI ---
st.title("🍃 Mulberry Leaf Disease Detector")
st.markdown("Upload a mulberry leaf image to detect disease and get treatment advice.")
st.divider()

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)
    
    # Analyze button
    if st.button("🔬 Analyze Leaf", use_container_width=True):
        
        with st.spinner("Analyzing..."):
            # Preprocess
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)[0]
            predicted_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = predictions[predicted_idx] * 100

        st.divider()

        # Result
        if predicted_class == "Disease Free":
            st.success(f"### ✅ {predicted_class}")
        elif predicted_class == "Potential Leaf Rust":
            st.warning(f"### ⚠️ {predicted_class}")
        else:
            st.error(f"### 🚨 {predicted_class}")

        st.metric("Confidence", f"{confidence:.1f}%")
        st.divider()

        # Confidence bar chart for all classes
        st.subheader("📊 Confidence Breakdown")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"**{name}**")
            st.progress(float(predictions[i]), text=f"{predictions[i]*100:.1f}%")

        st.divider()

        # Recommendations
        st.subheader("🌱 Recommendations")
        for rec in RECOMMENDATIONS[predicted_class]:
            st.info(rec)