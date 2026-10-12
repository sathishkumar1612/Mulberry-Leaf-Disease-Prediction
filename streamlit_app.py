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

# --- Class Info ---
CLASS_NAMES_4 = ["Disease Free", "Potential Leaf Rust", "Potential Leaf Spot", "Unknown / Unidentified Disease"]
CLASS_NAMES_3 = ["Disease Free", "Potential Leaf Rust", "Potential Leaf Spot"]

RECOMMENDATIONS = {
    "Disease Free": [
        "✅ Leaf appears healthy and vigorous. Continue regular cultivation care.",
        "💧 Maintain balanced irrigation; avoid soil waterlogging.",
        "🌿 Apply recommended NPK fertilizer schedule to preserve plant immunity.",
        "🔍 Conduct regular weekly scouting to catch early signs of pest or pathogen stress.",
    ],
    "Potential Leaf Rust": [
        "⚠️ Leaf Rust (Cerotelium fici) detected! Remove and destroy severely infected leaves immediately.",
        "🧪 Spray copper-based fungicide or Mancozeb (0.2%) every 7–10 days.",
        "💨 Prune dense inner branches to maximize sunlight penetration and reduce canopy humidity.",
        "🚫 Water strictly at plant base — avoid overhead sprinkler watering on foliage.",
        "🐛 Observe safe waiting period (10–14 days) before feeding sprayed leaves to silkworms.",
    ],
    "Potential Leaf Spot": [
        "⚠️ Leaf Spot (Pseudocercospora mori / Phloeospora maculans) identified!",
        "✂️ Remove heavily spotted leaves from lower shoots to halt upward spore spread.",
        "🧪 Spray systemic fungicide such as Carbendazim (0.1%) or Chlorothalonil.",
        "🌱 Increase spacing between rows to improve cross-ventilation.",
        "💧 Schedule watering in early morning so leaf surfaces dry rapidly during daylight.",
        "🧹 Clear and incinerate all fallen diseased leaf litter around the base of the plant.",
    ],
    "Unknown / Unidentified Disease": [
        "🔍 The leaf pattern does not decisively match standard Leaf Rust or Leaf Spot.",
        "📸 Retake a close-up photo in natural daylight focusing on a single leaf against a plain background.",
        "🔬 Check for other common mulberry conditions: Powdery Mildew, Mosaic Virus, or nutrient deficiency.",
        "👨‍🌾 Consult your local sericulture extension officer or plant pathology lab for microscopic analysis.",
        "🚫 Avoid applying broad-spectrum fungicides until pathogen is confirmed to safeguard silkworm health.",
    ],
}

CONFIDENCE_THRESHOLD = 70.0
MARGIN_THRESHOLD = 20.0

# --- Load Model (only once) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mulberry_leaf_disease_Identification.keras")
    img_size = (128, 128)
    if hasattr(model, 'input_shape') and model.input_shape and len(model.input_shape) >= 3:
        h, w = model.input_shape[1], model.input_shape[2]
        if h and w:
            img_size = (int(h), int(w))
    return model, img_size

model, IMG_SIZE = load_model()

# --- UI ---
st.title("🍃 Mulberry Leaf Disease Detector")
st.markdown("Upload a mulberry leaf image to detect diseases, identify unknown afflictions, and obtain sericulture recommendations.")
st.divider()

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=320)
    
    # Analyze button
    if st.button("🔬 Analyze Leaf", use_container_width=True):
        with st.spinner("Analyzing leaf pathology..."):
            # Preprocess
            img_resized = image.resize(IMG_SIZE)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            raw_predictions = model.predict(img_array, verbose=0)[0]
            num_classes = len(raw_predictions)

            if num_classes == 4:
                predicted_idx = int(np.argmax(raw_predictions))
                predicted_class = CLASS_NAMES_4[predicted_idx]
                confidence = float(raw_predictions[predicted_idx]) * 100
                if confidence < CONFIDENCE_THRESHOLD and predicted_class != "Unknown / Unidentified Disease":
                    predicted_class = "Unknown / Unidentified Disease"
                    confidence = 100.0 - confidence
                all_confidences = {
                    CLASS_NAMES_4[i]: float(raw_predictions[i])
                    for i in range(4)
                }
            else:
                sorted_indices = np.argsort(raw_predictions)[::-1]
                top_idx = int(sorted_indices[0])
                second_idx = int(sorted_indices[1])
                top_conf = float(raw_predictions[top_idx]) * 100
                second_conf = float(raw_predictions[second_idx]) * 100

                all_confidences = {
                    CLASS_NAMES_3[i]: float(raw_predictions[i])
                    for i in range(len(CLASS_NAMES_3))
                }

                if top_conf < CONFIDENCE_THRESHOLD or (top_conf - second_conf) < MARGIN_THRESHOLD:
                    predicted_class = "Unknown / Unidentified Disease"
                    unknown_conf = max(30.0, 100.0 - top_conf)
                    confidence = unknown_conf
                    all_confidences["Unknown / Unidentified Disease"] = unknown_conf / 100.0
                else:
                    predicted_class = CLASS_NAMES_3[top_idx]
                    confidence = top_conf
                    all_confidences["Unknown / Unidentified Disease"] = 0.0

        st.divider()

        # Result Banner
        if predicted_class == "Disease Free":
            st.success(f"### 🌿 {predicted_class}")
        elif predicted_class == "Potential Leaf Rust":
            st.warning(f"### ⚠️ {predicted_class}")
        elif predicted_class == "Potential Leaf Spot":
            st.error(f"### 🚨 {predicted_class}")
        else:
            st.info(f"### ❓ {predicted_class}")

        st.metric("Confidence", f"{confidence:.1f}%")
        st.divider()

        # Confidence bar chart
        st.subheader("📊 Confidence Breakdown")
        for name, prob in all_confidences.items():
            st.write(f"**{name}**")
            st.progress(float(min(1.0, max(0.0, prob))), text=f"{prob * 100:.1f}%")

        st.divider()

        # Recommendations
        st.subheader("🌱 Recommended Next Steps")
        recs = RECOMMENDATIONS.get(predicted_class, RECOMMENDATIONS["Unknown / Unidentified Disease"])
        for rec in recs:
            st.info(rec)