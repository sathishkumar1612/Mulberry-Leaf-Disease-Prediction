import streamlit as st
from PIL import Image
from app import perform_leaf_inference, RECOMMENDATIONS

# --- Page Config ---
st.set_page_config(
    page_title="MulberryCare AI — Foliage Diagnostic Portal",
    page_icon="🍃",
    layout="centered"
)

# --- UI Header ---
st.title("🍃 MulberryCare AI Diagnostic Portal")
st.markdown("Upload a mulberry leaf image (*Morus alba*) to detect foliar diseases, prevent silkworm crop loss, and obtain sericulture recommendations.")
st.divider()

uploaded_file = st.file_uploader("Upload Leaf Image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display uploaded leaf preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Mulberry Foliage", width=360)

    # Analyze button
    if st.button("🔬 Run Pathology Diagnostic Analysis", use_container_width=True):
        with st.spinner("Analyzing chlorophyll distribution and fungal symptoms..."):
            predicted_class, confidence, all_confidences = perform_leaf_inference(image)

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

        st.metric("Diagnostic Confidence", f"{confidence:.1f}%")
        st.divider()

        # Confidence bar chart
        st.subheader("📊 Probability Breakdown")
        for name, prob in all_confidences.items():
            prob_val = float(prob) if prob <= 1.0 else float(prob) / 100.0
            st.write(f"**{name}** ({prob:.1f}%)")
            st.progress(float(min(1.0, max(0.0, prob_val))))

        st.divider()

        # Recommendations
        st.subheader("🌱 Sericulture-Safe Treatment Steps")
        recs = RECOMMENDATIONS.get(predicted_class, RECOMMENDATIONS["Unknown / Unidentified Disease"])
        for rec in recs:
            st.info(rec)