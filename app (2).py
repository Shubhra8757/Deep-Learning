import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Plant Leaf Disease Classifier", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------- constants
IMG_SIZE = (150, 150)          # same size the model was trained on
MODEL_PATH = "final_model.h5"
CONF_THRESHOLD = 0.60          # below this we say "not sure"

CLASS_NAMES = [
    "Cherry_Powdery_mildew", "Cherry_healthy", "Grape_Black_rot",
    "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape_healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Strawberry_Leaf_scorch", "Strawberry_healthy", "Tomato_Bacterial_spot",
    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites", "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_healthy",
]

# Training history copied from the notebook output (16 epochs, early stopping)
HISTORY = pd.DataFrame({
    "Epoch": range(1, 17),
    "Train accuracy": [0.6260, 0.7026, 0.7329, 0.7559, 0.7861, 0.7988, 0.8135, 0.8335,
                       0.8442, 0.8530, 0.8511, 0.8799, 0.8745, 0.8809, 0.8708, 0.9072],
    "Val accuracy":   [0.6577, 0.6577, 0.7173, 0.7578, 0.7461, 0.7910, 0.8008, 0.7964,
                       0.8247, 0.8042, 0.8286, 0.8223, 0.8359, 0.8052, 0.8315, 0.8457],
    "Train loss":     [1.1819, 0.9455, 0.8129, 0.7624, 0.6710, 0.6186, 0.5673, 0.5194,
                       0.4885, 0.4515, 0.4376, 0.3794, 0.3773, 0.3576, 0.3819, 0.2959],
    "Val loss":       [1.1298, 1.0665, 0.8481, 0.7672, 0.7988, 0.6235, 0.6192, 0.6116,
                       0.5335, 0.6218, 0.5073, 0.5342, 0.4937, 0.6043, 0.5328, 0.5034],
}).set_index("Epoch")

SUPPORT = {  # validation support per class (from classification report)
    "Cherry_Powdery_mildew": 210, "Cherry_healthy": 210, "Grape_Black_rot": 236,
    "Grape_Esca_(Black_Measles)": 276, "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": 215,
    "Grape_healthy": 135, "Pepper,_bell___Bacterial_spot": 202, "Pepper,_bell___healthy": 318,
    "Strawberry_Leaf_scorch": 221, "Strawberry_healthy": 140, "Tomato_Bacterial_spot": 425,
    "Tomato_Early_blight": 200, "Tomato_Late_blight": 381, "Tomato_Leaf_Mold": 190,
    "Tomato_Septoria_leaf_spot": 354, "Tomato_Spider_mites": 335, "Tomato_Target_Spot": 280,
    "Tomato_Yellow_Leaf_Curl_Virus": 1071, "Tomato_healthy": 396,
}

def pretty(name: str) -> str:
    return name.replace("___", " – ").replace("_", " ").replace(",", "")

# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict(model, img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    probs = model.predict(arr[None, ...], verbose=0)[0]
    return probs

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌿 Leaf Disease CNN")
page = st.sidebar.radio("Go to", ["Overview", "Training results", "Live prediction", "Conclusions"])
st.sidebar.caption("Built with TensorFlow/Keras + Streamlit")

# ---------------------------------------------------------------- pages
if page == "Overview":
    st.title("🌿 Plant Leaf Disease Classification")
    st.write("A CNN that classifies leaf images into **19 classes** across cherry, grape, "
             "bell pepper, strawberry and tomato (healthy + diseased).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training images", "23,219")
    c2.metric("Validation images", "5,795")
    c3.metric("Classes", "19")
    c4.metric("Validation accuracy", "83.85%")

    st.subheader("Model architecture")
    st.code(
        "Input 150×150×3\n"
        "Conv2D(32, 3×3, ReLU) → MaxPool(2×2)\n"
        "Conv2D(64, 3×3, ReLU) → MaxPool(2×2)\n"
        "Conv2D(128, 3×3, ReLU) → MaxPool(2×2)\n"
        "Flatten → Dropout(0.5) → Dense(128, ReLU) → Dense(19, Softmax)",
        language="text",
    )
    st.subheader("Training setup")
    st.table(pd.DataFrame({
        "Setting": ["Optimizer", "Learning rate", "Loss", "Batch size", "Epochs (max)",
                    "Steps per epoch", "Early stopping", "Preprocessing", "Split"],
        "Value": ["Adam", "0.001", "Categorical cross-entropy", "32", "20 (stopped at 16)",
                  "64", "monitor val_loss, patience 3, restore best weights",
                  "Rescale 1/255 (no augmentation)", "80% train / 20% validation"],
    }).set_index("Setting"))

    st.subheader("Classes")
    st.dataframe(pd.DataFrame({
        "Class": [pretty(c) for c in CLASS_NAMES],
        "Validation images": [SUPPORT[c] for c in CLASS_NAMES],
    }), use_container_width=True, hide_index=True)

elif page == "Training results":
    st.title("📈 Training results")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy")
        st.line_chart(HISTORY[["Train accuracy", "Val accuracy"]])
    with c2:
        st.subheader("Loss")
        st.line_chart(HISTORY[["Train loss", "Val loss"]])

    c1, c2, c3 = st.columns(3)
    c1.metric("Final train accuracy", "90.72%")
    c2.metric("Final val accuracy (training)", "84.57%")
    c3.metric("Full validation evaluation", "83.85%", help="model.evaluate on all 5,795 validation images")

    st.subheader("Epoch table")
    st.dataframe(HISTORY.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Class distribution (validation set)")
    dist = pd.Series(SUPPORT).rename(index=pretty)
    st.bar_chart(dist)

    st.warning(
        "**About the per-class report in the notebook:** it shows ~8% accuracy and near-zero "
        "precision/recall for every class. That contradicts the 83.85% from `model.evaluate` and is "
        "an evaluation bug, not a model result: the validation generator was created with the default "
        "`shuffle=True`, so `validation.classes` is not in the same order as `model.predict` output. "
        "Re-run with `shuffle=False` on the validation generator to get a valid report and confusion matrix."
    )

elif page == "Live prediction":
    st.title("🔍 Live prediction")
    model = load_model()
    if model is None:
        st.error(f"`{MODEL_PATH}` not found. Put the saved model next to `app.py` in your repo.")
        st.stop()

    st.write("Upload a clear photo of a **single leaf** (cherry, grape, bell pepper, strawberry or tomato).")
    files = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    for f in files or []:
        img = Image.open(f)
        probs = predict(model, img)
        top = np.argsort(probs)[::-1][:5]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption=f.name, use_container_width=True)
        with col2:
            conf = float(probs[top[0]])
            label = pretty(CLASS_NAMES[top[0]])
            if conf < CONF_THRESHOLD:
                st.warning(f"Not sure – best guess is **{label}** ({conf:.1%}). "
                           "The image may not be a supported leaf, or the photo is unclear.")
            elif "healthy" in CLASS_NAMES[top[0]]:
                st.success(f"**{label}** – {conf:.1%} confidence")
            else:
                st.error(f"**{label}** – {conf:.1%} confidence")
            st.bar_chart(pd.Series({pretty(CLASS_NAMES[i]): float(probs[i]) for i in top}))
        st.divider()

    st.caption("Educational tool – not a substitute for advice from an agronomist.")

else:
    st.title("✅ Conclusions")
    st.markdown("""
### What worked
- A small 3-block CNN reached **83.85% accuracy** on 5,795 held-out validation images across **19 classes**
  (random guessing would be ~5%).
- Training accuracy rose steadily from 62.6% to 90.7%, and validation loss fell from 1.13 to ~0.49–0.50,
  so the model genuinely learned leaf-disease features.
- Early stopping (patience 3, best weights restored) ended training at epoch 16 of 20.
- The trained model is saved (`final_model.h5`) and reloads for inference, which powers the live demo page.

### Limitations
- **Generalisation gap:** ~90.7% train vs ~84.6% validation accuracy suggests mild overfitting;
  no data augmentation was used.
- **Invalid per-class report:** the notebook's classification report (~8% accuracy) is caused by an unshuffled-order
  mismatch (`shuffle=True` on the validation generator), so per-class precision/recall/F1 are **not yet known**.
- **Class imbalance:** Tomato Yellow Leaf Curl Virus has 1,071 validation images vs 135 for Grape healthy.
- **No "unknown" class:** the notebook's test on a car photo still returned *Tomato_Early_blight*, so the model
  will confidently label any image. The app uses a confidence threshold to partly mitigate this.
- **Image size mismatch:** the prediction cell used 224×224 while the model was trained at 150×150 – the app uses 150×150.
- Dataset images are PlantVillage-style (clean backgrounds); accuracy on real field photos is likely lower.
- Training used only 64 steps/epoch (~2,048 images), so each epoch saw a fraction of the training set.

### Next steps
1. Set `shuffle=False` on the validation generator and regenerate the classification report + confusion matrix.
2. Add augmentation (flip, rotate, zoom, brightness) and class weights.
3. Try transfer learning (MobileNetV2 / EfficientNet) for higher accuracy.
4. Add a "not a leaf" rejection mechanism and test on real field images.
""")
