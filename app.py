import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import os

# -------------------- 1. PAGE CONFIG --------------------
st.set_page_config(
    page_title="AgroBot Pro 🌱",
    page_icon="🌿",
    layout="centered"
)

# -------------------- 2. NATURE THEME CSS --------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%);
    }
    .chat-bubble {
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        font-family: 'Helvetica Neue', sans-serif;
        line-height: 1.7;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .user {
        background-color: #1b5e20;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 2px;
        width: fit-content;
        max-width: 80%;
    }
    .bot {
        background-color: #ffffff;
        color: #1b5e20;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border-left: 8px solid #43a047;
        width: fit-content;
        max-width: 90%;
    }
    hr { margin: 10px 0; border-top: 1px solid #ddd; }
    .main .block-container { padding-bottom: 150px; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- 3. ASSET LOADING --------------------
@st.cache_resource
def load_assets():
    try:
        model_path = "models/cnn_model.h5" if os.path.exists("models/cnn_model.h5") else "cnn_model.h5"
        nlp_path = "models/nlp_model.pkl" if os.path.exists("models/nlp_model.pkl") else "nlp_model.pkl"
        vec_path = "models/vectorizer.pkl" if os.path.exists("models/vectorizer.pkl") else "vectorizer.pkl"
        
        cnn = tf.keras.models.load_model(model_path)
        nlp = pickle.load(open(nlp_path, "rb"))
        vec = pickle.load(open(vec_path, "rb"))
        return cnn, nlp, vec
    except Exception as e:
        st.error(f"⚠️ Error: Assets not found. Please upload .h5 and .pkl files. Details: {e}")
        st.stop()

cnn_model, nlp_model, vectorizer = load_assets()
labels = list(nlp_model.classes_)

# -------------------- 4. ENHANCED KNOWLEDGE DATABASE --------------------
def get_detailed_interpretation(disease):
    db = {
        "Early Blight": {
            "symptoms": "Target-like brown spots on older leaves, yellowing halos, and eventual stem cankers.",
            "cause": "The fungus *Alternaria solani*. It thrives in high humidity and frequent rainfall.",
            "organic": "Prune lower leaves to prevent soil splash. Apply Neem Oil or Compost Tea every 7 days.",
            "chemical": "Fungicides containing Chlorothalonil or Copper soap should be applied at the first sign of spots.",
            "prevention": "Practice a 3-year crop rotation and use drip irrigation to keep foliage dry."
        },
        "Late Blight": {
            "symptoms": "Dark, water-soaked patches on leaves that turn black and papery. White fuzz may appear under leaves.",
            "cause": "*Phytophthora infestans*. This is highly aggressive and can destroy crops in days.",
            "organic": "There is no cure; immediately pull and burn infected plants. Use copper sprays as a preventative measure.",
            "chemical": "Mancozeb or Ridomil Gold. These are best used before the disease spreads across the field.",
            "prevention": "Ensure wide plant spacing and destroy all 'volunteer' plants from previous seasons."
        },
        "Powdery Mildew": {
            "symptoms": "White, flour-like powder coating on leaves, stems, and fruit. Leaves may twist or dry up.",
            "cause": "Warm, dry days and cool, damp nights that encourage spore germination.",
            "organic": "Spray a mixture of 40% milk and 60% water, or use Potassium Bicarbonate sprays.",
            "chemical": "Triadimefon or Myclobutanil fungicides.",
            "prevention": "Plant in full sun and prune the center of the plant to allow wind to pass through."
        },
        "Leaf Mold": {
            "symptoms": "Pale green or yellow spots on top of leaves with olive-green velvety mold underneath.",
            "cause": "High humidity (above 85%) and poor ventilation in greenhouse settings.",
            "organic": "Increase temperature and decrease humidity. Vinegar-based sprays can help in early stages.",
            "chemical": "Fungicides containing Difenoconazole.",
            "prevention": "Stake plants to keep them off the ground and use fans to move air."
        },
        "Healthy": {
            "symptoms": "Deep green foliage, firm stems, and vigorous new growth.",
            "cause": "Optimal soil nutrients, proper watering, and good sunlight.",
            "organic": "Maintain soil health with organic mulch and worm castings.",
            "chemical": "None required. Maintain current care.",
            "prevention": "Sterilize garden tools between uses to prevent accidental infection."
        }
    }
    
    data = db.get(disease, {
        "symptoms": "Irregular spotting, discoloration, or wilting observed.",
        "cause": "Likely environmental stress or local pathogens.",
        "organic": "Remove affected tissue and spray with diluted Neem oil.",
        "chemical": "Broad-spectrum fungicide application.",
        "prevention": "Monitor local weather and improve soil drainage."
    })
    
    return f"""
**🔍 CLINICAL INTERPRETATION**
* **Symptoms:** {data['symptoms']}
* **Root Cause:** {data['cause']}

---

**🛠️ RECOVERY PROTOCOL**
* **Organic Remedy:** {data['organic']}
* **Chemical Solution:** {data['chemical']}

---

**🛡️ LONG-TERM PREVENTION**
{data['prevention']}
    """

# -------------------- 5. PROFESSIONAL DIALOGUE --------------------
def handle_conversation(text, has_image=False):
    t = text.lower().strip()
    
    if any(g in t for g in ["hi", "hello", "hey", "asalaam", "help"]) and len(t) < 15:
        return "👋 **Greetings!** I am AgroBot Pro. I am configured to provide expert-level plant pathology analysis. Please describe your symptoms or provide a high-resolution leaf image."

    plant_keywords = ["leaf", "plant", "spot", "yellow", "water", "soil", "grow", "disease", "blight", "mildew", "pest", "bug", "wilt", "tomato", "potato", "crop", "tree", "garden", "fruit"]
    if not any(k in t for k in plant_keywords) and not has_image:
        return "🔭 **Out of Scope:** I am a specialized agricultural assistant. I am unable to engage in non-botanical discussions. Please provide plant-related details for a diagnosis."

    vec_input = vectorizer.transform([t])
    disease = nlp_model.predict(vec_input)[0]
    return get_detailed_interpretation(disease)

# -------------------- 6. UI CONTENT --------------------
st.markdown("<h1 style='text-align:center; color:#1b5e20;'>🌿 AgroBot Pro</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [("bot", "☀️ Please provide leaf symptoms or an image for diagnostic processing.")]

with st.sidebar:
    st.header("🎛️ App Controls")
    if st.button("🗑️ Reset Consultation"):
        st.session_state.messages = []
        st.rerun()

for sender, msg in st.session_state.messages:
    css_class = "user" if sender == "user" else "bot"
    st.markdown(f"<div class='chat-bubble {css_class}'>{msg}</div>", unsafe_allow_html=True)

# -------------------- 7. MULTIMODAL INPUT --------------------
with st.container():
    col1, col2 = st.columns([1, 5])
    with col1:
        img_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    with col2:
        user_input = st.chat_input("Enter your observation here...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    
    if img_file:
        img = Image.open(img_file).resize((224, 224))
        arr = np.array(img) / 255.0
        pred = cnn_model.predict(np.expand_dims(arr, axis=0))
        img_disease = labels[np.argmax(pred)]
        
        detail_view = get_detailed_interpretation(img_disease)
        bot_reply = f"📸 **IMAGE ANALYSIS DETECTED: {img_disease}**\n\n{detail_view}\n\n---\n💬 **ADDITIONAL CONTEXT:**\n{handle_conversation(user_input, has_image=True)}"
    else:
        bot_reply = handle_conversation(user_input)

    st.session_state.messages.append(("bot", bot_reply))
    st.rerun()
