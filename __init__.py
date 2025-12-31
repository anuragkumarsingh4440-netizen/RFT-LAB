# ============================================================
# RFT-LAB — STREAMLIT APPLICATION (FILE 06)
# ============================================================
# This file is ONLY a UI + orchestration layer.
# All intelligence comes from 01–05 notebooks.
# ============================================================

import streamlit as st
import tempfile
import os

# ------------------------------------------------------------
# IMPORT RFT CORE LOGIC (FROM 01–05)
# ------------------------------------------------------------
# These functions MUST exist in your notebooks
# (or exported .py versions if you convert later)

from input_handling import handle_input
from understanding_encoder import estimate_complexity
from reasoning_block import decide_reasoning_depth
from answer_decoder import generate_answer
from system_metrics_dashboard import collect_system_metrics


# ------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="RFT-Lab — Reasoning First AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# DARK MODE + HIGH VISIBILITY CSS
# ------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b1220;
    color: #ffffff;
}
h1, h2, h3 {
    color: #ffffff;
    font-weight: 800;
}
.stChatMessage {
    background-color: #111827;
    border-radius: 14px;
    padding: 14px;
}
.stButton>button {
    background-color: #1f6fff;
    color: white;
    font-weight: bold;
}
.stSlider label {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# APP HEADER (PRODUCT STORY)
# ------------------------------------------------------------
st.title("🧠 RFT-Lab — Reasoning-First AI System")

st.markdown("""
This is **not a normal chatbot**.

• Understanding is separated from reasoning  
• Reasoning depth is controllable  
• Generation happens LAST  
• Confidence & transparency are visible  

**GPT is only the decoder — intelligence is the system.**
""")

# ------------------------------------------------------------
# SIDEBAR — USER CONTROL PANEL
# ------------------------------------------------------------
with st.sidebar:
    st.header("🧠 Reasoning Controls")

    # User bias on how deep AI should think
    user_depth = st.slider(
        "Reasoning Intensity",
        min_value=1,
        max_value=10,
        value=4
    )

    # These parameters influence complexity estimation
    heads = st.slider(
        "Attention Heads",
        min_value=1,
        max_value=20,
        value=8
    )

    layers = st.slider(
        "Latent Reasoning Layers",
        min_value=1,
        max_value=50,
        value=16
    )

    # Decoder creativity (LLM temperature)
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.0
    )

    show_metrics = st.checkbox(
        "Show Reasoning Metrics",
        value=True
    )

# ------------------------------------------------------------
# SESSION STATE — CHAT HISTORY
# ------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------
# USER INPUT SECTION (CHATGPT STYLE)
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Image / PDF / Audio (optional)",
    type=["png", "jpg", "jpeg", "pdf", "wav", "mp3"]
)

user_query = st.chat_input(
    "Ask anything: math, code, reasoning, explanation..."
)

# ------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ------------------------------------------------------------
if user_query or uploaded_file:

    with st.spinner("RFT is thinking deeply..."):

        # -------------------------------
        # STEP 01 — INPUT HANDLING
        # -------------------------------
        if uploaded_file:
            suffix = uploaded_file.name.split(".")[-1]

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            # Convert image/pdf/audio → clean text
            clean_text = handle_input("file", file_path)

        else:
            # Plain text input
            clean_text = handle_input("text", user_query)

        # -------------------------------
        # STEP 02 — UNDERSTANDING ENCODER
        # Estimate complexity of input
        # -------------------------------
        complexity_score = estimate_complexity(
            clean_text,
            heads=heads,
            layers=layers
        )

        # -------------------------------
        # STEP 03 — REASONING BLOCK
        # Decide how deep reasoning should go
        # -------------------------------
        reasoning_depth = decide_reasoning_depth(
            complexity_score,
            user_depth
        )

        # -------------------------------
        # STEP 04 — ANSWER DECODER
        # REAL LLM CALL happens here
        # -------------------------------
        final_answer = generate_answer(
            prompt=clean_text,
            depth=reasoning_depth,
            temperature=temperature
        )

        # -------------------------------
        # STEP 05 — SYSTEM METRICS
        # Transparency & confidence
        # -------------------------------
        metrics = collect_system_metrics({
            "steps_used": reasoning_depth,
            "avg_representation_shift": complexity_score,
            "reasoned_state": None
        })

    # --------------------------------------------------------
    # UPDATE CHAT HISTORY
    # --------------------------------------------------------
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query or uploaded_file.name}
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": final_answer,
            "metrics": metrics
        }
    )

# ------------------------------------------------------------
# DISPLAY CHAT (FORMATTED LIKE CHATGPT)
# ------------------------------------------------------------
for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show reasoning metrics only for assistant messages
        if msg["role"] == "assistant" and show_metrics:
            st.markdown("#### 🔍 Reasoning Transparency")
            st.json(msg["metrics"])
