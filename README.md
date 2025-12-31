# 🧠 RFT-LAB — Reasoning-First Transformer (Experimental Project)

**Author:** Anurag Kumar Singh  
📧 Email: anuragkumarsingh4440@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/anurag-kumar-singh-649a23249  

---

## 📌 Project Overview

**RFT-LAB (Reasoning-First Transformer)** is an **experimental research-oriented project** built to explore **how reasoning can be explicitly separated from understanding and answer generation in Transformer architectures**.

This project is **NOT a production ChatGPT clone**.  
It is a **conceptual + implementation-level experiment** created while studying Transformer internals.

> 🎯 The main goal is to **learn, experiment, and demonstrate deep architectural thinking**, not deployment polish.

---

## 🧩 Core Idea — What is RFT?

Traditional LLMs mix everything:
- understanding  
- reasoning  
- answering  

inside one opaque process.

**RFT breaks this into explicit phases:**

1. **Understanding** — encode input meaning  
2. **Reasoning** — iterative latent transformation  
3. **Answer Decoding** — verbalization only  
4. **Metrics** — transparency & confidence  

This makes reasoning:
- observable  
- controllable  
- explainable  

---

## 📁 Project File Structure
<img width="331" height="334" alt="image" src="https://github.com/user-attachments/assets/f843e4ef-b181-44cb-9a9a-4c0867f46389" />


### 🔍 File-wise Explanation

#### 🟦 `01_input_handling.ipynb`
- Handles text normalization
- PDF → text
- Image → OCR
- Audio / mic → speech-to-text
- Input validation

➡️ **No ML, no reasoning**

---

#### 🟦 `02_understanding_encoder.ipynb`
- Tokenization
- Embeddings
- Positional encoding
- Transformer encoder blocks

➡️ **Pure understanding, no reasoning**

---

#### 🟥 `03_reasoning_block.ipynb` (Core of RFT)
- Latent reasoning layer
- Iterative transformations
- Reasoning depth controller
- Representation shift measurement

➡️ **No text generation here**

---

#### 🟦 `04_answer_decoder.ipynb`
- Latent → logits
- Logits → tokens
- Tokens → text

➡️ **Decoder only speaks what is already reasoned**

---

#### 🟨 `05_system_metrics_dashboard.ipynb`
- Reasoning depth
- Confidence score
- Representation stability
- Warnings

➡️ **Transparency & trust layer**

---

## ⚠️ About `app.py` (Important Note)

`app.py` exists only as a **practice-level Streamlit orchestration attempt**.

### Why `app.py` may not run / deploy properly

- This project was **not built with Streamlit expertise**
- Streamlit UI code was **assisted using LLMs (ChatGPT / Gemini)**  
- Focus was on **architecture & reasoning**, not frontend robustness
- Multiple Streamlit constraints (keys, layout, state) were learned during experimentation

➡️ **Recruiters should treat `app.py` as optional**  
➡️ **The real value lies in the notebooks and architectural thinking**

---

## 👨‍💻 Author Background (Honest Context)

I am a **Data Scientist** by role and training.

My core strengths:
- Data understanding
- Feature engineering
- Model building
- ML/DL experimentation
- Transformer internals

I am **not a deployment/UI specialist**, and this project was never intended as a polished product.

> This repository reflects **learning depth and architectural curiosity**, not UI perfection.

---

## 🌱 Why This Project Matters

- Demonstrates **deep Transformer understanding**
- Shows **original thinking** beyond model fine-tuning
- Explicitly separates reasoning (rare in projects)
- Honest about limitations and scope
- Research-oriented mindset

---

## 🚧 Current Status

- ✅ Architecture design complete
- ✅ Core reasoning concept implemented
- ✅ Metrics & transparency explored
- ⚠️ Streamlit app experimental
- ❌ Not production-ready (by design)

---

## 🧠 Final Note

> RFT-LAB is an **exploration**, not a product.  
> It reflects how I think about models internally — step by step, transparently, and critically.

If you are evaluating **thinking depth rather than UI polish**, this project is best read **notebook-by-notebook**.

---

⭐ Thank you for reviewing this work.

