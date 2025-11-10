# =============================================================
# 🤖 HealthRAG Assistant — Retrieval-Augmented Health Chatbot
# =============================================================
# Streamlit + OpenRouter (Grok LLM) + Secure Secrets Handling
# =============================================================

import streamlit as st
import PyPDF2
from openai import OpenAI
import io

# -------------------------------------------------------------
# 🔐 Secure API Key Handling (Streamlit Secrets)
# -------------------------------------------------------------
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error(
        "❌ API key missing! Please add it under Streamlit → Settings → Secrets as:\n\n"
        "OPENROUTER_API_KEY = 'sk-or-v1-xxxxxxxxxxxxxxxxxxxx'"
    )
    st.stop()

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# -------------------------------------------------------------
# 🧠 Initialize OpenRouter Client (Grok Model)
# -------------------------------------------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL = "grok-beta"

# -------------------------------------------------------------
# ⚙️ Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="HealthRAG Assistant",
    layout="wide",
    page_icon="🧠"
)

# -------------------------------------------------------------
# 🎨 Custom Modern UI Styling
# -------------------------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    h1, h2, h3, h4 {
        color: #00ffcc !important;
        text-shadow: 0 0 10px #00ffcc;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00bfa6;
        color: white;
        transform: scale(1.03);
    }
    .stTextInput>div>div>input {
        background-color: rgba(255,255,255,0.1);
        color: white;
        border-radius: 8px;
    }
    .uploadedFile {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.9rem;
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 🧩 PDF Text Extraction
# -------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """Extracts all text content from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip(), len(reader.pages)
    except Exception as e:
        st.error(f"⚠️ Error extracting text: {e}")
        return "", 0

# -------------------------------------------------------------
# 🩺 Summarization Function
# -------------------------------------------------------------
def summarize_health_paper(pdf_text):
    """Summarize a health/medical research paper."""
    prompt = f"""
You are **HealthRAG Assistant**, an intelligent health research summarizer.
Summarize the following medical or health-related document clearly and concisely.

📄 Text:
{pdf_text[:12000]}

Provide the summary in this format:
1. 🩺 Title or Topic
2. 🎯 Objective
3. ⚗️ Methods Used
4. 🔑 Key Findings
5. 💬 Conclusion
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a precise and professional health research summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# -------------------------------------------------------------
# 💬 Question Answering Function
# -------------------------------------------------------------
def ask_health_question(pdf_text, question):
    """Answer user questions based only on PDF content."""
    prompt = f"""
You are **HealthRAG Assistant**, a retrieval-augmented LLM specialized in health and medicine.
Answer the question using ONLY the information from the following document text.

📄 Text:
{pdf_text[:12000]}

❓ Question:
{question}

🧠 Answer (based only on the text, do not add external info):
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Answer strictly based on the provided medical text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# -------------------------------------------------------------
# 🧠 App Interface
# -------------------------------------------------------------
st.title("🧠 HealthRAG Assistant")
st.caption("Retrieval-Augmented Intelligent Health Chatbot (Powered by Grok LLM via OpenRouter)")

uploaded_file = st.file_uploader("📤 Upload your health or medical research paper (PDF)", type=["pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    pdf_text, pages = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        st.info(f"📘 Extracted text from **{pages} pages** successfully.")

        # === Summarization Section ===
        st.subheader("✨ Generate Summary")
        if st.button("🩺 Summarize Document"):
            with st.spinner("Generating structured summary..."):
                summary = summarize_health_paper(pdf_text)
            st.markdown("### 📋 Summary")
            st.write(summary)

            if summary and not summary.startswith("⚠️"):
                st.download_button(
                    label="⬇️ Download Summary (.txt)",
                    data=summary.encode("utf-8"),
                    file_name="HealthRAG_Summary.txt",
                    mime="text/plain"
                )

        st.divider()

        # === Q&A Section ===
        st.subheader("💬 Ask a Question About the Document")
        question = st.text_input("Enter your question here:")
        if st.button("🤖 Get Answer"):
            if question.strip():
                with st.spinner("Analyzing document for the answer..."):
                    answer = ask_health_question(pdf_text, question)
                st.markdown("### 🧠 Answer")
                st.write(answer)
            else:
                st.warning("⚠️ Please enter a valid question.")
    else:
        st.error("⚠️ Could not extract text. Ensure the PDF is text-based (not scanned).")
else:
    st.info("📥 Upload a medical research paper to begin.")

st.markdown("<div class='footer'>🚀 HealthRAG Assistant | Built with ❤️ using Streamlit & OpenRouter Grok</div>", unsafe_allow_html=True)
