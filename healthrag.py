# =============================================================
# 🧠 HealthRAG Assistant — Intelligent RAG-style Health Chatbot
# =============================================================
# Section Summaries | AI Insights | True Conversational Memory
# =============================================================

import streamlit as st
import PyPDF2
from openai import OpenAI
import io

# -------------------------------------------------------------
# 🔐 Secure API Key via Streamlit Secrets
# -------------------------------------------------------------
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("❌ Missing API key! Add in Streamlit → Settings → Secrets:\n\nOPENROUTER_API_KEY = sk-or-v1-xxxx")
    st.stop()

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
MODEL = "x-ai/grok-4"

# -------------------------------------------------------------
# 🎨 Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="HealthRAG Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #031628, #1e3c72, #2a5298);
    color: white;
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
    padding: 0.6rem 1.2rem;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #00bfa6;
    color: white;
}
.chat-bubble-user {
    background-color: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 5px;
}
.chat-bubble-ai {
    background-color: rgba(0,255,200,0.1);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
.insight-card {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 🧩 PDF Extraction
# -------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text.strip(), len(reader.pages)
    except Exception as e:
        st.error(f"⚠️ Error reading PDF: {e}")
        return "", 0

# -------------------------------------------------------------
# 🧠 Section Summarizer
# -------------------------------------------------------------
def section_summarizer(text):
    prompt = f"""
You are HealthRAG Assistant. Summarize this medical document by section.
If sections like Abstract, Methods, or Conclusion are missing, skip them.

Text:
{text[:12000]}

Format response as:
### Abstract
...
### Methods
...
### Results
...
### Conclusion
...
"""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize research papers by sections clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# -------------------------------------------------------------
# 💡 AI-driven Insight Extraction
# -------------------------------------------------------------
def extract_ai_insights(pdf_text):
    prompt = f"""
You are HealthRAG Insight Extractor.
Read the research text and summarize key **quantitative and qualitative insights**.

Focus on:
- Dataset size / study population
- Performance metrics (accuracy, F1, etc.)
- Novel contributions
- Limitations
- Clinical or real-world impact

Text:
{pdf_text[:8000]}

Return bullet points only.
"""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You extract insights from medical research papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating insights: {e}"

# -------------------------------------------------------------
# 💬 Conversational Chat with Memory
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat_with_memory(context, user_question):
    """Use last few turns for conversational continuity"""
    history = "\n".join([f"User: {u}\nAI: {a}" for u, a in st.session_state.chat_history[-3:]])
    prompt = f"""
You are HealthRAG Assistant, a retrieval-augmented LLM.
Use document text and conversation history to answer naturally.

Document Context:
{context[:10000]}

Conversation:
{history}

User Question:
{user_question}

Answer precisely and conversationally using only context.
"""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Be a helpful, medically accurate assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        answer = res.choices[0].message.content.strip()
        st.session_state.chat_history.append((user_question, answer))
        return answer
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# -------------------------------------------------------------
# 🧠 Streamlit UI
# -------------------------------------------------------------
st.title("🧠 HealthRAG Assistant")
st.caption("Retrieval-Augmented Health Chatbot with Section Summaries, AI Insights, and True Conversational Memory")

uploaded_file = st.file_uploader("📤 Upload your medical research paper (PDF)", type=["pdf"])

if uploaded_file:
    pdf_text, pages = extract_text_from_pdf(uploaded_file)
    st.success(f"✅ Extracted {pages} pages successfully")

    # === Summarization ===
    st.subheader("📑 Section-wise Summary")
    if st.button("🩺 Generate Structured Summary"):
        with st.spinner("Analyzing document..."):
            summary = section_summarizer(pdf_text)
            st.markdown(summary)

        st.subheader("💡 Key Health Insights")
        with st.spinner("Extracting key metrics and insights..."):
            insights = extract_ai_insights(pdf_text)
            st.markdown(f"<div class='insight-card'>{insights}</div>", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Summary & Insights",
            data=(summary + "\n\n" + insights).encode("utf-8"),
            file_name="HealthRAG_Report.txt",
            mime="text/plain"
        )

    st.divider()

    # === Chat Section ===
    st.subheader("💬 Interactive Chat with Memory")
    for q, a in st.session_state.chat_history:
        st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-ai'><b>HealthRAG:</b> {a}</div>", unsafe_allow_html=True)

    user_q = st.text_input("Ask your question here:")
    if st.button("🤖 Ask"):
        if user_q.strip():
            with st.spinner("Thinking..."):
                answer = chat_with_memory(pdf_text, user_q)
            st.markdown(f"<div class='chat-bubble-ai'><b>🧠 HealthRAG:</b> {answer}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please type a question.")
else:
    st.info("📥 Upload a research paper to begin.")
