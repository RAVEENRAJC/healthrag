# =============================================================
# 🧠 HealthRAG Assistant — Intelligent RAG-Style Health Chatbot
# =============================================================
# Modern UI + Section Summaries + Health Insights + Multi-turn Q&A
# =============================================================

import streamlit as st
import PyPDF2
from openai import OpenAI
import re
import io

# -------------------------------------------------------------
# 🔐 Secure API Key from Streamlit Secrets
# -------------------------------------------------------------
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("❌ Missing API key! Add in Streamlit → Settings → Secrets:\n\nOPENROUTER_API_KEY = sk-or-v1-xxxx")
    st.stop()

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
MODEL = "x-ai/grok-4"

# -------------------------------------------------------------
# 🎨 Page Configuration and Custom Styling
# -------------------------------------------------------------
st.set_page_config(page_title="HealthRAG Assistant", layout="wide", page_icon="🧠")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #021B79, #0575E6);
    color: white;
}
h1, h2, h3, h4 {
    color: #00ffcc !important;
    text-shadow: 0 0 10px #00ffcc;
}
.stButton>button {
    background-color: #00ffcc;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
}
.stButton>button:hover {
    background-color: #00bfa6;
    color: white;
}
div[data-testid="stChatMessage"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px;
}
.insight-card {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 🧩 PDF Text Extraction
# -------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
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
# 🧠 Section-based Summarizer
# -------------------------------------------------------------
def section_summarizer(text):
    prompt = f"""
You are **HealthRAG Assistant**, a medical research analyzer.
Identify the main sections (Abstract, Introduction, Methodology, Results, Discussion, Conclusion)
and summarize each briefly.

If a section is missing, skip it.

Text:
{text[:12000]}

Provide result as:
### Abstract
(summary)
### Methods
(summary)
### Results
(summary)
### Conclusion
(summary)
"""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize section-wise clearly and professionally."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# -------------------------------------------------------------
# 📊 Health Insight Extractor
# -------------------------------------------------------------
def extract_health_insights(text):
    """Finds useful metrics (accuracy %, sample size, etc.)"""
    insights = []
    metrics = re.findall(r'(\d+(?:\.\d+)?%)', text)
    samples = re.findall(r'(\d+\s+(?:patients|subjects|samples|participants))', text, re.I)
    if metrics:
        insights.append(f"Performance Metrics Found: {', '.join(set(metrics))}")
    if samples:
        insights.append(f"Study Size Mentions: {', '.join(set(samples))}")
    if not insights:
        insights.append("No quantitative results detected.")
    return insights

# -------------------------------------------------------------
# 💬 Multi-turn Chat (Q&A Memory)
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat_with_pdf(context, user_question):
    history_text = "\n".join([f"User: {u}\nAI: {a}" for u, a in st.session_state.chat_history[-3:]])
    prompt = f"""
You are **HealthRAG Assistant**, a retrieval-augmented health research chatbot.
Use the document context and conversation history to answer the new question.

Document Context:
{context[:12000]}

Conversation History:
{history_text}

User Question:
{user_question}

Answer only based on the above text.
"""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an accurate health assistant specialized in research-based answers."},
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
# 🧠 Streamlit App Layout
# -------------------------------------------------------------
st.title("🧠 HealthRAG Assistant")
st.caption("Retrieval-Augmented Health Research Chatbot with Section Summaries, Insights, and Interactive Q&A")

uploaded_file = st.file_uploader("📤 Upload your medical research paper (PDF)", type=["pdf"])

if uploaded_file:
    pdf_text, pages = extract_text_from_pdf(uploaded_file)
    st.success(f"✅ Extracted content from {pages} pages")

    # Section Summaries
    st.subheader("📑 Section-wise Summary")
    if st.button("🩺 Generate Structured Summary"):
        with st.spinner("Analyzing document and generating structured summary..."):
            summary = section_summarizer(pdf_text)
            st.markdown(summary)

        # Insights
        insights = extract_health_insights(pdf_text)
        with st.container():
            st.markdown("### 💡 Health Insights")
            for ins in insights:
                st.markdown(f"<div class='insight-card'>🔹 {ins}</div>", unsafe_allow_html=True)

        # Download option
        summary_full = f"{summary}\n\nHealth Insights:\n" + "\n".join(insights)
        st.download_button(
            "⬇️ Download Full Summary & Insights",
            data=summary_full.encode("utf-8"),
            file_name="HealthRAG_Report.txt",
            mime="text/plain"
        )

    st.divider()

    # Chat Q&A Section
    st.subheader("💬 Interactive Q&A Chat")
    user_q = st.text_input("Ask your question here:")
    if st.button("🤖 Ask HealthRAG"):
        if user_q.strip():
            with st.spinner("Analyzing..."):
                answer = chat_with_pdf(pdf_text, user_q)
            st.markdown(f"**🧠 Answer:** {answer}")

            # Display recent chat history
            st.markdown("### 🕓 Recent Conversation")
            for q, a in reversed(st.session_state.chat_history[-3:]):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**HealthRAG:** {a}")
        else:
            st.warning("Please enter a valid question.")
else:
    st.info("📥 Upload a research paper to start using HealthRAG Assistant.")
