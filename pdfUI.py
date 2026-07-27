import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_docling.loader import DoclingLoader
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Structured output schema (generic — works for any PDF)
# ---------------------------------------------------------------------------
class DocumentSummary(BaseModel):
    title: str = Field(description="A short, descriptive title for the document")
    document_type: Optional[str] = Field(
        default=None, description="What kind of document this is, e.g. report, article, contract, resume, manual"
    )
    summary: str = Field(description="A concise 3-6 sentence overview of the document's content")
    key_points: List[str] = Field(default_factory=list, description="The most important points or takeaways")
    topics: List[str] = Field(default_factory=list, description="Main topics, themes, or keywords covered")
    conclusion: Optional[str] = Field(
        default=None, description="Any final conclusion, recommendation, or call to action in the document"
    )

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 20% 20%, #1b1f3b 0%, #0f1117 55%, #0a0b10 100%);
        }

        .hero {
            text-align: center;
            padding: 2.2rem 1rem 1rem 1rem;
        }

        .hero h1 {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #8ec5fc 0%, #e0c3fc 50%, #f6d365 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }

        .hero p {
            color: #a9adc1;
            font-size: 1.05rem;
            margin-top: 0;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1.8rem 1.6rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(6px);
            margin-bottom: 1.5rem;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1.5px dashed rgba(142, 197, 252, 0.45);
            border-radius: 14px;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(142, 197, 252, 0.8);
        }

        .stButton > button {
            width: 100%;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-weight: 600;
            font-size: 1rem;
            color: #0f1117;
            background: linear-gradient(90deg, #8ec5fc 0%, #e0c3fc 60%, #f6d365 100%);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(142, 197, 252, 0.35);
        }

        .summary-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.8rem;
            color: #e6e6ec;
            line-height: 1.7;
            font-size: 1.02rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        }

        .summary-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.2rem;
            font-weight: 700;
            color: #f6d365;
            margin-bottom: 1rem;
        }

        section[data-testid="stSidebar"] { display: none; }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📄 PDF Summarizer</h1>
        <p>Upload a PDF and let AI distill it into a clear, concise summary.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload card
# ---------------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
summarize_clicked = st.button("✨ Summarize PDF")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Processing (same logic as original script)
# ---------------------------------------------------------------------------
if summarize_clicked:
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        os.makedirs("documents", exist_ok=True)
        file_path = os.path.join("documents", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading and summarizing your document..."):
            template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an expert, you have to summarize the pdf uploaded by user"),
                    ("human", "{data}"),
                ]
            )

            data = DoclingLoader(file_path=file_path)
            docs = data.load()

            prompt = template.format_messages(data=docs)

            model = ChatMistralAI(model="mistral-small-2506")
            structured_model = model.with_structured_output(DocumentSummary)

            result: DocumentSummary = structured_model.invoke(prompt)

        # ------------------------------------------------------------------
        # Render structured JSON as styled cards
        # ------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-title">🧠 {result.title}</div>
                <p style="color:#c9cbe0; margin-top:-0.5rem;">
                    {result.document_type or "Document"}
                </p>
                <p>{result.summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def render_list_card(title: str, items: List[str], icon: str):
            if not items:
                return
            list_html = "".join(f"<li>{item}</li>" for item in items)
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-title">{icon} {title}</div>
                    <ul style="margin:0; padding-left:1.2rem;">{list_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_list_card("Key Points", result.key_points, "🔑")
        render_list_card("Topics", result.topics, "🏷️")

        if result.conclusion:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-title">✅ Conclusion</div>
                    <p>{result.conclusion}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View raw JSON"):
            st.json(result.model_dump())