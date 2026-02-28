import streamlit as st
import pickle
import numpy as np
import os
import re
import matplotlib.pyplot as plt
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Load artifacts
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-mpnet-base-v2', device='cpu')

embedder = load_embedder()

def summarize(text, num_sentences=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in summary)

def process_complaint(text):
    summary = summarize(text)
    embedding = embedder.encode([text], convert_to_numpy=True, device='cpu')
    label_idx = model.predict(embedding)[0]
    department = le.inverse_transform([label_idx])[0]
    probas = model.predict_proba(embedding)[0]
    return department, probas, summary

# --- Session state init ---
if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""

# --- Sidebar ---
with st.sidebar:
    st.title("ℹ️ Model Info")
    st.markdown("---")
    st.metric("Accuracy", "89%")
    st.metric("Training Samples", "50,000")
    st.metric("Classes", "5")
    st.metric("Embedding Model", "SBERT MPNet")
    st.metric("Classifier", "SVM")
    st.markdown("---")
    st.markdown("**Dataset:** [CFPB Consumer Complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/)")

# --- Main UI ---
st.title("🏦 Consumer Complaint Router")
st.markdown("Enter a consumer complaint to automatically route it to the right department.")

# Example buttons
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)
if col1.button("💳 Credit Report"):
    st.session_state.complaint_text = "I have been trying to get a fraudulent account removed from my credit report for months. The credit bureau keeps verifying the account despite me sending them proof that I never opened it."
if col2.button("🏠 Mortgage"):
    st.session_state.complaint_text = "I took out a 30 year fixed mortgage in 2018 and the bank has been charging me incorrect escrow amounts every month. They refuse to provide a proper escrow analysis."
if col3.button("📞 Debt Collection"):
    st.session_state.complaint_text = "A debt collector keeps calling me 5 times a day even after I sent them a cease and desist letter. They are violating the FDCPA by continuing to harass me."

complaint = st.text_area("Complaint", value=st.session_state.complaint_text,
                          height=200, placeholder="Describe your complaint here...")

if st.button("Analyze"):
    if complaint.strip():
        with st.spinner("Analyzing..."):
            dept, probas, summary = process_complaint(complaint)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        col1.metric("🏢 Department", dept)
        col2.metric("🎯 Confidence", f"{probas.max()*100:.1f}%")
        st.info(f"📝 **Summary:** {summary}")

        # Confidence bar chart
        st.subheader("Confidence by Department")
        fig, ax = plt.subplots(figsize=(8, 3))
        departments = le.classes_
        colors = ['#2ecc71' if d == dept else '#3498db' for d in departments]
        bars = ax.barh(departments, probas * 100, color=colors)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        for bar, prob in zip(bars, probas):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Please enter a complaint.")
