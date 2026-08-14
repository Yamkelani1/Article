import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import re
from collections import Counter

# Page setup
st.set_page_config(page_title="Article Analyzer Pro", layout="wide")

st.title("Article Analyzer Pro")
st.write("Paste any article below to analyze sentiment, generate a summary, and download PDF reports.")

# Text input
article_text = st.text_area("Paste Article Text Here", height=250)

def generate_lightweight_summary(text, num_sentences=3):
    """Generates an extractive summary using word-frequency ranking (No PyTorch needed!)."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if len(sentences) <= num_sentences:
        return text

    # Word frequency map
    words = re.findall(r'\w+', text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'it', 'this', 'that', 'are', 'was', 'as'}
    filtered_words = [w for w in words if w not in stop_words]
    freq = Counter(filtered_words)

    # Score sentences
    scores = {}
    for i, sent in enumerate(sentences):
        score = sum(freq[w.lower()] for w in re.findall(r'\w+', sent) if w.lower() in freq)
        scores[i] = score

    # Get top N sentences in original order
    top_indices = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    top_indices.sort()

    return " ".join([sentences[i] for i in top_indices])

def generate_pdf(text, summary, sentiment, polarity, subjectivity):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Article Analysis Report")

    p.setFont("Helvetica", 12)
    p.drawString(100, 720, f"Overall Sentiment: {sentiment}")
    p.drawString(100, 700, f"Polarity Score: {polarity:.2f}")
    p.drawString(100, 680, f"Subjectivity Score: {subjectivity:.2f}")

    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 640, "Summary:")

    p.setFont("Helvetica", 10)
    y = 620
    for line in summary.split(". "):
        if y < 100:
            break
        p.drawString(100, y, line[:80] + ".")
        y -= 15

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

if st.button("Analyze Article"):
    if not article_text.strip():
        st.warning("Please paste some text first!")
    else:
        # Sentiment Analysis
        blob = TextBlob(article_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            sentiment = "Positive"
            color = "green"
        elif polarity < -0.1:
            sentiment = "Negative"
            color = "red"
        else:
            sentiment = "Neutral"
            color = "gray"

        # Generate Summary
        summary = generate_lightweight_summary(article_text, num_sentences=3)

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Sentiment Category", value=sentiment)
            st.metric(label="Polarity Score", value=f"{polarity:.2f}")
            st.metric(label="Subjectivity Score", value=f"{subjectivity:.2f}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = (polarity + 1) * 50,
                title = {'text': "Sentiment Gauge (0-100)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps' : [
                        {'range': [0, 40], 'color': "#ffcccc"},
                        {'range': [40, 60], 'color': "#e6e6e6"},
                        {'range': [60, 100], 'color': "#ccffcc"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Article Summary")
        st.info(summary)

        pdf_data = generate_pdf(article_text, summary, sentiment, polarity, subjectivity)
        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="article_analysis.pdf",
            mime="application/pdf"
        )