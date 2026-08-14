import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Page setup
st.set_page_config(page_title="Article Analyzer Pro", layout="wide")

st.title("Article Analyzer Pro")
st.write("Paste any article below to analyze its sentiment and generate downloadable PDF reports.")

# Text input
article_text = st.text_area("Paste Article Text Here", height=250)

def generate_pdf(text, sentiment, polarity, subjectivity):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Article Analysis Report")

    p.setFont("Helvetica", 12)
    p.drawString(100, 720, f"Overall Sentiment: {sentiment}")
    p.drawString(100, 700, f"Polarity Score: {polarity:.2f}")
    p.drawString(100, 680, f"Subjectivity Score: {subjectivity:.2f}")

    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 640, "Article Preview:")

    p.setFont("Helvetica", 10)
    preview = text[:500] + "..." if len(text) > 500 else text

    # Simple line wrapping
    y = 620
    for line in preview.split("\n"):
        if y < 100:
            break
        p.drawString(100, y, line[:80])
        y -= 15

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

if st.button("Analyze Article"):
    if not article_text.strip():
        st.warning("Please paste some text first!")
    else:
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

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Sentiment Category", value=sentiment)
            st.metric(label="Polarity Score", value=f"{polarity:.2f}")
            st.metric(label="Subjectivity Score", value=f"{subjectivity:.2f}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = (polarity + 1) * 50, # Scale -1..1 to 0..100
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

        pdf_data = generate_pdf(article_text, sentiment, polarity, subjectivity)
        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="article_analysis.pdf",
            mime="application/pdf"
        )