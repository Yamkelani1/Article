import streamlit as st
import requests
import pandas as pd
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader
import json

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# Page Setup
st.set_page_config(page_title="Article & Social Sentiment Analyzer", layout="wide")

st.title("Article & Public Link Sentiment Analyzer")
st.write("Analyze pasted article text or public YouTube video comments using AI-powered sentiment analysis and executive reporting.")

# Sidebar Configuration
st.sidebar.header("Configuration")
openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key to power AI sentiment analysis.")

# Input Mode Switcher
analysis_mode = st.radio(
    "Select Input Type:",
    ["Paste Raw Article Text", "Analyze YouTube Video Link"],
    horizontal=True
)

if analysis_mode == "Paste Raw Article Text":
    article_text = st.text_area("Paste Article Text Here", height=250, placeholder="Paste your article or long-form text here...")
else:
    youtube_url = st.text_input("Paste YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("Number of comments to fetch & analyze", min_value=5, max_value=50, value=15)

def fetch_youtube_comments(url, max_results):
    """Extracts public YouTube comments, author handles, and like counts."""
    downloader = YoutubeCommentDownloader()
    comments_generator = downloader.get_comments_from_url(url, sort_by=0)

    parsed_comments = []
    for comment in islice(comments_generator, max_results):
        parsed_comments.append({
            "author": comment.get("author"),
            "text": comment.get("text"),
            "likes": comment.get("votes", 0)
        })
    return parsed_comments

def analyze_with_openrouter(content_data, mode, api_key):
    """Sends content or YouTube comments to OpenRouter for AI sentiment categorization and executive summary."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if mode == "text":
        prompt = f"""
        Analyze the following article text:
        "{content_data}"

        Return a JSON object containing:
        1. "overall_sentiment": "Positive", "Negative", or "Neutral"
        2. "polarity_score": A float between -1.0 (most negative) and 1.0 (most positive)
        3. "subjectivity_score": A float between 0.0 (objective) and 1.0 (subjective)
        4. "executive_summary": A concise, well-structured summary highlighting key points and main takeaways.
        
        Respond STRICTLY with valid JSON.
        """
    else:
        prompt = f"""
        Analyze the following YouTube video comments:
        {json.dumps(content_data)}

        Return a JSON object containing:
        1. "executive_summary": A detailed Executive Summary highlighting key takeaways, main praise from viewers, and major customer complaints or pain points.
        2. "categorized_comments": A list of objects with keys: "author", "text", "likes", and "sentiment" ("Positive", "Negative", or "Neutral").
        
        Respond STRICTLY with valid JSON.
        """

    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenRouter API Error ({response.status_code}): {response.text}")

    res_json = response.json()
    return json.loads(res_json['choices'][0]['message']['content'])

def generate_pdf(title, summary, extra_info=None):
    """Generates a clean PDF report with automatic text wrapping."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=15, bold=True)
    heading_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=8, bold=True)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=6)

    story = [
        Paragraph(title, title_style),
        Spacer(1, 10),
        Paragraph("Executive Summary", heading_style),
        Paragraph(summary, body_style),
        Spacer(1, 15)
    ]

    if extra_info:
        story.append(Paragraph("Key Metrics / Overview", heading_style))
        for line in extra_info:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

if st.button("Run Analysis"):
    if not openrouter_api_key:
        st.warning("Please enter your OpenRouter API Key in the sidebar.")
    else:
        with st.spinner("Processing analysis via OpenRouter AI..."):
            try:
                if analysis_mode == "Paste Raw Article Text":
                    if not article_text.strip():
                        st.warning("Please paste some text first.")
                        st.stop()

                    ai_output = analyze_with_openrouter(article_text, "text", openrouter_api_key)
                    sentiment = ai_output.get("overall_sentiment", "Neutral")
                    polarity = ai_output.get("polarity_score", 0.0)
                    subjectivity = ai_output.get("subjectivity_score", 0.0)
                    summary = ai_output.get("executive_summary", "No summary available.")

                    st.subheader("📊 Sentiment Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Overall Sentiment", sentiment)
                    col2.metric("Polarity Score", f"{polarity:.2f}")
                    col3.metric("Subjectivity Score", f"{subjectivity:.2f}")

                    st.markdown("---")
                    st.subheader("📝 Article Summary")
                    st.info(summary)

                    pdf_extra = [
                        f"<b>Overall Sentiment:</b> {sentiment}",
                        f"<b>Polarity Score:</b> {polarity:.2f}",
                        f"<b>Subjectivity Score:</b> {subjectivity:.2f}"
                    ]
                    pdf_data = generate_pdf("Article Analysis Report", summary, pdf_extra)

                else:
                    if not youtube_url.strip():
                        st.warning("Please enter a valid YouTube video URL.")
                        st.stop()

                    comments = fetch_youtube_comments(youtube_url, max_comments)
                    if not comments:
                        st.error("No comments retrieved. Ensure the video URL is valid and public.")
                        st.stop()

                    ai_output = analyze_with_openrouter(comments, "youtube", openrouter_api_key)
                    summary = ai_output.get("executive_summary", "No summary available.")
                    categorized = ai_output.get("categorized_comments", [])
                    df = pd.DataFrame(categorized)

                    st.subheader("📊 Executive Summary")
                    st.info(summary)

                    st.markdown("---")
                    st.subheader("💬 Viewer Comments & Sentiment Breakdown")

                    tab_all, tab_pos, tab_neu, tab_neg = st.tabs(["All Comments", "Positive", "Neutral", "Negative"])

                    with tab_all:
                        st.dataframe(df, use_container_width=True)
                    with tab_pos:
                        st.dataframe(df[df['sentiment'] == 'Positive'] if not df.empty else df, use_container_width=True)
                    with tab_neu:
                        st.dataframe(df[df['sentiment'] == 'Neutral'] if not df.empty else df, use_container_width=True)
                    with tab_neg:
                        st.dataframe(df[df['sentiment'] == 'Negative'] if not df.empty else df, use_container_width=True)

                    pos_c = len(df[df['sentiment'] == 'Positive']) if not df.empty else 0
                    neu_c = len(df[df['sentiment'] == 'Neutral']) if not df.empty else 0
                    neg_c = len(df[df['sentiment'] == 'Negative']) if not df.empty else 0

                    pdf_extra = [
                        f"<b>Total Comments Analyzed:</b> {len(df)}",
                        f"<b>Positive Comments:</b> {pos_c}",
                        f"<b>Neutral Comments:</b> {neu_c}",
                        f"<b>Negative Comments:</b> {neg_c}"
                    ]
                    pdf_data = generate_pdf("YouTube Feedback & Sentiment Report", summary, pdf_extra)

                st.markdown("---")
                st.download_button(
                    label="Download Executive PDF Report",
                    data=pdf_data,
                    file_name="sentiment_analysis_report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error executing analysis: {str(e)}")