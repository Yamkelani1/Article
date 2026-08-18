import io
import json
import re
import requests
import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from youtube_comment_downloader import YoutubeCommentDownloader

# ==========================================
# PAGE CONFIGURATION & STYLING FIXES
# ==========================================
st.set_page_config(
    page_title="YouTube Sentiment Dashboard",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS ensuring crisp contrast, explicit text colors, and visible radio buttons
st.markdown("""
<style>
    /* Global Page Body & Background */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    
    /* Hero Header Styling */
    .hero-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        color: #1E3A8A !important;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #475569 !important;
        font-size: 1.15rem !important;
        font-weight: 500;
        margin-bottom: 2rem;
    }

    /* Force Radio Button Text Visibility */
    div[data-testid="stRadio"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        background-color: #FFFFFF !important;
        padding: 0.5rem 1rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #CBD5E1 !important;
    }

    /* Input Box Contrast Fix */
    div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 0.5rem !important;
        font-size: 1rem !important;
    }
    div[data-baseweb="input"] input::placeholder, div[data-baseweb="textarea"] textarea::placeholder {
        color: #94A3B8 !important;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        padding: 1.25rem !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #1E3A8A !important;
        font-weight: 800 !important;
    }

    /* Qualitative Cards */
    .quote-card {
        background-color: #FFFFFF;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: #1E293B;
        font-size: 0.95rem;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Report Container */
    .report-card {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# API Key Retrieval
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", "")
if not openrouter_api_key:
    import os
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_text_for_pdf(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()


def fetch_youtube_comments(video_url, max_comments=30):
    """Fetch public comments with detailed error logging."""
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        generator = downloader.get_comments_from_url(video_url, sort_by=0)
        for count, comment in enumerate(generator):
            if count >= max_comments:
                break
            comments.append({
                "author": comment.get("author", "Anonymous"),
                "text": comment.get("text", ""),
                "likes": comment.get("votes", 0)
            })
    except Exception as e:
        st.error(f"Failed to fetch YouTube comments: {str(e)}")
        return None
    return comments


def analyze_with_openrouter(content_data, mode="text", api_key=""):
    if mode == "text":
        prompt = f"""
        Analyze the following article text and provide a structured JSON response with:
        1. "overall_sentiment": "Positive", "Negative", or "Neutral"
        2. "summary": A concise executive summary of the article (3-5 sentences).
        3. "key_points": A list of 3 to 5 key bullet points.

        Article Text:
        {content_data}
        
        Respond STRICTLY with valid JSON.
        """
    else:
        prompt = f"""
        Analyze these YouTube comments and categorize each comment individually while extracting qualitative themes.
        Provide a structured JSON response with:
        1. "overall_sentiment": "Positive", "Negative", or "Neutral"
        2. "summary": An executive summary highlighting key takeaways, main praise, and major complaints.
        3. "key_points": A list of 3 to 5 main recurring themes.
        4. "most_interesting": A list of up to 3 highly engaging or insightful comment strings.
        5. "hot_takes": A list of up to 3 controversial, strong, or provocative opinions from comments.
        6. "questions": A list of up to 3 viewer questions raised in comments.
        7. "recommendations": A list of 3 strategic recommendations based on audience response.
        8. "categorized_comments": A list of objects where EVERY input comment is included with "author", "text", "likes", and "sentiment" ("Positive", "Negative", "Neutral").

        Comments Data:
        {json.dumps(content_data)}

        Respond STRICTLY with valid JSON.
        """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenRouter API Error ({response.status_code}): {response.text}")

    result = response.json()
    response_text = result["choices"][0]["message"]["content"]
    return json.loads(response_text)


def generate_pdf(title, summary, extra_info=None, sentiment=None, comments_data=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1E3A8A'))
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2563EB'))
    body_style = styles['Normal']
    table_cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontSize=8, leading=10)
    table_header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=9, leading=11, fontName='Helvetica-Bold', textColor=colors.white)

    elements = [
        Paragraph(clean_text_for_pdf(title), title_style),
        Spacer(1, 10)
    ]

    if sentiment:
        sentiment_style = ParagraphStyle('SentimentStyle', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold')
        elements.append(Paragraph(f"<b>Overall Audience Sentiment:</b> {clean_text_for_pdf(sentiment)}", sentiment_style))
        elements.append(Spacer(1, 10))

    elements.extend([
        Paragraph("Executive Summary", heading_style),
        Spacer(1, 6),
        Paragraph(clean_text_for_pdf(summary), body_style),
        Spacer(1, 10)
    ])

    if extra_info:
        elements.append(Paragraph("Key Takeaways / Highlights", heading_style))
        elements.append(Spacer(1, 6))
        for item in extra_info:
            elements.append(Paragraph(f"• {clean_text_for_pdf(item)}", body_style))
            elements.append(Spacer(1, 4))
        elements.append(Spacer(1, 12))

    if comments_data:
        elements.append(Paragraph("Public Comments & Sentiment Breakdown", heading_style))
        elements.append(Spacer(1, 8))

        table_data = [[
            Paragraph("Author", table_header_style),
            Paragraph("Comment Text", table_header_style),
            Paragraph("Likes", table_header_style),
            Paragraph("Sentiment", table_header_style)
        ]]

        for item in comments_data:
            clean_author = clean_text_for_pdf(item.get("author", ""))
            clean_comment = clean_text_for_pdf(item.get("text", ""))
            clean_likes = str(item.get("likes", 0))
            clean_sent = clean_text_for_pdf(item.get("sentiment", "Neutral"))

            table_data.append([
                Paragraph(clean_author if clean_author else "Anonymous", table_cell_style),
                Paragraph(clean_comment if clean_comment else "[Emoji/Media]", table_cell_style),
                Paragraph(clean_likes, table_cell_style),
                Paragraph(clean_sent, table_cell_style)
            ])

        pdf_table = Table(table_data, colWidths=[100, 280, 40, 60], repeatRows=1)
        pdf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))

        elements.append(pdf_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# UI & FORM EXECUTION
# ==========================================
st.markdown('<div class="hero-title">YouTube Sentiment Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Analyze YouTube comments to understand audience sentiment and engagement patterns with AI.</div>', unsafe_allow_html=True)

# Using st.form ensures input stability and prevents premature re-renders
with st.form("analysis_form"):
    analysis_mode = st.radio(
        "Select Mode:",
        ("Analyze YouTube Video Link", "Paste Raw Article Text"),
        horizontal=True
    )

    youtube_url = st.text_input("YouTube Link", placeholder="https://www.youtube.com/watch?v=...")
    article_text = st.text_area("Raw Text (if using Raw Text mode)", placeholder="Paste article text...", height=120)

    submit_button = st.form_submit_button("🚀 Run Dashboard Analysis", use_container_width=True)

if submit_button:
    if not openrouter_api_key:
        st.error("API Key missing. Please set OPENROUTER_API_KEY in Streamlit Secrets.")
        st.stop()

    if analysis_mode == "Analyze YouTube Video Link":
        if not youtube_url.strip():
            st.warning("Please enter a YouTube video URL.")
            st.stop()

        with st.spinner("Fetching comments from YouTube..."):
            comments = fetch_youtube_comments(youtube_url.strip())

        if comments is None:
            st.stop()
        elif len(comments) == 0:
            st.error("No public comments could be retrieved from this URL. Ensure comments are enabled on the video.")
            st.stop()

        with st.spinner("Analyzing comments with AI..."):
            try:
                ai_output = analyze_with_openrouter(comments, mode="youtube", api_key=openrouter_api_key)
            except Exception as err:
                st.error(f"AI Processing failed: {err}")
                st.stop()

        cat_comments = ai_output.get("categorized_comments", comments)
        df = pd.DataFrame(cat_comments)

        st.success(f"Analysis Complete! Processed {len(comments)} comments.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Comments Analyzed", len(comments))
        with col2:
            st.metric("Dominant Sentiment", ai_output.get("overall_sentiment", "Neutral"))
        with col3:
            total_likes = df["likes"].sum() if "likes" in df.columns else 0
            st.metric("Total Comment Likes", total_likes)

        st.markdown("---")

        st.subheader("💬 Comment Categories")
        tab1, tab2, tab3 = st.tabs(["⭐ Most Interesting", "🔥 Hot Takes", "❓ Questions"])

        with tab1:
            for item in ai_output.get("most_interesting", []):
                st.markdown(f'<div class="quote-card">"{item}"</div>', unsafe_allow_html=True)
        with tab2:
            for item in ai_output.get("hot_takes", []):
                st.markdown(f'<div class="quote-card">"{item}"</div>', unsafe_allow_html=True)
        with tab3:
            for item in ai_output.get("questions", []):
                st.markdown(f'<div class="quote-card">"{item}"</div>', unsafe_allow_html=True)

        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Analysis Report")
        st.write(ai_output.get("summary", ""))

        st.markdown("**Key Takeaways:**")
        for pt in ai_output.get("key_points", []):
            st.markdown(f"* {pt}")

        st.markdown("**Strategic Recommendations:**")
        for idx, rec in enumerate(ai_output.get("recommendations", []), 1):
            st.markdown(f"{idx}. {rec}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("📋 Public Comments Breakdown")
        st.dataframe(df[["author", "text", "likes", "sentiment"]], use_container_width=True)

        pdf_bytes = generate_pdf("YouTube Audience Analysis Report", ai_output.get("summary", ""), ai_output.get("key_points", []), ai_output.get("overall_sentiment", "Neutral"), cat_comments)
        st.download_button("📥 Download PDF Report", pdf_bytes, file_name="youtube_analysis_report.pdf", mime="application/pdf")

    else:
        if not article_text.strip():
            st.warning("Please paste article text.")
            st.stop()

        with st.spinner("Analyzing text..."):
            ai_output = analyze_with_openrouter(article_text, mode="text", api_key=openrouter_api_key)

        st.subheader("Analysis Results")
        st.info(f"Overall Sentiment: **{ai_output.get('overall_sentiment', 'Neutral')}**")
        st.write(ai_output.get("summary", ""))

        pdf_bytes = generate_pdf("Article Analysis Report", ai_output.get("summary", ""), ai_output.get("key_points", []), ai_output.get("overall_sentiment", "Neutral"))
        st.download_button("📥 Download PDF Report", pdf_bytes, file_name="article_report.pdf", mime="application/pdf")