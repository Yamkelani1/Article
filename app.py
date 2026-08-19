import io
import json
import re
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from youtube_comment_downloader import YoutubeCommentDownloader

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Article and YouTube Sentiment Analyzer",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1E1B4B 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .quote-card {
        background-color: #F8FAFC;
        border-left: 4px solid #7C3AED;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
        color: #334155;
        font-size: 0.95rem;
    }
    .report-card {
        background-color: #F5F3FF;
        border: 1px solid #DDD6FE;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load API Key from Streamlit Secrets or Environment
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", "")
if not openrouter_api_key:
    import os
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_text_for_pdf(text):
    """Remove emojis and non-standard ASCII characters that crash ReportLab PDF fonts."""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()


def fetch_youtube_comments(video_url, max_comments=100):
    """Fetch public comments from a YouTube video URL safely (up to max_comments)."""
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
        st.error(f"Error fetching YouTube comments: {e}")
        return []
    return comments


def analyze_with_openrouter(content_data, mode="text", api_key=""):
    """Send text or comments to OpenRouter for AI sentiment analysis and qualitative extraction."""
    if not content_data:
        raise ValueError("Content data provided for analysis is empty.")

    if mode == "text":
        prompt = f"""
        Analyze the following article text and provide a structured JSON response with:
        1. "overall_sentiment": "Positive", "Negative", or "Neutral"
        2. "confidence_score": A percentage value between 0% and 100% representing AI analytical confidence.
        3. "article_tone": A description of the overall tone (e.g., Authoritative, Critical, Informative, Persuasive, Conversational).
        4. "summary": A detailed, thorough executive summary (at least 6 to 8 sentences). Do NOT make it brief. Explicitly discuss the article's core narrative, the tone used by the author, key facts, and the confidence level of the analysis.
        5. "key_points": A list of 4 to 6 comprehensive bullet points.

        Article Text:
        {content_data}
        
        Respond STRICTLY with valid JSON.
        """
    else: # mode == "youtube"
        prompt = f"""
        Analyze these YouTube comments and categorize each comment individually while extracting qualitative themes.
        Provide a structured JSON response with:
        1. "overall_sentiment": "Positive", "Negative", or "Neutral"
        2. "confidence_score": A percentage value between 0% and 100% representing confidence in sentiment classification.
        3. "overall_tone": A description of the audience's tone (e.g., Enthusiastic, Critical, Sarcastic, Inquisitive).
        4. "summary": A detailed executive summary (at least 6 to 8 sentences). Discuss the general audience reaction, overall tone, main areas of praise or criticism, and the analytical confidence score.
        5. "key_points": A list of 4 to 6 main recurring themes.
        6. "most_interesting": A list of 12 to 20 highly engaging, detailed, or insightful comment quotes extracted from the data.
        7. "hot_takes": A list of 12 to 20 controversial, strong, or provocative opinion quotes extracted from the data.
        8. "questions": A list of 12 to 20 viewer questions raised in comments.
        9. "recommendations": A list of 3 to 5 strategic recommendations based on audience feedback.
        10. "categorized_comments": A list of objects where EVERY input comment is included with "author", "text", "likes", and "sentiment" ("Positive", "Negative", "Neutral").

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

    choices = result.get("choices", [])
    if not choices or not choices[0].get("message", {}).get("content"):
        raise ValueError("OpenRouter API returned an empty completion response.")

    response_text = choices[0]["message"]["content"]
    return json.loads(response_text)


def generate_pdf(title, summary, extra_info=None, sentiment=None, confidence=None, tone=None, comments_data=None):
    """Generate downloadable PDF report containing summary, points, and comment breakdown table."""
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
        meta_text = f"<b>Overall Sentiment:</b> {clean_text_for_pdf(sentiment)}"
        if confidence:
            meta_text += f" | <b>Confidence Score:</b> {clean_text_for_pdf(str(confidence))}"
        if tone:
            meta_text += f" | <b>Tone:</b> {clean_text_for_pdf(tone)}"

        sentiment_style = ParagraphStyle('SentimentStyle', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold')
        elements.append(Paragraph(meta_text, sentiment_style))
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
# MAIN STREAMLIT UI
# ==========================================
st.markdown('<div class="main-title">Article and YouTube Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze YouTube comments and raw article text to extract deep sentiment metrics, audience tone, and qualitative insights</div>', unsafe_allow_html=True)

# Mode Selector
analysis_mode = st.radio(
    "Select Input Mode:",
    ("Analyze YouTube Video Link", "Paste Raw Article Text"),
    horizontal=True
)

if analysis_mode == "Analyze YouTube Video Link":
    youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
else:
    article_text = st.text_area("Paste Article Text Here", height=200)

if st.button("Run Dashboard Analysis", type="primary"):
    if not openrouter_api_key:
        st.error("API Key missing from Secrets. Please set OPENROUTER_API_KEY in Streamlit Secrets.")
        st.stop()

    with st.spinner("Analyzing content and generating metrics..."):
        try:
            if analysis_mode == "Analyze YouTube Video Link":
                if not youtube_url.strip():
                    st.warning("Please enter a valid YouTube URL first.")
                    st.stop()

                comments = fetch_youtube_comments(youtube_url, max_comments=100)
                if not comments or len(comments) == 0:
                    st.error("No comments found or couldn't fetch comments from this video link. Verify comments are publicly enabled.")
                    st.stop()

                ai_output = analyze_with_openrouter(comments, mode="youtube", api_key=openrouter_api_key)
                cat_comments = ai_output.get("categorized_comments", comments)
                df = pd.DataFrame(cat_comments)

                # Status Banner
                st.success(f" Analysis Complete — Successfully fetched and analyzed {len(df)} video comments.")

                # Metric Cards Bar
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Comments Analyzed", len(df))
                with col2:
                    st.metric("Dominant Sentiment", ai_output.get("overall_sentiment", "Neutral"))
                with col3:
                    st.metric("Confidence Score", ai_output.get("confidence_score", "N/A"))
                with col4:
                    st.metric("Audience Tone", ai_output.get("overall_tone", "N/A"))

                st.markdown("---")

                # Qualitative Comment Categories Section
                st.subheader("💬 Qualitative Comment Categories")
                tab_interesting, tab_takes, tab_questions = st.tabs([
                    f"⭐ Most Interesting ({len(ai_output.get('most_interesting', []))})",
                    f"🔥 Hot Takes ({len(ai_output.get('hot_takes', []))})",
                    f"❓ Questions ({len(ai_output.get('questions', []))})"
                ])

                with tab_interesting:
                    interesting_items = ai_output.get("most_interesting", [])
                    if interesting_items:
                        for q in interesting_items:
                            st.markdown(f'<div class="quote-card">"{q}"</div>', unsafe_allow_html=True)
                    else:
                        st.info("No key interesting highlights categorized.")

                with tab_takes:
                    hot_items = ai_output.get("hot_takes", [])
                    if hot_items:
                        for q in hot_items:
                            st.markdown(f'<div class="quote-card">"{q}"</div>', unsafe_allow_html=True)
                    else:
                        st.info("No strong hot takes categorized.")

                with tab_questions:
                    question_items = ai_output.get("questions", [])
                    if question_items:
                        for q in question_items:
                            st.markdown(f'<div class="quote-card">"{q}"</div>', unsafe_allow_html=True)
                    else:
                        st.info("No explicit audience questions detected.")

                # Full Executive AI Analysis Report Box
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("🤖 Detailed AI Analysis Report")

                st.markdown("**Executive Summary:**")
                st.write(ai_output.get("summary", ""))

                st.markdown("**Key Takeaways:**")
                for pt in ai_output.get("key_points", []):
                    st.markdown(f"* {pt}")

                rec_list = ai_output.get("recommendations", [])
                if rec_list:
                    st.markdown("**Strategic Recommendations:**")
                    for idx, rec in enumerate(rec_list, 1):
                        st.markdown(f"{idx}. {rec}")

                st.markdown('</div>', unsafe_allow_html=True)

                # Sentiment Distribution Pie Chart
                st.subheader("📊 Sentiment Breakdown Chart")
                if "sentiment" in df.columns and not df.empty:
                    sentiment_counts = df["sentiment"].value_counts().reset_index()
                    sentiment_counts.columns = ["Sentiment", "Count"]

                    color_map = {"Positive": "#22C55E", "Neutral": "#64748B", "Negative": "#EF4444"}

                    fig = px.pie(
                        sentiment_counts,
                        names="Sentiment",
                        values="Count",
                        title="Audience Sentiment Distribution",
                        color="Sentiment",
                        color_discrete_map=color_map,
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                # Raw Comments Data Table & PDF Download
                st.subheader(f"📋 Public Comments Breakdown Table ({len(df)} Comments)")
                st.dataframe(df[["author", "text", "likes", "sentiment"]], use_container_width=True)

                pdf_bytes = generate_pdf(
                    title="YouTube Audience Analysis Report",
                    summary=ai_output.get("summary", ""),
                    extra_info=ai_output.get("key_points", []),
                    sentiment=ai_output.get("overall_sentiment", "Neutral"),
                    confidence=ai_output.get("confidence_score", "N/A"),
                    tone=ai_output.get("overall_tone", "N/A"),
                    comments_data=cat_comments
                )
                st.download_button("📥 Download PDF Executive Report", pdf_bytes, file_name="youtube_analysis_report.pdf", mime="application/pdf")

            else:
                # Raw Text Mode
                if not article_text.strip():
                    st.warning("Please paste some article text first.")
                    st.stop()

                ai_output = analyze_with_openrouter(article_text, mode="text", api_key=openrouter_api_key)

                st.success(" Analysis Complete — Article content evaluated.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Sentiment", ai_output.get("overall_sentiment", "Neutral"))
                with col2:
                    st.metric("Confidence Score", ai_output.get("confidence_score", "N/A"))
                with col3:
                    st.metric("Article Tone", ai_output.get("article_tone", "N/A"))

                # Detailed Summary Section
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("📝 Executive Summary & Tone Analysis")
                st.write(ai_output.get("summary", ""))

                st.markdown("### Key Takeaways")
                for point in ai_output.get("key_points", []):
                    st.markdown(f"* {point}")
                st.markdown('</div>', unsafe_allow_html=True)

                pdf_bytes = generate_pdf(
                    title="Article Analysis Report",
                    summary=ai_output.get("summary", ""),
                    extra_info=ai_output.get("key_points", []),
                    sentiment=ai_output.get("overall_sentiment", "Neutral"),
                    confidence=ai_output.get("confidence_score", "N/A"),
                    tone=ai_output.get("article_tone", "N/A")
                )
                st.download_button("📥 Download PDF Report", pdf_bytes, file_name="article_report.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Error executing analysis: {e}")