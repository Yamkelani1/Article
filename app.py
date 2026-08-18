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
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Article & Social Sentiment Analyzer",
    page_icon="📰",
    layout="wide"
)

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
    # Strip emojis / non-latin characters for PDF safety
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()


def fetch_youtube_comments(video_url, max_comments=30):
    """Fetch public comments from a YouTube video URL."""
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
    return comments


def analyze_with_openrouter(content_data, mode="text", api_key=""):
    """Send text or comments to OpenRouter for AI sentiment analysis and summarization."""
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
    else: # mode == "youtube"
        prompt = f"""
        Analyze these YouTube comments and categorize each comment individually.
        Provide a structured JSON response with:
        1. "overall_sentiment": "Positive", "Negative", or "Neutral"
        2. "summary": An executive summary highlighting key takeaways, main praise, and major complaints/pain points.
        3. "key_points": A list of 3 to 5 main recurring themes or talking points from viewers.
        4. "categorized_comments": A list of objects where EVERY input comment is included with its original "author", "text", "likes", and an assigned "sentiment" ("Positive", "Negative", or "Neutral").

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

    # Add Comments Table to PDF
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
st.title("📰 Article & Social Sentiment Analyzer")
st.markdown("Analyze raw article text or YouTube video audience feedback seamlessly using AI.")

analysis_mode = st.radio(
    "Select Input Mode:",
    ("Paste Raw Article Text", "Analyze YouTube Video Link"),
    horizontal=True
)

if analysis_mode == "Paste Raw Article Text":
    article_text = st.text_area("Paste Article Text Here", height=250)
else:
    youtube_url = st.text_input("Enter YouTube Video URL")

if st.button("Run Analysis"):
    if not openrouter_api_key:
        st.error("API Key missing from Secrets. Please set OPENROUTER_API_KEY in Streamlit Secrets.")
        st.stop()

    with st.spinner("Processing analysis via OpenRouter AI..."):
        try:
            if analysis_mode == "Paste Raw Article Text":
                if not article_text.strip():
                    st.warning("Please paste some text first.")
                    st.stop()

                ai_output = analyze_with_openrouter(article_text, mode="text", api_key=openrouter_api_key)

                st.subheader("Analysis Results")
                sentiment = ai_output.get("overall_sentiment", "Neutral")

                if sentiment == "Positive":
                    st.success(f"Overall Sentiment: **{sentiment}**")
                elif sentiment == "Negative":
                    st.error(f"Overall Sentiment: **{sentiment}**")
                else:
                    st.info(f"Overall Sentiment: **{sentiment}**")

                st.markdown("### Executive Summary")
                st.write(ai_output.get("summary", ""))

                st.markdown("### Key Takeaways")
                for point in ai_output.get("key_points", []):
                    st.markdown(f"* {point}")

                pdf_bytes = generate_pdf(
                    title="Article Analysis Report",
                    summary=ai_output.get("summary", ""),
                    extra_info=ai_output.get("key_points", []),
                    sentiment=sentiment
                )
                st.download_button("📥 Download PDF Report", pdf_bytes, file_name="article_report.pdf", mime="application/pdf")

            else:
                if not youtube_url.strip():
                    st.warning("Please enter a valid YouTube URL first.")
                    st.stop()

                comments = fetch_youtube_comments(youtube_url)
                if not comments:
                    st.error("No comments found or couldn't fetch comments from this video.")
                    st.stop()

                ai_output = analyze_with_openrouter(comments, mode="youtube", api_key=openrouter_api_key)

                st.subheader("YouTube Audience Feedback Results")
                sentiment = ai_output.get("overall_sentiment", "Neutral")

                if sentiment == "Positive":
                    st.success(f"Overall Audience Sentiment: **{sentiment}**")
                elif sentiment == "Negative":
                    st.error(f"Overall Audience Sentiment: **{sentiment}**")
                else:
                    st.info(f"Overall Audience Sentiment: **{sentiment}**")

                st.markdown("### Executive Summary")
                st.write(ai_output.get("summary", ""))

                st.markdown("### Recurring Feedback Themes")
                for point in ai_output.get("key_points", []):
                    st.markdown(f"* {point}")

                cat_comments = ai_output.get("categorized_comments", comments)

                # Generate PDF with full comments table safely sanitized
                pdf_bytes = generate_pdf(
                    title="YouTube Audience Analysis Report",
                    summary=ai_output.get("summary", ""),
                    extra_info=ai_output.get("key_points", []),
                    sentiment=sentiment,
                    comments_data=cat_comments
                )
                st.download_button("📥 Download PDF Report", pdf_bytes, file_name="youtube_report.pdf", mime="application/pdf")

                st.markdown("---")
                st.subheader("💬 Public Comments & Sentiment Breakdown")

                df = pd.DataFrame(cat_comments)

                if "sentiment" not in df.columns:
                    df["sentiment"] = "Neutral"

                tab_all, tab_pos, tab_neu, tab_neg = st.tabs(["All Comments", "Positive 😀", "Neutral 😐", "Negative 😡"])

                with tab_all:
                    st.dataframe(df[["author", "text", "likes", "sentiment"]], use_container_width=True)

                with tab_pos:
                    pos_df = df[df["sentiment"].str.lower() == "positive"]
                    if not pos_df.empty:
                        st.dataframe(pos_df[["author", "text", "likes"]], use_container_width=True)
                    else:
                        st.info("No positive comments detected.")

                with tab_neu:
                    neu_df = df[df["sentiment"].str.lower() == "neutral"]
                    if not neu_df.empty:
                        st.dataframe(neu_df[["author", "text", "likes"]], use_container_width=True)
                    else:
                        st.info("No neutral comments detected.")

                with tab_neg:
                    neg_df = df[df["sentiment"].str.lower() == "negative"]
                    if not neg_df.empty:
                        st.dataframe(neg_df[["author", "text", "likes"]], use_container_width=True)
                    else:
                        st.info("No negative comments detected.")

        except Exception as e:
            st.error(f"Error executing analysis: {e}")