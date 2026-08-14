from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

def generate_pdf(text, summary, sentiment, polarity, subjectivity):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=15,
        bold=True
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        bold=True
    )
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    story = []

    # Title
    story.append(Paragraph("Article Analysis Report", title_style))
    story.append(Spacer(1, 10))

    # Metrics Section
    story.append(Paragraph(f"<b>Overall Sentiment:</b> {sentiment}", body_style))
    story.append(Paragraph(f"<b>Polarity Score:</b> {polarity:.2f}", body_style))
    story.append(Paragraph(f"<b>Subjectivity Score:</b> {subjectivity:.2f}", body_style))
    story.append(Spacer(1, 15))

    # Summary Section
    story.append(Paragraph("Summary:", heading_style))
    story.append(Paragraph(summary, body_style))

    # Build PDF with automatic page-wrapping
    doc.build(story)
    buffer.seek(0)
    return buffer