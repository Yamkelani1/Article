import os
from openai import OpenAI  # OpenRouter uses the standard OpenAI client SDK
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Article Analyzer", layout="centered")

# Retrieve API Key securely from Streamlit Secrets or Environment Variables
OPENROUTER_API_KEY = st.secrets.get(
    "OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")
)

# Initialize Hugging Face sentiment pipeline with caching


@st.cache_resource
def load_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )


sentiment_analyzer = load_sentiment_analyzer()


def analyze_sentiment(text):
    """Analyze sentiment using local Hugging Face model."""
    if not text.strip():
        return "Please enter some text to analyze."
    result = sentiment_analyzer(text[:512])[0]
    label = result["label"]
    score = result["score"]
    return f"**Sentiment:** {label}\n\n**Confidence:** {score:.2%}"


def summarize_with_llm(text):
    """Summarize text using OpenRouter free LLM via OpenAI SDK."""
    if not text.strip():
        return "Please enter some text to summarize."
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not set in Secrets/Environment."

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Summarize the following text concisely in 2-3 sentences.",
                },
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"


# --- UI LAYOUT ---
st.title("Article Analyzer")
st.caption(
    "Paste any article to get AI-powered sentiment analysis and summarization."
)

tab1, tab2, tab3 = st.tabs(["Summarize", "Sentiment Analysis", "Full Analysis"])

# Tab 1: Summarize
with tab1:
    summary_input = st.text_area(
        "Article Text",
        placeholder="Paste your article here...",
        height=200,
        key="sum_in",
    )
    if st.button("Summarize", key="sum_btn"):
        with st.spinner("Generating summary..."):
            summary_output = summarize_with_llm(summary_input)
            st.write(summary_output)

# Tab 2: Sentiment
with tab2:
    sentiment_input = st.text_area(
        "Article Text",
        placeholder="Paste your article here...",
        height=200,
        key="sent_in",
    )
    if st.button("Analyze Sentiment", key="sent_btn"):
        with st.spinner("Analyzing sentiment..."):
            sentiment_output = analyze_sentiment(sentiment_input)
            st.markdown(sentiment_output)

# Tab 3: Full Analysis
with tab3:
    full_input = st.text_area(
        "Article Text",
        placeholder="Paste your article here...",
        height=200,
        key="full_in",
    )
    if st.button("Run Full Analysis", key="full_btn"):
        if not full_input.strip():
            st.warning("Please enter text first.")
        else:
            with st.spinner("Running full analysis..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentiment")
                    st.markdown(analyze_sentiment(full_input))
                with col2:
                    st.subheader("Summary")
                    st.write(summarize_with_llm(full_input))