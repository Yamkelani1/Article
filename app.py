import streamlit as st

def analyze_article(url):
    return f"Summary for {url}"

st.title("Article Analyser")
url = st.text_input("Enter Article URL:")

if st.button("Analyze"):
    if url:
        result = analyze_article(url)
        st.write(result)

# Load environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

# Initialize Hugging Face sentiment pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


def analyze_sentiment(text):
    """Analyze sentiment using local Hugging Face model."""
    if not text.strip():
        return "Please enter some text to analyze."
    result = sentiment_analyzer(text[:512])[0]
    label = result["label"]
    score = result["score"]
    return f"Sentiment: {label}\nConfidence: {score:.2%}"


def summarize_with_llm(text):
    """Summarize text using OpenRouter free LLM."""
    if not text.strip():
        return "Please enter some text to summarize."
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not set. Please add it to your .env file."

    with OpenRouter(api_key=OPENROUTER_API_KEY) as client:
        response = client.chat.send(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Summarize the following text concisely in 2-3 sentences.",
                },
                {"role": "user", "content": text},
            ],
            stream=False,
        )
    return response.choices[0].message.content

def full_analysis(text):
    """Run both sentiment analysis and summarization."""
    if not text.strip():
        return "Please enter some text to analyze."
    sentiment_result = analyze_sentiment(text)
    summary_result = summarize_with_llm(text)
    return f"--- SENTIMENT ---\n{sentiment_result}\n\n--- SUMMARY ---\n{summary_result}"

# Test
# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Article Analyzer\nPaste any article to get AI-powered sentiment analysis and summarization.")

    with gr.Tab("Summarize"):
        summary_input = gr.Textbox(
            label="Article Text",
            placeholder="Paste your article here...",
            lines=8,
        )
        summary_button = gr.Button("Summarize")
        summary_output = gr.Textbox(label="Summary", lines=4)
        summary_button.click(
            fn=summarize_with_llm, inputs=summary_input, outputs=summary_output
        )
        with gr.Tab("Sentiment Analysis"):
         sentiment_input = gr.Textbox(
         label="Article Text",
         placeholder="Paste your article here...",
         lines=8,
    )
    sentiment_button = gr.Button("Analyze Sentiment")
    sentiment_output = gr.Textbox(label="Sentiment Result", lines=2)
    sentiment_button.click(
        fn=analyze_sentiment, inputs=sentiment_input, outputs=sentiment_output
    )
    with gr.Tab("Full Analysis"):
        full_input = gr.Textbox(
            label="Article Text",
            placeholder="Paste your article here...",
            lines=8,
        )
    full_button = gr.Button("Run Full Analysis")
    full_output = gr.Textbox(label="Combined Results", lines=8)
    full_button.click(
        fn=full_analysis, inputs=full_input, outputs=full_output
    )

demo.launch(share=True)