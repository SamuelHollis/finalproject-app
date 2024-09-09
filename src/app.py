import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download and load the model and tokenizer locally
@st.cache_resource
def load_local_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load the local model
sentiment_analysis = load_local_model()

# Mapping RoBERTa labels to understandable sentiments
label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}

# Function to analyze sentiments in a DataFrame (CSV)
def analyze_sentiments_chunked(df):
    sentiment_list = []
    score_list = []

    for idx, text in enumerate(df['text']):
        logging.info(f"Processing comment {idx + 1}/{len(df)}")
        try:
            # Execute sentiment analysis
            result = sentiment_analysis(text)
            # Get the sentiment and score
            sentiment = result[0]['label']
            score = result[0]['score']
        except Exception as e:
            logging.error(f"Error processing text: {e}")
            sentiment = "error"
            score = 0.0

        sentiment_list.append(sentiment)
        score_list.append(score)

    # Add the results to the DataFrame
    df['sentiment'] = sentiment_list
    df['score'] = score_list
    return df

# Function to calculate and display sentiment percentages
def calculate_sentiment_percentages(df):
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    sentiments = ['LABEL_0', 'LABEL_1', 'LABEL_2']  # LABEL_0: Negative, LABEL_1: Neutral, LABEL_2: Positive
    percentages = [sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
    return percentages

# CSS for a modern and clean look
page_bg_css = '''
<style>
body {
    background: url("https://www.omfif.org/wp-content/uploads/2024/01/GettyImages-1183053829.jpg"); /* Background image */
    background-size: cover;
    background-position: cover;
    background-repeat: no-repeat;
    font-family: 'Helvetica Neue', sans-serif;
    opacity: 0.7; /* Slight opacity to blend the background */
}

[data-testid="stAppViewContainer"] {
    background: rgba(0, 0, 0, 0.7); /* Darker overlay for better readability */
    background-blend-mode: overlay;
    padding: 2rem;
    color: white; /* Ensure text is white and more visible */
}

h1 {
    color: #B22222; /* Firebrick for the title */
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
    opacity: 1; /* Fully opaque title */
    background-color: rgba(255, 255, 255, 0.4); /* Semi-transparent white background */
    padding: 4px;
    border-radius: 10px; /* Rounded corners */
    display: inline-block; /* Shrink background to text width */
}

h2, h3 {
    color: white; /* White text for subtitles */
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
}

.stButton>button {
    background-color: #1E90FF; /* DodgerBlue */
    color: white;
    font-size: 18px;
    border-radius: 12px; /* Rounded corners */
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15); /* Soft shadow */
}

.stButton>button:hover {
    background-color: #1E90FF; /* Lighter blue on hover */
    transform: scale(1.05); /* Subtle zoom effect */
}

.stTextArea textarea {
    background-color: rgba(107, 107, 107, 0.9); /* More opaque gray for the text area */
    border-radius: 12px;
    font-size: 16px;
    padding: 15px;
    color: white; /* White text */
}

footer {
    visibility: hidden;
}

.result-card {
    background-color: rgba(107, 107, 107, 0.9); /* More opaque dark gray for the result cards */
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    color: white; /* White text for the result cards */
}

.card-header {
    font-size: 24px;
    font-weight: bold;
    color: #1E90FF; /* Blue header for the result card */
    margin-bottom: 15px;
}
</style>
'''

# Inject CSS into the application
st.markdown(page_bg_css, unsafe_allow_html=True)

# Title of the application
st.title("Sentiment Analysis")

# Section 1: CSV File Analysis
st.subheader("📂 Analyze CSV File")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    # Cargar el archivo CSV sin incluir el índice como columna
    df = pd.read_csv(uploaded_file, index_col=None)

    # Display the first few records of the CSV
    st.write("First 5 comments from the file:")
    st.write(df.head())

    # Button to execute sentiment analysis on the CSV
    if st.button("🔍 Analyze Sentiments in CSV"):
        if 'text' not in df.columns:
            st.error("The CSV file must contain a 'text' column.")
        else:
            with st.spinner("🔄 Analyzing sentiments, please wait..."):
                analyzed_df = analyze_sentiments_chunked(df)

            st.success("✅ Analysis complete!")

            # Display results
            st.write("Analysis Results:")
            st.write(analyzed_df.head())

            # Calculate and display sentiment percentages
            percentages = calculate_sentiment_percentages(analyzed_df)
            labels = ['Negative', 'Neutral', 'Positive']
            colors = ['#FF6B6B', '#F7D794', '#4CAF50']  # Colors for negative, neutral, positive

            # Create a bar chart
            fig, ax = plt.subplots()
            ax.barh(labels, percentages, color=colors)
            ax.set_xlabel('Percentage (%)')
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)

            # Download the results as a CSV without an index
            csv = analyzed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv',
            )

# Section 2: Individual Sentence Analysis
st.subheader("📝 Analyze a Single Sentence")
user_input = st.text_area("Write a sentence to analyze", "")

if st.button("📊 Analyze Sentence"):
    if user_input:
        with st.spinner("🔄 Analyzing sentence..."):
            try:
                result = sentiment_analysis(user_input)
                sentiment = label_mapping[result[0]['label']]  # Map the label to a sentiment
                score = result[0]['score']

                # Display results in a card
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">Analysis Result:</div>
                    <p><strong>Sentiment:</strong> {sentiment}</p>
                    <p><strong>Confidence:</strong> {score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
