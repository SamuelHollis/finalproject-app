import streamlit as st
import pandas as pd
import logging
import time
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from matplotlib.patches import FancyBboxPatch

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo y tokenizador localmente
def load_local_model():
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Detectar si CUDA está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Intentar cargar el modelo y moverlo a la GPU si está disponible
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

        return model, tokenizer

    except ImportError as e:
        st.error(f"Error importing required backend: {e}")
        st.stop()

# Cargar el modelo local
model, tokenizer = load_local_model()

# Mapeo de etiquetas
label_mapping = ['Negative', 'Neutral', 'Positive']

# Preprocesar texto
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Función para obtener los scores de cada etiqueta (Negative, Neutral, Positive)
def get_sentiment_scores(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)  # Aplicar softmax para obtener las probabilidades
    return scores

# Función para analizar los sentimientos de un archivo CSV y actualizar la barra de progreso
def analyze_sentiments_csv(df):
    total_chunks = len(df)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    sentiments = []
    negative_scores = []
    neutral_scores = []
    positive_scores = []

    for idx, row in df.iterrows():
        text = row['text']
        # Obtener los scores de cada sentimiento
        try:
            scores = get_sentiment_scores(text)
            sentiments.append(label_mapping[np.argmax(scores)])  # El sentimiento con mayor puntuación
            negative_scores.append(scores[0])
            neutral_scores.append(scores[1])
            positive_scores.append(scores[2])
        except Exception as e:
            st.error(f"Error during sentiment analysis: {e}")
            sentiments.append("error")
            negative_scores.append(0)
            neutral_scores.append(0)
            positive_scores.append(0)

        # Actualizar barra de progreso
        progress_percentage = (idx + 1) / total_chunks
        progress_bar.progress(progress_percentage)
        progress_text.text(f"Processing {idx + 1} of {total_chunks}")

    df['sentiment'] = sentiments
    df['negative_score'] = negative_scores
    df['neutral_score'] = neutral_scores
    df['positive_score'] = positive_scores

    # Completar la barra de progreso
    progress_bar.progress(1.0)
    st.success("Sentiment analysis complete!")

    # Convertir el DataFrame en CSV y permitir la descarga
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv,
        file_name='sentiment_analysis_results.csv',
        mime='text/csv',
    )

    return df

# Función para calcular los porcentajes de cada sentimiento
def calculate_sentiment_percentages(df):
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    return [sentiment_counts.get('Negative', 0), sentiment_counts.get('Neutral', 0), sentiment_counts.get('Positive', 0)]

# CSS para mejorar el aspecto
page_bg_css = '''
<style>
body {
    background: url("https://www.omfif.org/wp-content/uploads/2024/01/GettyImages-1183053829.jpg");
    background-size: cover;
    background-position: cover;
    background-repeat: no-repeat;
    font-family: 'Helvetica Neue', sans-serif;
    opacity: 0.7;
}
[data-testid="stAppViewContainer"] {
    background: rgba(0, 0, 0, 0.7);
    background-blend-mode: overlay;
    padding: 2rem;
    color: white;
}
h1 {
    color: #000080;
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
    opacity: 1.2;
    background-color: rgba(150, 150, 150, 0.9);
    padding: 4px;
    border-radius: 10px;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}
h2, h3 {
    color: white;
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
}
.stButton>button {
    background-color: #1E90FF;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.stButton>button:hover {
    background-color: #1E90FF;
    transform: scale(1.05);
}
.stTextArea textarea {
    background-color: rgba(40, 40, 40, 0.9);
    border-radius: 12px;
    font-size: 16px;
    padding: 15px;
    color: white;
}
footer {
    visibility: hidden;
}
.result-card {
    background-color: rgba(40, 40, 40, 0.9);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    color: white;
}
.card-header {
    font-size: 24px;
    font-weight: bold;
    color: #1E90FF;
    margin-bottom: 15px;
}
</style>
'''

# Inyectar CSS
st.markdown(page_bg_css, unsafe_allow_html=True)

# Título de la aplicación
st.title("SENTIMENT ANALYSIS")

# Section 1: Individual Sentence Analysis
st.subheader("📝 Analyze a Single Sentence")

# Campo para que el usuario ingrese una oración
user_input = st.text_area("Write a sentence to analyze", "", key="single_sentence_input")

if st.button("📊 Analyze Sentence", key="analyze_sentence_button"):
    if user_input:  # Si el usuario ha ingresado texto
        with st.spinner("🔄 Analyzing sentence..."):
            try:
                # Obtener los scores completos de cada etiqueta
                scores = get_sentiment_scores(user_input)
                
                # Crear DataFrame con los scores
                sentiment_df = pd.DataFrame({
                    'Sentiment': label_mapping,
                    'Probability': [score * 100 for score in scores]  # Convertir a porcentaje
                })

                # Mostrar el resultado del análisis principal
                sentiment = label_mapping[np.argmax(scores)]
                confidence = max(scores) * 100

                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">Analysis Result:</div>
                    <p><strong>Sentiment:</strong> {sentiment}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Configurar tema de Seaborn
                sns.set_theme(style="whitegrid", font_scale=1.2)
                
                # Crear una paleta personalizada
                colors = sns.color_palette("icefire")

                # Crear el gráfico con barras horizontales
                fig, ax = plt.subplots(figsize=(7, 4))

                # Cambiar la opacidad de las barras y usar una paleta de colores
                sns.barplot(x="Probability", y="Sentiment", data=sentiment_df, palette=colors, ax=ax, alpha=0.8)  # alpha controla la opacidad

                # Añadir los valores sobre las barras
                for index, value in enumerate(sentiment_df['Probability']):
                    ax.text(value + 1, index, f'{value:.2f}%', va='center', fontweight='bold', fontsize=11)

                # Estilo del gráfico
                ax.set_title("Sentiment Probabilities", fontsize=16, fontweight='bold', color="#333")
                ax.set_xlim(0, 100)  # Limitar el eje de las probabilidades a 100%
                ax.set_xlabel("Probability (%)", fontsize=12, fontweight='bold')
                ax.set_ylabel("Sentiment", fontsize=12, fontweight='bold')

                # Añadir un borde redondeado al gráfico
                bbox = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05", linewidth=2, edgecolor="black", facecolor='none', transform=ax.transAxes)
                ax.add_patch(bbox)

                # Añadir un borde suave al gráfico y mejorar su presentación
                sns.despine(left=True, bottom=True)
                plt.tight_layout()

                # Mostrar el gráfico en Streamlit
                st.pyplot(fig)


            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sección 2: Análisis de archivo CSV
st.subheader("📂 Analyze CSV File")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    # Cargar el archivo CSV sin incluir el índice como columna
    df = pd.read_csv(uploaded_file, index_col=None)

    # Mostrar las primeras filas del CSV
    st.write("First 5 comments from the file:")
    st.write(df.head())

    # Botón para ejecutar el análisis de sentimientos en el CSV
    if st.button("🔍 Analyze Sentiments in CSV"):
        if 'text' not in df.columns:
            st.error("The CSV file must contain a 'text' column.")
        else:
            with st.spinner("🔄 Analyzing sentiments, please wait..."):
                analyzed_df = analyze_sentiments_csv(df)

            st.success("✅ Analysis complete!")

            # Mostrar resultados
            st.write("Analysis Results:")
            st.write(analyzed_df.head())

            # Calcular y mostrar porcentajes de sentimientos
            percentages = calculate_sentiment_percentages(analyzed_df)
            labels = ['Negative', 'Neutral', 'Positive']
            colors = ['#FF6B6B', '#F7D794', '#4CAF50']

            # Crear gráfico de barras
            fig, ax = plt.subplots()
            ax.barh(labels, percentages, color=colors)
            ax.set_xlabel('Percentage (%)')
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)