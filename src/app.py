import streamlit as st
import pandas as pd
import logging
import time
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo y tokenizador localmente
@st.cache_resource
def load_local_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Detectar si CUDA está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar el modelo y moverlo a la GPU si está disponible
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # Configurar el pipeline para usar GPU
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return sentiment_analysis, tokenizer

# Cargar el modelo local
sentiment_analysis, tokenizer = load_local_model()

# Mapeo de etiquetas
label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}

# Función para analizar los sentimientos de un archivo CSV y actualizar la barra de progreso
def analyze_sentiments_csv(df, chunk_size=512, process_chunk_size=5000):
    total_chunks = len(df)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    sentiments = []
    scores = []

    for idx, row in df.iterrows():
        text = row['text']
        # Análisis de sentimiento con el pipeline cargado
        try:
            response = sentiment_analysis(text)
            sentiment = label_mapping[response[0]['label']]
            score = response[0]['score']
        except Exception as e:
            st.error(f"Error during sentiment analysis: {e}")
            sentiment = "error"
            score = 0.0

        sentiments.append(sentiment)
        scores.append(score)

        # Actualizar barra de progreso
        progress_percentage = (idx + 1) / total_chunks
        progress_bar.progress(progress_percentage)
        progress_text.text(f"Processing {idx + 1} of {total_chunks}")

    df['sentiment'] = sentiments
    df['score'] = scores

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

import streamlit as st

# Definimos el CSS personalizado para el fondo y los estilos de los componentes
page_bg_css = '''
<style>
body {
    background: url("https://www.omfif.org/wp-content/uploads/2024/01/GettyImages-1183053829.jpg");
    background-size: cover;
    background-position: center;
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
    color: #B22222;
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
    background-color: rgba(255, 255, 255, 0.5);
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
    background-color: rgba(0, 0, 0, 0.7);  /* Fondo negro con transparencia */
    border-radius: 12px;
    font-size: 16px;
    padding: 15px;
    color: white;  /* Texto blanco */
}
.stTextInput input {
    background-color: rgba(0, 0, 0, 0.7);  /* Fondo negro con transparencia para inputs de texto */
    border-radius: 12px;
    font-size: 16px;
    padding: 10px;
    color: white;  /* Texto blanco */
}
footer {
    visibility: hidden;
}
.result-card {
    background-color: rgba(0, 0, 0, 0.7);  /* Fondo negro con transparencia */
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
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

# Aplicamos el CSS personalizado
st.markdown(page_bg_css, unsafe_allow_html=True)

# Ejemplo de contenido en la app
st.title("Título de Ejemplo")
st.header("Subtítulo de Ejemplo")
st.text_area("Ingrese su texto aquí")

# Botón de ejemplo
if st.button("Haga clic aquí"):
    st.write("¡Botón presionado!")

# Tarjeta de ejemplo con estilo personalizado
st.markdown('<div class="result-card"><div class="card-header">Resultado</div>Contenido de ejemplo en una tarjeta estilizada.</div>', unsafe_allow_html=True)


# Inyectar CSS
st.markdown(page_bg_css, unsafe_allow_html=True)

# Título de la aplicación
st.title("Sentiment Analysis")

# Sección 1: Análisis de archivo CSV
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

# Section 2: Individual Sentence Analysis
st.subheader("📝 Analyze a Single Sentence")

# Campo para que el usuario ingrese una oración
user_input = st.text_area("Write a sentence to analyze", "", key="single_sentence_input")

if st.button("📊 Analyze Sentence", key="analyze_sentence_button"):
    if user_input:  # Si el usuario ha ingresado texto
        with st.spinner("🔄 Analyzing sentence..."):
            try:
                # Obtener los resultados completos de cada etiqueta
                result = sentiment_analysis(user_input)

                # Crear listas para las etiquetas y las puntuaciones
                labels = [label_mapping[res['label']] for res in result]
                scores = [res['score'] for res in result]

                # Crear un DataFrame con las etiquetas y sus probabilidades
                sentiment_df = pd.DataFrame({
                    'Sentiment': labels,
                    'Probability': [score * 100 for score in scores]  # Convertir a porcentaje
                })

                # Mostrar el resultado del análisis principal
                max_index = scores.index(max(scores))
                sentiment = labels[max_index]
                confidence = scores[max_index]

                st.markdown(f"""
                <div class="result-card">
                    <div class="card-header">Analysis Result:</div>
                    <p><strong>Sentiment:</strong> {sentiment}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Graficar con Seaborn
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="Probability", y="Sentiment", data=sentiment_df, palette="coolwarm", ax=ax)

                # Añadir los valores sobre las barras
                for index, value in enumerate(sentiment_df['Probability']):
                    ax.text(value + 1, index, f'{value:.2f}%', va='center')

                # Estilo del gráfico
                ax.set_title("Sentiment Probabilities", fontsize=16, fontweight='bold')
                ax.set_xlim(0, 100)  # Limitar el eje de las probabilidades a 100%
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")

