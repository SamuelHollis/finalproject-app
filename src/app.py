import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import torch
import seaborn as sns
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Descargar y cargar el modelo y tokenizador localmente
@st.cache_resource
def load_local_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Detectar si CUDA está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar el modelo y moverlo a la GPU si está disponible
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # Configurar el pipeline para usar GPU (device=0 para GPU)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1), tokenizer

# Cargar el modelo local
sentiment_analysis, tokenizer = load_local_model()

# Mapeo de etiquetas de RoBERTa a sentimientos comprensibles
label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}

# Función para dividir texto en fragmentos respetando el límite de tokens
def chunk_text(text, tokenizer, chunk_size=512):
    tokens = tokenizer(text, truncation=True, max_length=chunk_size, return_tensors='pt')
    input_ids = tokens.input_ids[0]
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]
        yield tokenizer.decode(chunk_ids, skip_special_tokens=True)

# Función para analizar en chunks usando el modelo local y permitir la descarga de resultados como CSV
def analyze_sentiments_chunked(df, tokenizer, chunk_size=512, process_chunk_size=5000):
    ch_num = 0

    # Inicializar la barra de progreso y el mensaje de estado
    total_chunks = len(df) // process_chunk_size + (1 if len(df) % process_chunk_size > 0 else 0)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Analyzing...")  # Texto que muestra el estado de análisis

    # Procesar el dataframe en chunks de `process_chunk_size`
    for start in range(0, len(df), process_chunk_size):
        ch_num += 1
        end = min(start + process_chunk_size, len(df))
        chunk_df = df.iloc[start:end]
        sentiment_list = []
        score_list = []

        for idx, text in enumerate(chunk_df['text']):
            # Dividir en chunks
            chunks = list(chunk_text(text, tokenizer, chunk_size=chunk_size))

            # Análisis de sentimiento por chunks usando el pipeline local
            overall_sentiment = None
            max_score = -1  # Inicializar para que cualquier puntuación sea más alta
            for chunk in chunks:
                try:
                    # Usar el pipeline `sentiment_analysis` local
                    response = sentiment_analysis(chunk)

                    # Encontrar la etiqueta con la puntuación más alta
                    for element in response:
                        if element['score'] > max_score:
                            max_score = element['score']
                            overall_sentiment = element['label']

                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    st.error(f"Unexpected error: {e}")
                    continue

            sentiment_list.append(overall_sentiment)
            score_list.append(max_score)

        # Asignar los resultados al chunk procesado
        df.loc[start:end-1, 'sentiment'] = sentiment_list
        df.loc[start:end-1, 'score'] = score_list

        # Actualizar barra de progreso
        progress_percentage = (ch_num / total_chunks)
        progress_bar.progress(progress_percentage)

    # Completar la barra de progreso
    progress_bar.progress(1.0)
    progress_text.text("Analysis Complete!")
    st.success("Sentiment analysis complete!")

    # Convertir el DataFrame en CSV
    csv = df.to_csv(index=False).encode('utf-8')

    # Añadir botón para descargar el archivo CSV
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv,
        file_name='sentiment_analysis_results.csv',
        mime='text/csv',
    )

# Función para calcular y mostrar los porcentajes de sentimiento
def calculate_sentiment_percentages(df):
    # Contar la frecuencia de cada sentimiento
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    sentiments = ['LABEL_0', 'LABEL_1', 'LABEL_2']  # LABEL_0: Negative, LABEL_1: Neutral, LABEL_2: Positive
    
    # Crear una lista con los porcentajes de cada sentimiento
    percentages = [sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
    return percentages

# Inyección del CSS en la aplicación
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
    opacity: 1;
    background-color: rgba(255, 255, 255, 0.5); /* Semi-transparent white background */
    padding: 4px;
    border-radius: 10px; 
    max-width: 500px; /* Limit the width */
    margin-left: auto; /* Center the element */
    margin-right: auto; /* Center the element */
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
    background-color: rgba(107, 107, 107, 0.8); /* Más opaca */
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

# Injectar el CSS en la aplicación
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
                # Llamar a la función con todos los parámetros requeridos
                analyzed_df = analyze_sentiments_chunked(df, tokenizer, chunk_size=512, process_chunk_size=5000)

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

