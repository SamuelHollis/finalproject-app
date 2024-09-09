import streamlit as st
import pandas as pd
import time
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Descargar y cargar el modelo y el tokenizador localmente
@st.cache_resource  # Cachear el modelo para evitar recargarlo cada vez
def load_local_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Cargar el modelo localmente
sentiment_analysis = load_local_model()

# Función para analizar sentimientos en un DataFrame (CSV)
def analyze_sentiments_chunked(df, rate_limit_sleep=1.0):
    sentiment_list = []
    score_list = []

    for idx, text in enumerate(df['text']):
        logging.info(f"Procesando comentario {idx + 1}/{len(df)}")
        try:
            # Ejecutar el análisis de sentimientos
            result = sentiment_analysis(text)
            # Obtener el sentimiento y el score
            sentiment = result[0]['label']
            score = result[0]['score']
        except Exception as e:
            logging.error(f"Error al procesar el texto: {e}")
            sentiment = "error"
            score = 0.0

        sentiment_list.append(sentiment)
        score_list.append(score)
        time.sleep(rate_limit_sleep)  # Pausa entre cada análisis

    # Agregar los resultados al DataFrame
    df['sentiment'] = sentiment_list
    df['score'] = score_list
    return df

# CSS personalizado para el estilo
page_bg_css = '''
<style>
body {
    background: #F5F5F5; /* Fondo más oscuro */
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #FFB3B3, #FFFFFF, #B3C6FF);
    background-size: cover;
    background-attachment: fixed; /* Hace que el fondo sea dinámico */
}

h1 {
    color: #003366; /* Azul oscuro para el título */
    text-align: center;
    font-family: Arial, sans-serif;
}

p, div, span {
    color: #000000; /* Negro para el resto de los textos */
}

.stButton>button {
    background-color: #003366; /* Azul oscuro para el botón */
    color: white;
    border-radius: 5px;
    padding: 10px;
    border: 2px solid white;
    font-size: 16px;
}

.stTextArea textarea {
    background-color: #f5f5f5; /* Fondo claro */
    color: #000000; /* Texto negro */
    font-size: 16px;
    padding: 10px;
    border-radius: 8px; /* Bordes redondeados */
}

.stPlotlyChart div {
    border: 2px solid #003366; /* Borde azul oscuro para el gráfico */
    border-radius: 15px; /* Bordes redondeados */
    padding: 10px;
}

footer {
    visibility: hidden;
}
</style>
'''

# Inyectar el CSS personalizado en la aplicación
st.markdown(page_bg_css, unsafe_allow_html=True)

# Título de la aplicación
st.title("Análisis de Sentimiento Local de Comentarios")

# Sección 1: Análisis de archivo CSV
st.subheader("Análisis de archivo CSV")
uploaded_file = st.file_uploader("Sube un archivo CSV con una columna llamada 'text'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Mostrar los primeros registros del CSV
    st.write("Primeros 5 comentarios del archivo:")
    st.write(df.head())

    # Botón para ejecutar el análisis de sentimientos
    if st.button("Analizar Sentimientos CSV"):
        if 'text' not in df.columns:
            st.error("El archivo CSV debe contener una columna llamada 'text'.")
        else:
            with st.spinner("Analizando los sentimientos, por favor espera..."):
                analyzed_df = analyze_sentiments_chunked(df)

            st.success("Análisis completado!")

            # Mostrar los resultados
            st.write("Resultados del análisis:")
            st.write(analyzed_df.head())

            # Descargar los resultados como CSV
            csv = analyzed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar resultados como CSV",
                data=csv,
                file_name='resultados_sentimientos.csv',
                mime='text/csv',
            )

# Sección 2: Análisis de una frase individual
st.subheader("Análisis de una frase individual")
user_input = st.text_area("Escribe una frase para analizar", "")

if st.button("Analizar frase"):
    if user_input:
        with st.spinner("Analizando la frase..."):
            result = sentiment_analysis(user_input)
            sentiment = result[0]['label']
            score = result[0]['score']
            st.write(f"**Sentimiento:** {sentiment}")
            st.write(f"**Confianza:** {score:.2f}")



