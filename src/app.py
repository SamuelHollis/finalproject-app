import streamlit as st
import pandas as pd
import time
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import logging
import os
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener el token de Hugging Face desde las variables de entorno
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Autenticación con Hugging Face
client = InferenceClient(token=HUGGINGFACE_TOKEN)

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Función para dividir el texto en chunks según el límite de tokens
def chunk_text(text, tokenizer, chunk_size=512):
    tokens = tokenizer(text, truncation=True, max_length=chunk_size, return_tensors='pt')
    input_ids = tokens.input_ids[0]
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]
        yield tokenizer.decode(chunk_ids, skip_special_tokens=True)

# Variables de backoff para evitar rate-limiting
initial_backoff = 5
max_backoff = 300
backoff_factor = 2

def backoff_sleep(intento):
    sleep_time = min(initial_backoff * (backoff_factor ** intento), max_backoff)
    logging.info(f"Rate limit hit. Sleeping for {sleep_time} seconds...")
    time.sleep(sleep_time)

# Función para analizar sentimientos y manejar el rate-limit
def analyze_sentiments_chunked(df, tokenizer, rate_limit_sleep, chunk_size=512):
    intento = 0
    sentiment_list = []
    score_list = []

    for idx, text in enumerate(df['text']):
        logging.info(f"Procesando comentario {idx + 1}/{len(df)}")
        chunks = list(chunk_text(text, tokenizer, chunk_size=chunk_size))

        overall_sentiment = None
        max_score = -1  # Inicializar para encontrar el máximo
        for chunk in chunks:
            while True:
                try:
                    response = client.text_classification(
                        model="cardiffnlp/twitter-roberta-base-sentiment",
                        text=chunk
                    )
                    break  # Si la respuesta es exitosa, salimos del bucle
                except Exception as e:
                    logging.error(f'Error al procesar el chunk: {e}')
                    if hasattr(e, 'response') and e.response.status_code == 429:
                        backoff_sleep(intento)
                        intento += 1
                    else:
                        # Si hay un error y no es un rate limit, saltamos este chunk
                        response = None
                        break

            # Solo procesar si response tiene un valor
            if response:
                for element in response:
                    if element['score'] > max_score:
                        max_score = element['score']
                        overall_sentiment = element['label']

        sentiment_list.append(overall_sentiment)
        score_list.append(max_score)
        time.sleep(rate_limit_sleep)  # Pausa para evitar rate-limit

    # Agregar los resultados al dataframe
    df['sentiment'] = sentiment_list
    df['score'] = score_list
    return df


# Configuración del estilo con CSS
page_bg_css = '''
<style>
body {
    background: #F5F5F5; /* Fondo más oscuro */
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #FFB3B3, #FFFFFF, #B3C6FF);
    background-size: cover;
    background-attachment: fixed; /* Hace que el fondo sea dinámico y se mueva con el scroll */
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
    border-radius: 8px; /* Bordes redondeados para el área de texto */
}

.stPlotlyChart div {
    border: 2px solid #003366; /* Azul oscuro para el borde del gráfico */
    border-radius: 15px; /* Bordes redondeados para el gráfico */
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
st.title("Análisis de Sentimiento de Comentarios en CSV o Texto")

# Subida de archivo CSV para análisis
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
                analyzed_df = analyze_sentiments_chunked(df, tokenizer, rate_limit_sleep=1.0)

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

# Análisis de una frase individual
st.subheader("Análisis de una frase individual")
user_input = st.text_area("Escribe una frase para analizar", "")

if st.button("Analizar frase"):
    if user_input:
        with st.spinner("Analizando la frase..."):
            # Crear un DataFrame temporal con una sola frase
            result_df = pd.DataFrame([{'text': user_input}])
            analyzed_single_df = analyze_sentiments_chunked(result_df, tokenizer, rate_limit_sleep=1.0)

            # Mostrar el resultado del análisis
            st.write(analyzed_single_df[['text', 'sentiment', 'score']].iloc[0])

