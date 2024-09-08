import streamlit as st
from model import analyze_sentiment
from utils import visualize_sentiment_distribution, process_comments

# Título de la aplicación
st.title("Análisis de Sentimiento de Comentarios sobre las Elecciones de USA")

# Entrada de texto por parte del usuario
user_input = st.text_area("Escribe un comentario para analizar", "")

# Botón para ejecutar el análisis
if st.button("Analizar"):
    if user_input:
        # Analiza el comentario ingresado
        result = analyze_sentiment(user_input)
        sentiment = result[0]['label']
        score = result[0]['score']

        # Muestra el resultado
        st.write(f"**Sentimiento:** {sentiment}")
        st.write(f"**Confianza:** {score:.2f}")
        
        # Visualización del sentimiento
        st.bar_chart([score, 1 - score])

# Análisis de múltiples comentarios
st.subheader("Análisis de múltiples comentarios")
uploaded_file = st.file_uploader("Sube un archivo CSV con comentarios", type=["csv"])

if uploaded_file:
    # Procesar archivo CSV
    comments_df = pd.read_csv(uploaded_file)
    comments = comments_df['comment'].tolist()
    cleaned_comments = process_comments(comments)

    # Analizar cada comentario
    results = [analyze_sentiment(comment) for comment in cleaned_comments]

    # Visualización de resultados agregados
    visualize_sentiment_distribution(results)

