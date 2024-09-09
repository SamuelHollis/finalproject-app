import streamlit as st
from model import analyze_sentiment
from utils import visualize_sentiment_distribution, process_comments

# CSS personalizado para el fondo con gradiente en tonos pastel
page_bg_img = '''
<style>
body {
    background: linear-gradient(to bottom, #FFB3B3, #FFFFFF, #B3C6FF);
    background-size: cover;
    color: black;
}

h1 {
    color: #D64545; /* Rojo pastel para el título */
    text-align: center;
    font-family: Arial, sans-serif;
}

.stButton>button {
    background-color: #7386D5; /* Azul pastel */
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
}

footer {
    visibility: hidden;
}
</style>
'''

# Inyectar el CSS personalizado en la aplicación
st.markdown(page_bg_img, unsafe_allow_html=True)

# Mapeo de las etiquetas numéricas a etiquetas de texto
label_mapping = {
    'LABEL_0': 'Negativo',
    'LABEL_1': 'Neutro',
    'LABEL_2': 'Positivo'
}

# Título de la aplicación
st.title("Análisis de Sentimiento de Comentarios sobre las Elecciones de USA")

# Entrada de texto por parte del usuario
user_input = st.text_area("Escribe un comentario para analizar", "")

# Botón para ejecutar el análisis
if st.button("Analizar"):
    if user_input:
        # Llama a la función que analiza el sentimiento
        result = analyze_sentiment(user_input)

        # Obtener la etiqueta y la confianza
        label = result[0]['label']
        score = result[0]['score']

        # Mapea la etiqueta numérica a un texto descriptivo (Negativo, Neutro, Positivo)
        sentiment_text = label_mapping[label]

        # Mostrar el resultado al usuario
        st.write(f"**Sentimiento:** {sentiment_text}")
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

