import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

# Cargar el modelo de análisis de sentimiento usando RoBERTa (Twitter)
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

# Función para analizar el sentimiento y devolver los resultados
def analyze_sentiment(text):
    model = load_model()
    return model(text)

# CSS personalizado
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
st.title("Análisis de Sentimiento de Comentarios sobre las Elecciones de USA")

# Entrada de texto por parte del usuario
user_input = st.text_area("Escribe un comentario para analizar", "")

# Botón para ejecutar el análisis
if st.button("Analizar"):
    if user_input:
        # Llama a la función que analiza el sentimiento
        results = analyze_sentiment(user_input)

        # Mapeo de las etiquetas devueltas por RoBERTa
        label_mapping = {
            'LABEL_0': 'Negativo',
            'LABEL_1': 'Neutro',
            'LABEL_2': 'Positivo'
        }

        # Mostrar las probabilidades de cada etiqueta
        for result in results:
            label = result['label']  # Etiqueta devuelta por el modelo ('LABEL_0', 'LABEL_1', etc.)
            score = result['score']  # Probabilidad asociada a esa etiqueta
            sentiment_text = label_mapping[label]  # Convertir la etiqueta en texto descriptivo
            st.write(f"**{sentiment_text}:** {score * 100:.2f}%")

        # Determinar el sentimiento con mayor probabilidad
        most_probable_sentiment = max(results, key=lambda x: x['score'])
        sentiment_text = label_mapping[most_probable_sentiment['label']]
        st.write(f"**Sentimiento predominante:** {sentiment_text} ({most_probable_sentiment['score'] * 100:.2f}%)")

        # Crear un gráfico de barras para visualizar los resultados
        labels = ['Negativo', 'Neutro', 'Positivo']
        scores = [result['score'] * 100 for result in results]

        fig, ax = plt.subplots()
        ax.bar(labels, scores, color=['#FFB3B3', '#D3D3D3', '#B3C6FF'])
        ax.set_ylabel('Probabilidad (%)')
        ax.set_title('Distribución de Sentimientos')
        
        # Mostrar el gráfico con el estilo de bordes redondeados
        st.pyplot(fig)
