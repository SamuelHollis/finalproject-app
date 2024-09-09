from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    # Carga el modelo preentrenado de Hugging Face
    return pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(comment):
    # Ejecuta el análisis de sentimiento sobre el comentario
    model = load_model()
    return model(comment)

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

        # Extraer y mostrar las probabilidades de cada etiqueta
        for result in results:
            label = result['label']  # Etiqueta devuelta por el modelo ('LABEL_0', 'LABEL_1', etc.)
            score = result['score']  # Probabilidad asociada a esa etiqueta
            sentiment_text = label_mapping[label]  # Convertir la etiqueta en texto descriptivo
            st.write(f"**{sentiment_text}:** {score * 100:.2f}%")

        # Determinar el sentimiento con mayor probabilidad
        most_probable_sentiment = max(results, key=lambda x: x['score'])
        sentiment_text = label_mapping[most_probable_sentiment['label']]
        st.write(f"**Sentimiento predominante:** {sentiment_text} ({most_probable_sentiment['score'] * 100:.2f}%)")
