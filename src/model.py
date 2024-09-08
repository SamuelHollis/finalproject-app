from transformers import pipeline

@st.cache_resource
def load_model():
    # Carga el modelo preentrenado de Hugging Face
    return pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(comment):
    # Ejecuta el análisis de sentimiento sobre el comentario
    model = load_model()
    return model(comment)