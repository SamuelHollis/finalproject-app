import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sentiment_distribution(results):
    # Toma los resultados de los análisis y crea una gráfica
    sentiments = [r['label'] for r in results]
    df = pd.DataFrame(sentiments, columns=['Sentiment'])
    
    sns.countplot(x='Sentiment', data=df)
    plt.title("Distribución de Sentimientos")
    plt.show()

def process_comments(comments):
    # Proceso básico de limpieza de comentarios (puedes expandirlo)
    cleaned_comments = [comment.strip().lower() for comment in comments]
    return cleaned_comments
