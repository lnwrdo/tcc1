import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import requests  # Import necessário para fazer chamadas HTTP

app = Flask(__name__)

# Carregue os dados do arquivo data.csv
df = pd.read_csv('data.csv')

# Limpe as descrições removendo vírgulas desnecessárias entre palavras
df['Description'] = df['Description'].apply(lambda x: ' '.join(x.split(', ')))

# Carregue o modelo treinado
model = load_model('modelo_recomendacao.h5')

# Chave de API do OMDb - Substitua por sua chave
OMDB_API_KEY = "60ac2f57"

def fetch_movie_poster(title):
    """Busca o pôster de um filme no OMDb."""
    url = f"https://www.omdbapi.com/apikey.aspx?VERIFYKEY=d55e83b3-4f6b-4633-8e0a-b99035aab968"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == 'True':
            return data.get('Poster', None)  # Retorna a URL do pôster se disponível
    return None

@app.route('/')
def index():
    genres = df['Genre'].unique()  # Obtenha os gêneros únicos
    return render_template('index.html', genres=genres)

@app.route('/escolha')
def escolha_filme():
    genres = df['Genre'].unique()  # Obtenha os gêneros únicos
    return render_template('escolha.html', genres=genres)

@app.route('/sobre')
def sobre_o_projeto():
    genres = df['Genre'].unique()  # Obtenha os gêneros únicos
    return render_template('sobre.html', genres=genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_genre = request.form['genre']
    # Filtre os filmes pelo gênero selecionado
    filtered_movies = df[df['Genre'] == selected_genre]

    # Prepare os dados para a previsão
    X_predict = filtered_movies[['Movie Rating', 'Votes']].values
    predicted_ratings = model.predict(X_predict)

    # Adicione as previsões ao DataFrame
    filtered_movies['Predicted Rating'] = predicted_ratings

    # Obtenha os 5 filmes com as melhores previsões
    top_movies = filtered_movies.nlargest(5, 'Predicted Rating')

    # Limpe as descrições antes de criar as recomendações
    top_movies['Description'] = top_movies['Description'].apply(lambda x: ' '.join(x.split(', ')))

    # Prepare as recomendações, incluindo o pôster
    recommendations = []
    for _, row in top_movies.iterrows():
        movie_info = {
            'Movie Name': row['Movie Name'],
            'Year of Release': row['Year of Release'],
            'Director': row['Director'],
            'Genre': row['Genre'],
            'Description': row['Description'],
            'Poster': fetch_movie_poster(row['Movie Name'])  # Chama a função para obter o pôster
        }
        recommendations.append(movie_info)

    return render_template('recommendations.html', movies=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
