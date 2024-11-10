import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Carregue os dados do arquivo data.csv
df = pd.read_csv('data.csv')

# Carregue o modelo treinado
model = load_model('modelo_recomendacao.h5')

@app.route('/')
def index():
    genres = df['Genre'].unique()  # Obtenha os gêneros únicos
    return render_template('index.html', genres=genres)

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

    recommendations = top_movies[['Movie Name', 'Year of Release', 'Director', 'Genre', 'Description']].to_dict(orient='records')
    return render_template('recommendations.html', movies=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

    