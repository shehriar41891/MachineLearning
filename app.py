from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
app.secret_key = 'supersecretkey'

ps = PorterStemmer()
Stopwords = set(stopwords.words('english'))

# Load your data
data = pd.read_csv(r'C:\Users\Hp\OneDrive\Documents\asim\softwarehouse2.csv')

# Function to preprocess text
def transformtext(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    filtered_text = []
    refined_text = []

    for word in text:
        if word.isalnum():
            filtered_text.append(word)

    for word in filtered_text:
        if word not in Stopwords and word not in string.punctuation:
            refined_text.append(word)

    for i in range(len(refined_text)):
        refined_text[i] = ps.stem(refined_text[i])

    return " ".join(refined_text)

# Function to recommend software houses
def fit(user_info):
    recommended_software_info = []
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['About']).toarray()
    transformed_user_info = transformtext(user_info)
    user_vector = cv.transform([transformed_user_info]).toarray()

    if user_vector.shape[1] != vectors.shape[1]:
        cv_adjusted = CountVectorizer(max_features=vectors.shape[1], stop_words='english')
        vectors_adjusted = cv_adjusted.fit_transform(data['About']).toarray()
        user_vector = cv_adjusted.transform([transformed_user_info]).toarray()

    similarities = cosine_similarity(user_vector, vectors_adjusted if 'vectors_adjusted' in locals() else vectors)
    top_indices = np.argsort(similarities[0])[-10:][::-1]

    for index in top_indices:
        software_house = data.loc[index, 'Names']
        about = data.loc[index, 'About']
        recommended_software_info.append({'Software House': software_house, 'About': about})

    return recommended_software_info

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        print('User Input:', user_input)  # Debug statement
        recommendations = fit(user_input)
        print('Recommendations:', recommendations)  # Debug statement
        # Pass recommendations directly to the template
        return render_template('index.html', recommendations=recommendations)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
