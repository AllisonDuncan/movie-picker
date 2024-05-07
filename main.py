import re
import string
import pandas as pd
import nltk
import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# download nltk resources for tokenization, lemmatization, and stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# define clean_text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    
    # remove punctuation while retaining words and digits
    text = re.sub(r'[^\\w\\s\\d]', '', text)

    # tokenize the text into words
    words = word_tokenize(text)
    # define english stopwords
    stop_words = set(stopwords.words('english'))
    # remove stopwords
    words = [word for word in words if word not in stop_words]
    # initialize the wordnet lemmatizer
    lemmatizer = WordNetLemmatizer()
    # lemmatize each word
    words = [lemmatizer.lemmatize(word) for word in words]
    # join the words back into a string
    text = ' '.join(words)
    return text

movies = pd.read_csv("../data/tmdb_5000_movies.csv")
print(movies.columns)
# focus columns: id, overview, title, genres
movies = movies[['id', 'overview', 'title', 'genres']]

# genres column is a string that's actually a list of genres in json format [{'id': 18, 'name': 'Drama'}, {'id': 80, 'name': 'Crime'}]
# convert it to a list of genres
# The line below is applying a lambda function to each row in the 'genres' column. The lambda function takes in a row value (which is a string representing a JSON list of dictionaries, where each dictionary has 'id' and 'name' keys), evals it to convert it to a list of dictionaries, and then extracts the 'name' values from each dictionary in the list. This operation is done row-wise for each row in the 'genres' column.
# 
# Here's a breakdown of the syntax:
# - `movies['genres']` selects the 'genres' column from the movies DataFrame
# - `.apply(lambda x: [i['name'] for i in eval(x)])` applies a lambda function to each row in the 'genres' column. 
#   - `lambda x:` is a lambda function that takes in a single argument `x`.
#   - `eval(x)` takes the string value in `x` and interprets it as Python code, returning a list of dictionaries.
#   - `[i['name'] for i in ...]` is a list comprehension that extracts the 'name' values from each dictionary in the list returned by `eval(x)`.
#   - The result of this lambda function is a list of genre names for each row in the 'genres' column, resulting in a new column called 'genre_list'.
genre_list = movies['genres'].apply(lambda x: [i['name'] for i in eval(x)])

# add a new column 'tags by concatenating 'overview' and 'genres' - only include the name for each genre
movies['tags'] = movies['overview'] + ' ' + genre_list.apply(lambda x: ' '.join(x))

# remove the original 'overview' and 'genres' columns
new_data = movies.drop(columns=['overview', 'genres'])

# 'clean_text' function to clean the text data
new_data['tags_cleaned'] = new_data['tags'].apply(clean_text)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectorized_data = cv.fit_transform(new_data['tags_cleaned']).toarray()

similarity = cosine_similarity(vectorized_data)

#calculate similarity scores for the third movie with all other movies, sort them, and store the result
distance = sorted(list(enumerate(similarity[4])), reverse=True, key=lambda x: x[1])

for i in distance[0:5]:
    print(new_data.iloc[i[0]].title)


# Define a function to recommend the top 5 similar movies for a given movie
def recommend(movies):
    index = movies_list[movies_list['title'] == movies].index[0]
    distances = sorted(list(enumerate(similarity_loaded[index])), reverse=True, key=lambda vector: vector[1])
    for i in distances[1:6]:
        print(movies_list.iloc[i[0]].title + " " + movies_list.iloc[i[0]].tags)


pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

movies_list = pickle.load(open('movies_list.pkl', 'rb'))
similarity_loaded = pickle.load(open('similarity.pkl', 'rb'))

print(os.getcwd())

recommend('Shrek the Third')
