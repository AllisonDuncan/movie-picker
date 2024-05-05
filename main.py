import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# define clean_text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
movies = pd.read_csv("../data/tmdb_5000_movies.csv")
print(movies.columns)
# focus columns: id, overview, title, genres
movies = movies[['id', 'overview', 'title', 'genres']]

# add a new column 'tags by concatenating 'overview' and 'genres'
movies['tags'] = movies['overview'] + ' ' + movies['genres']

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
    index = new_data[new_data['title'] == movies].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    for i in distances[1:6]:
        print(new_data.iloc[i[0]].title + " " + new_data.iloc[i[0]].tags)

recommend('Tangled')
