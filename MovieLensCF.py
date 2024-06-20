import pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# ratings = pd.merge(movies,ratings)
# ratings.head()
#now columns have ID of the movie, the title, genres, ID of the users that rated these movies, ratings, and timestamp of when this was created
#for collaborative filtering we're only interested in userID and ratings so we're gonna drop a few columns

ratings = pd.merge(movies,ratings).drop(['genres', 'timestamp'], axis=1)
ratings.head()


user_ratings = ratings.pivot_table(index=['userId'],columns=['title'], values='rating')
user_ratings.head()

#let's remove movies which have less than 10 users who rated it and fill remaining NAn with 0
user_ratings = user_ratings.dropna(thresh=10,axis=1).fillna(0)

#now let's build similarity Matrix
item_similarity_df = user_ratings.corr(method='pearson')
item_similarity_df.head(50)

def get_similar_movies(movie_name, user_ratings):
    similar_score = item_similarity_df[movie_name]*user_ratings
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

action_lover = [("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)", 5),
                ("12 Years a Slave (2013)", 4), ("2012 (2009)", 3),
                ("(500) Days of Summer (2009)", 2)]
similar_movies = pd.DataFrame()

for movie,rating in action_lover:
    similar_movies = similar_movies._append(get_similar_movies(movie,rating), ignore_index=True)

similar_movies.head()
similar_movies.sum().sort_values(ascending=False)
