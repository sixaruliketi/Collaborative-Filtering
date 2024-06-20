import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("toy_dataset.csv", index_col=0)
ratings = ratings.fillna(0) #shevcvale NaN 0it
# print(ratings)

#this method will take each row of our data frame as inputs and then convert it to such that the new ratings will be minus the mean of
#all ratings and then divided by the range of ratings that the user gives
#so - we're trying to bring the mean of all the ratings that a user gives to zero and
#then we're dividing it by the range of the ratings that the user gives
#so this will correct for any users that are too harsh or they are too lenient so
def standardize(row) :
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row

ratings_std = ratings.apply(standardize)
# print(ratings_std)


#now we'll try item to item similarity (chemit vizan mere user to user similaritys)
#userToUser -istvis
#item_similarity = cosine_similarity(ratings_std) Transpose-is gareshe
#we are taking a transpose since we want similarity between items which need to be in rows
item_similarity = cosine_similarity(ratings_std.T)
# print(item_similarity) #this is similarity matrix - model based onn which we will give recommendations to the new users

#now we'll create data frame
#now it's in the form of array we're gonna convert it to data frame
item_similarity_df = pd.DataFrame(item_similarity, index=ratings.columns, columns = ratings.columns)
# print(item_similarity_df)

#let's make recommendations
def get_similar_movies(movie_name, user_ratings):
    similar_score = item_similarity_df[movie_name]*user_ratings
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

# print(get_similar_movies("action1", 5))

action_lover = [("action1", 5), ("romantic2",1), ("romantic3",1)]
similar_movies = pd.DataFrame()

for movie,rating in action_lover:
    similar_movies = similar_movies._append(get_similar_movies(movie,rating), ignore_index=True)

similar_movies.head()
similar_movies.sum().sort_values(ascending=False)

print(similar_movies.sum())

