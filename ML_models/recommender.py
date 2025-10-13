import pandas as pd
import sqlite3
from lightfm.data import Dataset
from lightfm import LightFM
# load data
movies = pd.read_csv('./../data/movies.csv')
ratings = pd.read_csv('./../data/ratings.csv')
tags = pd.read_csv('./../data/tags.csv')

# create SQLite in-memory db
conn = sqlite3.connect(':memory:')

# write dataframes to SQL tables
movies.to_sql('movies', conn, index=False, if_exists='replace')
ratings.to_sql('ratings',conn,index=False,if_exists='replace')
tags.to_sql('tags',conn,index=False,if_exists='replace')

# inner join ratings,tags and movies CSVs via movieId and userId
query = """
        SELECT  CAST(strftime('%Y', datetime(r.timestamp, 'unixepoch')) AS INTEGER) AS year,
                r.userId,
                r.movieId,
                m.title,
                m.genres,
                r.rating,
                t.tag
        FROM ratings r
        INNER JOIN movies m 
            ON r.movieId = m.movieId
        LEFT JOIN tags t 
            ON r.movieId = t.movieId AND r.userId = t.userId
"""

joined_df = pd.read_sql_query(query, conn)
# print(joined_df['rating'].max())
# print(joined_df.head(20))


'''
BUILD INTERACTION MATRIX
LightFM’s Dataset class to encode IDs into integer indices and construct the matrices:
'''
# Create dataset object
dataset = Dataset()

'''
LightFM internally creates integer mappings for all user and item IDs — it doesn’t care if they were originally integers or strings.
But if you pass them as strings during dataset.fit(), LightFM stores them as strings internally ('1', '2', '3', etc.).
Meanwhile SQLite database stores them as integers as read from the datasets
Therefore for consistency best set to integers so that LightFM’s internal IDs properly map back to the real movieIds in your SQL database.
'''
ratings['userId'] = ratings['userId'].astype(int)
movies['movieId'] = movies['movieId'].astype(int)
joined_df['userId'] = joined_df['userId'].astype(int)
joined_df['movieId'] = joined_df['movieId'].astype(int)


# Fit users, items, and item features (genres)
dataset.fit(
    users=ratings['userId'].unique().tolist(),
    items=movies['movieId'].unique().tolist(),
    item_features=set(
        g for genre_list in joined_df['genres'].dropna()
        for g in genre_list.split('|')
    )
)

'''
BUILD USER-ITEM INTERACTIONS
Convert ratings to implicit feedback (or leave as is if you’re using ranking).
Use rating as is : EXPLICIT FEEDBACK since all ratings are informative
'''
joined_df['rating_norm'] = (
    (joined_df['rating'] - joined_df['rating'].min()) /
    (joined_df['rating'].max() - joined_df['rating'].min())
)

(interactions, weights) = dataset.build_interactions([
    (row['userId'], row['movieId'], row['rating_norm'])
    for _, row in joined_df.iterrows()
])

'''
Add item (movie) features (optional)
You can use movie genres or tags as features:
'''
# Extract genres list per movie
movies['genres_list'] = movies['genres'].apply(lambda g: g.split('|'))

# Build item features only once per movie
item_features = dataset.build_item_features(
    ((row['movieId'], row['genres_list']) for _, row in movies.iterrows())
)

# print("Interactions shape:", interactions.shape)

'''
TRAIN LIGHTFM
'''
# WARP is good for ranking recommendations
model = LightFM(loss='warp')
model.fit(interactions, item_features=item_features, epochs=20, num_threads=4)

'''
AFTER TRAINING SERIALIZE MODEL AND DATASETS INORDER TO SAVE TRAINED OBJECTS
'''
import pickle

# Save model, dataset, and item_features
with open('lightfm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('lightfm_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

with open('lightfm_item_features.pkl', 'wb') as f:
    pickle.dump(item_features, f)


'''
CONFIRMATION INTERNAL(lightfm) IDs well-mapped to external moviesID
Your movieIds are being stored as integers, not strings as seen before fit()
LightFM’s internal item mapping is correct and consistent:
e.g., movieId 1 → internal index 0, movieId 2 → 1, etc.
So the mapping logic and data alignment are now solid.
'''
# user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
# print(list(item_id_map.items())[:10])

# '''
# MAKE RECOMMENDATIONS
# '''
# import numpy as np

# def recommend_movies(user_id, model, dataset, interactions, item_features=None, conn=None,n_recs=10):
#     """
#     COLLABORATIVE FILTERING BASED RECOMMDER SYSTEM USING LIGHTFM 
#     LIGHTFM is a hybrid matrix factorization model capable of handling:
#         =>User-item interactions (ratings → explicit feedback)
#         =>Item features (like genres, tags, etc.)
    
#     GOAL:
#     Suggest movies based on users’ behavior patterns and movie similarity at the same time — exactly what powers systems like Netflix
#     """
#     # Get mappings
#     user_id_map, _, item_id_map, _ = dataset.mapping()
    
#     # Convert external(dataset) userId → internal(lightfm) index
#     if user_id not in user_id_map:
#         print(f"User {user_id} not found in mapping.")
#         return []
    
#     internal_uid = user_id_map[user_id]

#     n_users, n_items = interactions.shape
#     item_ids = np.arange(n_items)
#     scores = model.predict(internal_uid, item_ids, item_features=item_features)
#     # print(np.min(scores), np.max(scores))
#     top_items = np.argsort(-scores)[:n_recs]

#     # Map internal LightFM item indices (IDs) → external movieIds
#     id_to_movie = {v: k for k, v in item_id_map.items()}
#     top_movie_ids = [int(id_to_movie[i]) for i in top_items]
#     # print("Top movie IDs:", top_movie_ids[:10])

#     # Fetch titles from SQLite
#     placeholders = ', '.join('?' for _ in top_movie_ids)
#     query = f"SELECT movieId, title FROM movies WHERE movieId IN ({placeholders})"
#     recommended_df = pd.read_sql_query(query, conn, params=top_movie_ids)
    
#     # Nicely formatted output
#     print(f"\n🎬 Top {n_recs} Recommendations for userId {user_id}:")
#     print("------------------------------------------------------------")
#     for i, title in enumerate(recommended_df['title'].tolist(), start=1):
#         print(f"{i}. {title}")

#     return recommended_df['title'].tolist()

# print('_'*30)
# print(recommend_movies(2, model, dataset, interactions, item_features=item_features, conn=conn))
# print('_'*30)
# print(recommend_movies(70, model, dataset, interactions, item_features=item_features, conn=conn))
# print('_'*30)
# print(recommend_movies(53, model, dataset, interactions, item_features=item_features, conn=conn))
# print('_'*30)
# print(recommend_movies(23, model, dataset, interactions, item_features=item_features, conn=conn))


