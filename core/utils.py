# # def recommend_movies(user_id,n_recs=10):
# #     """
# #     GOAL:
# #     Suggest movies based on users’ behavior patterns and movie similarity at the same time — 
# #     exactly what powers systems like Netflix
# #     """
# #     # Get mappings
# #     user_id_map, _, item_id_map, _ = dataset.mapping()
    
# #     # Convert external(dataset) userId → internal(lightfm) index
# #     if user_id not in user_id_map:
# #         print(f"User {user_id} not found in mapping.")
# #         return []
    
# #     internal_uid = user_id_map[user_id]

# #     n_users, n_items = interactions.shape
# #     item_ids = np.arange(n_items)
# #     scores = model.predict(internal_uid, item_ids, item_features=item_features)
# #     # print(np.min(scores), np.max(scores))
# #     top_items = np.argsort(-scores)[:n_recs]

# #     # Map internal LightFM item indices (IDs) → external movieIds
# #     id_to_movie = {v: k for k, v in item_id_map.items()}
# #     top_movie_ids = [int(id_to_movie[i]) for i in top_items]
# #     # print("Top movie IDs:", top_movie_ids[:10])

# #     # Fetch titles from SQLite
# #     placeholders = ', '.join('?' for _ in top_movie_ids)
# #     query = f"SELECT movieId, title FROM movies WHERE movieId IN ({placeholders})"
# #     recommended_df = pd.read_sql_query(query, conn, params=top_movie_ids)
    
# #     # Nicely formatted output
# #     print(f"\n🎬 Top {n_recs} Recommendations for userId {user_id}:")
# #     print("------------------------------------------------------------")
# #     for i, title in enumerate(recommended_df['title'].tolist(), start=1):
# #         print(f"{i}. {title}")

# #     return recommended_df['title'].tolist()


# # core/utils.py
# import numpy as np
# import pandas as pd
# import sqlite3
# from .load_model import get_recommmender_model  

# def recommend_movies(user_id, n_recs=10):
#     """
#     Generate top N movie recommendations for a given user_id.

#     GOAL:
#     Suggest movies based on users’ behavior patterns and movie similarity at the same time — exactly what powers systems like Netflix
#     """
#     model, dataset, item_features = get_recommmender_model()

#     # Since the training used in-memory SQLite, we must rebuild a mini SQLite
#     # for fetching movie titles
#     movies = pd.read_csv('./../data/movies.csv')  # or adjust path
#     conn = sqlite3.connect(':memory:')
#     movies.to_sql('movies', conn, index=False, if_exists='replace')

#     user_id_map, _, item_id_map, _ = dataset.mapping()
#     if user_id not in user_id_map:
#         return {"error": f"User {user_id} not found in dataset."}

#     internal_uid = user_id_map[user_id]
#     n_users, n_items = dataset.interactions_shape()

#     item_ids = np.arange(n_items)
#     scores = model.predict(internal_uid, item_ids, item_features=item_features)
#     top_items = np.argsort(-scores)[:n_recs]

#     id_to_movie = {v: k for k, v in item_id_map.items()}
#     top_movie_ids = [int(id_to_movie[i]) for i in top_items]

#     placeholders = ', '.join('?' for _ in top_movie_ids)
#     query = f"SELECT movieId, title FROM movies WHERE movieId IN ({placeholders})"
#     recommended_df = pd.read_sql_query(query, conn, params=top_movie_ids)

#     return recommended_df['title'].tolist()



