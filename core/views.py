from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import pandas as pd
import os
import numpy as np
from difflib import get_close_matches
from .load_model import get_recommmender_model

# Load data and model once (global, efficient)
model, dataset, item_features = get_recommmender_model()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
movies_path = os.path.join(BASE_DIR, "data", "movies.csv")
movies_core = pd.read_csv(movies_path) 
movies_core['genres'] = movies_core['genres'].fillna('')

@csrf_exempt
def search_recommendations(request):
    query = request.GET.get('q', '').strip().lower()

    # default page : template
    if not query:
        return render(request, 'search_movie.html')
    
    # Find close matches in titles or genres
    matched_titles = movies_core[movies_core['title'].str.lower().str.contains(query)]
    matched_genres = movies_core[movies_core['genres'].str.lower().str.contains(query)]
    matched = pd.concat([matched_titles, matched_genres]).drop_duplicates()

    if matched.empty:
        return JsonResponse({'results': []})

    # Take the first matching movie as the "seed"
    seed_movie_id = matched.iloc[0]['movieId']

    # Compute similarity scores (simple cosine or model-based)
    item_id_map = dataset.mapping()[2]  # external→internal item map
    if seed_movie_id not in item_id_map:
        return JsonResponse({'results': []})

    internal_item_id = item_id_map[seed_movie_id]
    n_items = dataset.interactions_shape()[1]
    item_ids = np.arange(n_items)

    # Predict similarity scores for all items
    scores = model.predict(0, item_ids, item_features=item_features)  # using dummy user 0
    # top_items = np.argsort(-scores)[:10]
    # top_items = np.argsort(-scores)[:20]
    # top_items = np.argsort(-scores)[:50]
    top_items = np.argsort(-scores)[:100]


    id_to_movie = {v: k for k, v in item_id_map.items()}
    top_movie_ids = [int(id_to_movie[i]) for i in top_items]
    recs = movies_core[movies_core['movieId'].isin(top_movie_ids)][['title', 'genres']].to_dict('records')

    return JsonResponse({'results': recs})
