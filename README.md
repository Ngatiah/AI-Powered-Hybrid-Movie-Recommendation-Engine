# 🎬 Simple AI-powered Movie Recommendation System (Django + LightFM)

An AI-powered movie recommendation system built with Django and LightFM, combining Collaborative Filtering (CF) and Content-Based Filtering (CBF) to deliver personalized movie suggestions.

The system learns user preferences from historical ratings and movie metadata (genres), using matrix factorization and feature embeddings to generate recommendations similar to Netflix’s personalization approach.

---

## 🚀 Features
- 🔹 **Hybrid Recommendation Engine** — merges collaborative and content-based signals
- 🔹 **Machine Learning Model (LightFM)** trained on data from the **MovieLens Dataset (Kaggle)**
- 🔹 **Django Web Integration** — interactive search and real-time recommendation display
- 🔹  **AJAX + JSON API** for dynamic frontend updates
- 🔹 **SQLite** for lightweight data storage
- 🔹 **Responsive UI** for user search and recommendation results
- 
---

## 🧰 Tech Stack
| Tool | Purpose |
|------|----------|
| **Python** | Core Programming Language |
| **Django** | Web Framework (backend + template rendering) |
| **Pandas** | Data manipulation and preprocessing |
| **LightFM** | Hybrid Recommendation Model |
| **NumPy** | Numerical computations and matrix operations |
| **SQLite** | Lightweight database |
| **HTML** / **JS** / **Fetch API** | Frontend Template Interactivity |


---

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ngatiah/Movie-Recommendation-System.git
   cd movie-recommender

2. **Create virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   pip install -r requirements.txt

3. **Run database migrations** 
   ```bash
   python manage.py makemigrations
   python manage.py migrate

4. **Start Django server** 
   ```bash
   python manage.py runserver

5. **Access the app** 
   Open your browser and go to :
   ```bash
   http://127.0.0.1:8000/MovieRecommender/
   ```
   Type a movie name or genre in the search box to get your **Top-N recommendations**


## Example Output 
   ```bash
 🎬 Top 10 Recommendations for “Toy Story”:
1. Finding Nemo (2003)
2. Monsters, Inc. (2001)
3. Incredibles, The (2004)
4. Shrek (2001)
5. Lion King, The (1994)
6. Ice Age (2002)
7. Ratatouille (2007)
8. Despicable Me (2010)
9. WALL·E (2008)
10. Up (2009)
  ```

## Dataset
   This project uses the **MovieLens Dataset** available on [https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset]:
   - **movies.csv** — Movie titles, IDs, and genres  
   - **ratings.csv** — User–movie rating interactions  
   - **tags.csv** — User-generated tags

## FUTURE ENHANCEMENT
   - Incorporate Contextual Recommendations
   - Incorporate more advanced user behaviours tracking such as clicks,scroll behaviour,likes/dislikes,pauses/skips,e.t.c.
   - Incorporate Deep Learning Models as well as A/B Testing at Scale  

## NOTE
   Netflix’s real system is more advanced, but conceptually, it blends collaborative and content-based signals — just like my LightFM hybrid recommender does. 
   So my project mimics the same AI foundation, albeit on a smaller scale.
