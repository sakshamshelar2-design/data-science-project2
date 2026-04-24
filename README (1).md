# 🎬 Movie Recommendation System

A machine learning project that recommends movies using **Content-Based Filtering** — powered by **TF-IDF on genres** and **cosine similarity on ratings**.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Goal** | Recommend movies similar to a given title or genre |
| **Method** | Content-Based Filtering |
| **Similarity** | TF-IDF (genres) + Cosine Similarity (ratings) |
| **Language** | Python 3.8+ |
| **Libraries** | pandas, numpy, matplotlib, scikit-learn |

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── movie_recommender.py     ← Main script (run this)
├── requirements.txt         ← Python dependencies
├── README.md                ← You are here
│
└── results/                 ← Auto-created on first run
    ├── movies.csv               ← Full movie dataset
    └── analysis_report.png      ← 6-panel visualisation
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python movie_recommender.py
```

---

## 🧠 How It Works

### Content-Based Filtering

The system recommends movies by measuring how *similar* two movies are based on their features — no user history needed.

```
Final Similarity = 0.70 × Genre Similarity + 0.30 × Rating Similarity
```

**Step-by-step:**
1. Genre tags (e.g. `"Action Sci-Fi Thriller"`) are converted to **TF-IDF vectors**
2. **Cosine similarity** is computed between all movie pairs
3. Ratings are **normalised** (0–1) and a rating similarity matrix is built
4. Both matrices are **combined** with weighted averaging
5. For any input movie, the top-N most similar movies are returned

---

## 🎯 Two Recommendation Modes

### Mode 1 — Similar to a movie title
```python
recommend("Inception", df, sim_matrix, top_n=5)
```
Returns 5 movies most similar to *Inception*.

### Mode 2 — By genre + minimum rating
```python
recommend_by_genre("Sci-Fi", df, min_rating=8.0, top_n=5)
```
Returns top 5 Sci-Fi movies rated ≥ 8.0.

---

## 📊 Sample Output

```
[4/5]  Running demo recommendations …

  🎬  Because you liked 'Inception':
  ────────────────────────────────────────────────────────
      title                        genres                  rating  year  similarity_%
  1   The Matrix                   Action Sci-Fi             8.7  1999          87.3
  2   Interstellar                 Adventure Drama Sci-Fi    8.7  2014          85.1
  3   Arrival                      Drama Mystery Sci-Fi      7.9  2016          80.4
  4   The Prestige                 Drama Mystery Sci-Fi...   8.5  2006          79.8
  5   Blade Runner 2049            Action Drama Mystery...   8.0  2017          76.2

  🎭  Top Sci-Fi movies (rating ≥ 8.0):
  ────────────────────────────────────────────────────────
      title                        genres                  rating  year
  1   Inception                    Action Adventure Sci-Fi   8.8  2010
  2   Interstellar                 Adventure Drama Sci-Fi    8.7  2014
  3   The Matrix                   Action Sci-Fi             8.7  1999
  4   The Prestige                 Drama Mystery Sci-Fi...   8.5  2006
  5   Arrival                      Drama Mystery Sci-Fi      7.9  2016
```

---

## 🖼️ Visualisation Report

The script generates a 6-panel dark-themed chart saved to `results/analysis_report.png`:

| Panel | Description |
|-------|-------------|
| Rating Distribution | Histogram of all movie ratings |
| Top 10 Rated Movies | Horizontal bar chart |
| Top 10 Genres | Genre frequency across dataset |
| Movies by Decade | Count of movies per decade |
| Rating vs Year | Scatter plot with trend line |
| Similarity Heatmap | Pairwise similarity of top 12 movies |

---

## 📚 Dataset

Built-in dataset of **50 popular movies** including:
- IMDb-style rating (6.9 – 9.3)
- Multi-label genre tags
- Release year (1972 – 2019)
- Vote counts

Genres covered: Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Fantasy, Family, History, Horror, Music, Mystery, Romance, Sci-Fi, Sport, Thriller, War

---

## 📚 Concepts Covered

- Content-Based Filtering
- TF-IDF Vectorisation (`TfidfVectorizer`)
- Cosine Similarity
- Feature normalisation (`MinMaxScaler`)
- Exploratory Data Analysis (EDA)
- Data visualisation (matplotlib)

---

## 🔮 Future Improvements

- [ ] Add a Streamlit web app UI
- [ ] Integrate real Kaggle movie dataset (TMDB / MovieLens)
- [ ] Add Collaborative Filtering (user-based)
- [ ] Build a Hybrid Recommender (content + collaborative)
- [ ] Add movie poster fetching via TMDB API
- [ ] Add director / cast as additional features

---

## 👤 Author

**Your Name**  
B.Tech / BSc Student  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
