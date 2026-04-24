# ============================================================
#   Movie Recommendation System (Content-Based Filtering)
#   Author  : Your Name
#   Method  : TF-IDF on Genres + Cosine Similarity + Ratings
#   Libraries: pandas, numpy, matplotlib, scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  BUILT-IN DATASET  (50 popular movies)
# ─────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """50 well-known movies with genre tags and IMDb-style ratings."""
    data = [
        # (title,                              genres,                                  rating, votes,    year)
        ("The Shawshank Redemption",           "Drama Crime",                            9.3,  2700000,  1994),
        ("The Godfather",                      "Crime Drama",                            9.2,  1900000,  1972),
        ("The Dark Knight",                    "Action Crime Drama Thriller",            9.0,  2700000,  2008),
        ("Pulp Fiction",                       "Crime Drama Thriller",                   8.9,  2100000,  1994),
        ("Schindler's List",                   "Biography Drama History War",            9.0,  1400000,  1993),
        ("The Lord of the Rings",              "Adventure Drama Fantasy",                8.9,  1900000,  2003),
        ("Fight Club",                         "Drama Thriller",                         8.8,  2100000,  1999),
        ("Forrest Gump",                       "Drama Romance Comedy",                   8.8,  2000000,  1994),
        ("Inception",                          "Action Adventure Sci-Fi Thriller",       8.8,  2400000,  2010),
        ("The Matrix",                         "Action Sci-Fi",                          8.7,  1900000,  1999),
        ("Goodfellas",                         "Biography Crime Drama",                  8.7,  1100000,  1990),
        ("The Silence of the Lambs",           "Crime Drama Thriller",                   8.6,  1400000,  1991),
        ("Interstellar",                       "Adventure Drama Sci-Fi",                 8.7,  1900000,  2014),
        ("The Lion King",                      "Animation Adventure Drama",              8.5,  1100000,  1994),
        ("Gladiator",                          "Action Adventure Drama",                 8.5,  1500000,  2000),
        ("The Departed",                       "Crime Drama Thriller",                   8.5,  1300000,  2006),
        ("Whiplash",                           "Drama Music",                            8.5,   750000,  2014),
        ("The Prestige",                       "Drama Mystery Sci-Fi Thriller",          8.5,  1300000,  2006),
        ("Memento",                            "Mystery Thriller",                       8.4,  1200000,  2000),
        ("The Green Mile",                     "Crime Drama Fantasy",                    8.6,  1300000,  1999),
        ("Avengers: Endgame",                  "Action Adventure Sci-Fi",                8.4,  1100000,  2019),
        ("Joker",                              "Crime Drama Thriller",                   8.4,  1100000,  2019),
        ("Parasite",                           "Comedy Drama Thriller",                  8.5,   800000,  2019),
        ("1917",                               "Drama War",                              8.3,   600000,  2019),
        ("Dunkirk",                            "Action Drama History War",               7.9,   700000,  2017),
        ("Arrival",                            "Drama Mystery Sci-Fi",                   7.9,   800000,  2016),
        ("Blade Runner 2049",                  "Action Drama Mystery Sci-Fi",            8.0,   600000,  2017),
        ("La La Land",                         "Comedy Drama Music Romance",             8.0,   700000,  2016),
        ("The Grand Budapest Hotel",           "Adventure Comedy Crime Drama",           8.1,   750000,  2014),
        ("Mad Max: Fury Road",                 "Action Adventure Sci-Fi Thriller",       8.1,   950000,  2015),
        ("Get Out",                            "Horror Mystery Thriller",                7.7,   700000,  2017),
        ("A Quiet Place",                      "Drama Horror Mystery Sci-Fi",            7.5,   600000,  2018),
        ("Hereditary",                         "Drama Horror Mystery Thriller",          7.3,   350000,  2018),
        ("The Witch",                          "Drama Horror Mystery",                   6.9,   250000,  2015),
        ("Midsommar",                          "Drama Horror Mystery",                   7.1,   280000,  2019),
        ("Toy Story",                          "Animation Adventure Comedy Family",      8.3,  1000000,  1995),
        ("Finding Nemo",                       "Animation Adventure Comedy Family",      8.1,  1000000,  2003),
        ("Up",                                 "Animation Adventure Comedy Drama",       8.2,  1000000,  2009),
        ("WALL-E",                             "Animation Adventure Romance Sci-Fi",     8.4,  1000000,  2008),
        ("Coco",                               "Animation Adventure Comedy Family",      8.4,   600000,  2017),
        ("Spider-Man: Into the Spider-Verse",  "Animation Action Adventure",             8.4,   600000,  2018),
        ("Spirited Away",                      "Animation Adventure Fantasy",            8.6,   700000,  2001),
        ("Your Name",                          "Animation Drama Romance",                8.4,   250000,  2016),
        ("Princess Mononoke",                  "Animation Action Adventure Fantasy",     8.3,   350000,  1997),
        ("Howl's Moving Castle",               "Animation Adventure Fantasy Romance",    8.2,   350000,  2004),
        ("The Social Network",                 "Biography Drama",                        7.8,   700000,  2010),
        ("Moneyball",                          "Biography Drama Sport",                  7.6,   350000,  2011),
        ("The Big Short",                      "Biography Comedy Drama",                 7.8,   400000,  2015),
        ("Catch Me If You Can",                "Biography Crime Drama",                  8.1,   900000,  2002),
        ("Cast Away",                          "Adventure Drama",                        7.8,   600000,  2000),
    ]

    df = pd.DataFrame(data, columns=["title", "genres", "rating", "votes", "year"])
    df["votes_m"] = (df["votes"] / 1_000_000).round(2)
    return df


# ─────────────────────────────────────────────
# 2.  BUILD CONTENT-BASED MODEL
# ─────────────────────────────────────────────
def build_model(df: pd.DataFrame):
    """
    Content-Based Filtering:
      - TF-IDF vectors on genre tags
      - Normalised rating similarity
      - Combined: 70% genre + 30% rating
    """
    # Genre similarity via TF-IDF + Cosine
    tfidf        = TfidfVectorizer(token_pattern=r"[A-Za-z\-]+")
    tfidf_matrix = tfidf.fit_transform(df["genres"])
    genre_sim    = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Rating similarity
    scaler      = MinMaxScaler()
    rating_norm = scaler.fit_transform(df[["rating"]])
    rating_sim  = cosine_similarity(rating_norm, rating_norm)

    # Weighted combination
    combined_sim = 0.70 * genre_sim + 0.30 * rating_sim
    return combined_sim, tfidf, tfidf_matrix


# ─────────────────────────────────────────────
# 3.  RECOMMEND BY SIMILAR MOVIE TITLE
# ─────────────────────────────────────────────
def recommend(title: str, df: pd.DataFrame, sim_matrix: np.ndarray,
              top_n: int = 5) -> pd.DataFrame:
    """Return top_n movies most similar to the given title."""
    matches = df[df["title"].str.lower() == title.strip().lower()]
    if matches.empty:
        # partial match fallback
        matches = df[df["title"].str.lower().str.contains(title.strip().lower())]
    if matches.empty:
        print(f"  ❌  '{title}' not found in dataset.")
        return pd.DataFrame()

    idx    = matches.index[0]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if i != idx][:top_n]

    results = df.iloc[[i for i, _ in scores]][["title", "genres", "rating", "year"]].copy()
    results["similarity_%"] = [round(s * 100, 1) for _, s in scores]
    results = results.reset_index(drop=True)
    results.index += 1
    return results


# ─────────────────────────────────────────────
# 4.  RECOMMEND BY GENRE + MIN RATING
# ─────────────────────────────────────────────
def recommend_by_genre(genre: str, df: pd.DataFrame,
                       min_rating: float = 7.5, top_n: int = 5) -> pd.DataFrame:
    """Return top-rated movies matching a genre with at least min_rating."""
    mask = (
        df["genres"].str.contains(genre, case=False) &
        (df["rating"] >= min_rating)
    )
    results = (
        df[mask]
        .sort_values("rating", ascending=False)
        .head(top_n)[["title", "genres", "rating", "year"]]
        .reset_index(drop=True)
    )
    results.index += 1
    return results


# ─────────────────────────────────────────────
# 5.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def explore_data(df: pd.DataFrame) -> None:
    print("\n" + "="*58)
    print("   DATASET OVERVIEW")
    print("="*58)
    print(f"\n  Total movies   : {len(df)}")
    print(f"  Year range     : {df['year'].min()} – {df['year'].max()}")
    print(f"  Rating range   : {df['rating'].min()} – {df['rating'].max()}")
    print(f"  Avg rating     : {df['rating'].mean():.2f}")

    all_genres = []
    for g in df["genres"]:
        all_genres.extend(g.split())
    genre_counts = Counter(all_genres)

    print(f"\n  Genre frequencies (top 8):")
    for genre, count in genre_counts.most_common(8):
        bar = "█" * count
        print(f"    {genre:<16} {count:>3}  {bar}")

    print(f"\n  Top 5 rated movies:")
    for _, row in df.nlargest(5, "rating").iterrows():
        print(f"    ⭐ {row['rating']}  {row['title']} ({row['year']})")


# ─────────────────────────────────────────────
# 6.  DEMO RECOMMENDATIONS
# ─────────────────────────────────────────────
def demo_recommendations(df: pd.DataFrame, sim_matrix: np.ndarray) -> None:
    print("\n" + "="*58)
    print("   DEMO: SIMILAR MOVIE RECOMMENDATIONS")
    print("="*58)

    for movie in ["Inception", "The Dark Knight", "Spirited Away", "Parasite"]:
        print(f"\n  🎬  Because you liked '{movie}':")
        print(f"  {'─'*56}")
        recs = recommend(movie, df, sim_matrix, top_n=5)
        if not recs.empty:
            print(recs.to_string())

    print("\n" + "="*58)
    print("   DEMO: GENRE-BASED RECOMMENDATIONS")
    print("="*58)

    for genre, min_r in [("Sci-Fi", 8.0), ("Animation", 8.2), ("Horror", 7.0)]:
        print(f"\n  🎭  Top {genre} movies (rating ≥ {min_r}):")
        print(f"  {'─'*56}")
        recs = recommend_by_genre(genre, df, min_rating=min_r, top_n=5)
        if not recs.empty:
            print(recs.to_string())


# ─────────────────────────────────────────────
# 7.  VISUALISATIONS  (6-panel report)
# ─────────────────────────────────────────────
def plot_results(df: pd.DataFrame, sim_matrix: np.ndarray) -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0d0d1a",
        "axes.facecolor"  : "#13132a",
        "axes.edgecolor"  : "#2e2e50",
        "axes.labelcolor" : "#dde0f5",
        "xtick.color"     : "#9090b8",
        "ytick.color"     : "#9090b8",
        "text.color"      : "#dde0f5",
        "grid.color"      : "#22224a",
        "grid.linestyle"  : "--",
        "grid.alpha"      : 0.45,
        "font.family"     : "monospace",
    })

    GOLD, PURPLE, TEAL, RED, GREEN = "#f4b942", "#8b5cf6", "#06b6d4", "#f43f5e", "#10b981"

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("🎬  Movie Recommendation System — Analysis Report",
                 fontsize=18, fontweight="bold", color="#dde0f5", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── 1 : Rating Distribution ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["rating"], bins=15, color=PURPLE, edgecolor="#0d0d1a", alpha=0.9)
    ax1.axvline(df["rating"].mean(), color=GOLD, linestyle="--", linewidth=2,
                label=f"Mean: {df['rating'].mean():.2f}")
    ax1.set(xlabel="IMDb Rating", ylabel="Number of Movies", title="Rating Distribution")
    ax1.set_title("Rating Distribution", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True)

    # ── 2 : Top 10 Rated Movies (horizontal bar) ──────────
    ax2 = fig.add_subplot(gs[0, 1])
    top10 = df.nlargest(10, "rating").iloc[::-1]
    short = [t[:22] + "…" if len(t) > 22 else t for t in top10["title"]]
    cols  = [GOLD if r >= 9.0 else TEAL if r >= 8.7 else PURPLE for r in top10["rating"]]
    ax2.barh(range(len(top10)), top10["rating"], color=cols, edgecolor="#0d0d1a")
    ax2.set_yticks(range(len(top10))); ax2.set_yticklabels(short, fontsize=7)
    ax2.set(xlabel="Rating", title="Top 10 Rated Movies", xlim=(7.9, 9.6))
    ax2.set_title("Top 10 Rated Movies", fontweight="bold")
    for i, val in enumerate(top10["rating"]):
        ax2.text(val + 0.01, i, f"{val}", va="center", fontsize=7, color="#dde0f5")
    ax2.grid(True, axis="x")

    # ── 3 : Top 10 Genres ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    all_genres = []
    for g in df["genres"]: all_genres.extend(g.split())
    genre_counts = Counter(all_genres)
    top_g = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    pal   = [TEAL, PURPLE, GOLD, RED, GREEN] * 2
    ax3.barh(list(top_g.keys()), list(top_g.values()), color=pal, edgecolor="#0d0d1a")
    ax3.set(xlabel="Number of Movies", title="Top 10 Genres")
    ax3.set_title("Top 10 Genres", fontweight="bold")
    ax3.invert_yaxis(); ax3.grid(True, axis="x")

    # ── 4 : Movies Per Decade ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"
    decade_counts = df["decade"].value_counts().sort_index()
    ax4.bar(decade_counts.index, decade_counts.values,
            color=PURPLE, edgecolor="#0d0d1a", alpha=0.9)
    ax4.set(xlabel="Decade", ylabel="Number of Movies", title="Movies by Decade")
    ax4.set_title("Movies by Decade", fontweight="bold")
    ax4.grid(True, axis="y")

    # ── 5 : Rating vs Year Scatter ────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sc = ax5.scatter(df["year"], df["rating"], c=df["rating"],
                     cmap="plasma", s=60, alpha=0.85, edgecolors="white", linewidths=0.3)
    plt.colorbar(sc, ax=ax5, label="Rating", fraction=0.046, pad=0.04)
    m, b = np.polyfit(df["year"], df["rating"], 1)
    xr = np.linspace(df["year"].min(), df["year"].max(), 100)
    ax5.plot(xr, m * xr + b, "--", color=GOLD, linewidth=1.8, label="Trend")
    ax5.set(xlabel="Year", ylabel="Rating", title="Rating vs Year")
    ax5.set_title("Rating vs Year", fontweight="bold")
    ax5.legend(fontsize=8); ax5.grid(True)

    # ── 6 : Similarity Heatmap (top 12 movies) ────────────
    ax6 = fig.add_subplot(gs[1, 2])
    idx12    = df.nlargest(12, "rating").index.tolist()
    sub_sim  = sim_matrix[np.ix_(idx12, idx12)]
    short12  = [t[:13] + "…" if len(t) > 13 else t for t in df.loc[idx12, "title"]]
    im = ax6.imshow(sub_sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax6.set_xticks(range(12)); ax6.set_xticklabels(short12, rotation=45, ha="right", fontsize=6)
    ax6.set_yticks(range(12)); ax6.set_yticklabels(short12, fontsize=6)
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    ax6.set_title("Similarity Heatmap\n(Top 12 Movies)", fontweight="bold")

    plt.savefig("results/analysis_report.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  📊  Plot saved → results/analysis_report.png")
    plt.show()


# ─────────────────────────────────────────────
# 8.  INTERACTIVE MODE
# ─────────────────────────────────────────────
def interactive_mode(df: pd.DataFrame, sim_matrix: np.ndarray) -> None:
    print("\n" + "="*58)
    print("   🎬  INTERACTIVE RECOMMENDATION ENGINE")
    print("="*58)
    print("\n  Sample movie titles you can try:")
    for t in df.sample(8, random_state=7)["title"]:
        print(f"    • {t}")
    print("\n  (type 'quit' to exit)\n")

    while True:
        user_input = input("  Enter a movie title: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye! 🎬"); break
        recs = recommend(user_input, df, sim_matrix, top_n=5)
        if not recs.empty:
            print(f"\n  ✅  Top 5 similar to '{user_input}':\n")
            print(recs.to_string(), "\n")


# ─────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────
def main():
    import os
    os.makedirs("results", exist_ok=True)

    print("\n" + "="*58)
    print("   MOVIE RECOMMENDATION SYSTEM")
    print("   Method : Content-Based Filtering")
    print("   Features: Genre (TF-IDF) + Rating (Cosine Sim)")
    print("="*58)

    print("\n[1/5]  Loading dataset …")
    df = load_dataset()
    df.to_csv("results/movies.csv", index=False)
    print(f"       ✅  {len(df)} movies loaded → results/movies.csv")

    print("\n[2/5]  Exploratory Data Analysis …")
    explore_data(df)

    print("\n[3/5]  Building content-based similarity model …")
    sim_matrix, tfidf, tfidf_matrix = build_model(df)
    print(       "       ✅  Similarity matrix built!")
    print(f"       Matrix shape: {sim_matrix.shape}")

    print("\n[4/5]  Running demo recommendations …")
    demo_recommendations(df, sim_matrix)

    print("\n[5/5]  Generating visualisation report …")
    plot_results(df, sim_matrix)

    print("\n" + "="*58)
    print("   ✅  All done! Check the 'results/' folder.")
    print("="*58)

    ans = input("\n  Try the interactive mode? (yes / no): ").strip().lower()
    if ans == "yes":
        interactive_mode(df, sim_matrix)


if __name__ == "__main__":
    main()
