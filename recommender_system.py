import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv("books(1).csv", on_bad_lines="skip")

books.columns = books.columns.str.strip()

print("Dataset loaded successfully.")
print("Dataset shape:", books.shape)

print("\nColumns:")
print(books.columns.tolist())

print("\nFirst five rows:")
print(books.head())

print("\nMissing values:")
print(books.isnull().sum())

books = books[
    [
        "bookID",
        "title",
        "authors",
        "average_rating",
        "isbn",
        "isbn13",
        "language_code",
        "num_pages",
        "ratings_count",
        "text_reviews_count",
        "publication_date",
        "publisher"
    ]
]

books["average_rating"] = pd.to_numeric(
    books["average_rating"],
    errors="coerce"
)

books["ratings_count"] = pd.to_numeric(
    books["ratings_count"],
    errors="coerce"
)

books = books.dropna(
    subset=[
        "title",
        "authors",
        "average_rating",
        "ratings_count"
    ]
)

print("\nCleaned dataset shape:", books.shape)

def popularity_recommender(data, top_n=10):

    books_pop = data.copy()

    C = books_pop["average_rating"].mean()

    m = books_pop["ratings_count"].quantile(0.90)

    qualified_books = books_pop[
        books_pop["ratings_count"] >= m
    ].copy()

    R = qualified_books["average_rating"]

    v = qualified_books["ratings_count"]

    qualified_books["weighted_score"] = (
        (v / (v + m) * R) +
        (m / (v + m) * C)
    )

    recommendations = qualified_books.sort_values(
        "weighted_score",
        ascending=False
    )

    return recommendations[
        [
            "title",
            "authors",
            "average_rating",
            "ratings_count",
            "weighted_score"
        ]
    ].head(top_n)

print("\nTop 10 Popular Books:")
print(popularity_recommender(books, top_n=10))

content_books = books.copy()

content_books["authors"] = content_books["authors"].fillna("")
content_books["title"] = content_books["title"].fillna("")
content_books["publisher"] = content_books["publisher"].fillna("")
content_books["language_code"] = content_books["language_code"].fillna("")

content_books = content_books.drop_duplicates(
    subset=["title"]
).reset_index(drop=True)

content_books["content"] = (
    content_books["title"] + " " +
    content_books["authors"] + " " +
    content_books["publisher"] + " " +
    content_books["language_code"]
)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    content_books["content"]
)

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

indices = pd.Series(
    content_books.index,
    index=content_books["title"]
)

def content_based_recommender(title, top_n=10):

    if title not in indices.index:
        return "Book title not found in the dataset."

    idx = indices[title]

    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n + 1]

    book_indices = [item[0] for item in sim_scores]

    recommendations = content_books.iloc[book_indices][
        [
            "title",
            "authors",
            "average_rating",
            "ratings_count",
            "publisher"
        ]
    ].copy()

    recommendations["similarity_score"] = [
        item[1] for item in sim_scores
    ]

    return recommendations

example_book = "Harry Potter and the Half-Blood Prince (Harry Potter  #6)"

print("\nContent-Based Recommendations:")
print(content_based_recommender(example_book, top_n=10))

print("\nTop 5 Popular Books:")
print(popularity_recommender(books, top_n=5))

print("\nBooks similar to Harry Potter:")
print(
    content_based_recommender(
        "Harry Potter and the Half-Blood Prince (Harry Potter  #6)",
        top_n=5
    )
)

print("\nBooks similar to The Hobbit:")
print(
    content_based_recommender(
        "The Hobbit  or There and Back Again",
        top_n=5
    )
)
