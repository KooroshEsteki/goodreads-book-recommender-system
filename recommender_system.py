# Goodreads Book Recommender System

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 
# Step 1: Read the dataset
#

books = pd.read_csv("books.csv", on_bad_lines="skip")

# Clean column names because num_pages has extra spaces in the dataset
books.columns = books.columns.str.strip()

print("Dataset loaded successfully.")
print("Dataset shape:", books.shape)
print("\nColumns:")
print(books.columns)


#
# Step 2: Basic exploration
# 

print("\nFirst five rows:")
print(books.head())

print("\nMissing values:")
print(books.isnull().sum())


# 
# Step 3: Keep useful columns
# 

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

# Drop rows with missing important values
books = books.dropna(subset=["title", "authors", "average_rating", "ratings_count"])

print("\nCleaned dataset shape:", books.shape)


# 
# Step 4: Popularity-Based Recommender
# 

def popularity_recommender(data, top_n=10):
    """
    This function recommends books based on popularity.

    It uses a weighted rating formula similar to IMDb.
    The formula considers both:
    1. The average rating of the book
    2. The number of ratings received by the book
    """

    books_pop = data.copy()

    # C is the mean rating of all books
    C = books_pop["average_rating"].mean()

    # m is the minimum number of ratings required
    m = books_pop["ratings_count"].quantile(0.90)

    # Select books that have enough ratings
    qualified_books = books_pop[books_pop["ratings_count"] >= m].copy()

    # R is the average rating of each book
    R = qualified_books["average_rating"]

    # v is the number of ratings for each book
    v = qualified_books["ratings_count"]

    # Weighted rating formula
    qualified_books["weighted_score"] = (v / (v + m) * R) + (m / (v + m) * C)

    # Sort books by weighted score
    recommendations = qualified_books.sort_values("weighted_score", ascending=False)

    return recommendations[
        [
            "title",
            "authors",
            "average_rating",
            "ratings_count",
            "weighted_score"
        ]
    ].head(top_n)


# Test popularity recommender
print("\nTop 10 Popular Books:")
print(popularity_recommender(books, top_n=10))


#
# Step 5: Content-Based Recommender
# 

content_books = books.copy()

content_books["authors"] = content_books["authors"].fillna("")
content_books["title"] = content_books["title"].fillna("")

content_books = content_books.drop_duplicates(subset=["title"]).reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(content_books["authors"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(
    content_books.index,
    index=content_books["title"]
)

def content_based_recommender(title, top_n=10):

    if title not in indices.index:
        return "Book title not found in the dataset."

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n + 1]

    book_indices = [i[0] for i in sim_scores]

    recommendations = content_books.iloc[book_indices][
        [
            "title",
            "authors",
            "average_rating",
            "ratings_count"
        ]
    ].copy()

    recommendations["similarity_score"] = [
        i[1] for i in sim_scores
    ]

    return recommendations


# Test content-based recommender
example_book = "Harry Potter and the Half-Blood Prince (Harry Potter  #6)"

print("\nContent-Based Recommendations:")
print(content_based_recommender(example_book, top_n=10))


# 
# Step 6: More examples
# 

print("\nTop 5 Popular Books:")
print(popularity_recommender(books, top_n=5))

print("\nBooks similar to Harry Potter:")
print(content_based_recommender("Harry Potter and the Half-Blood Prince (Harry Potter  #6)", top_n=5))

print("\nBooks similar to The Hobbit:")
print(content_based_recommender("The Hobbit  or There and Back Again", top_n=5))
