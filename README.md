# Goodreads Book Recommender System

This project builds a simple book recommender system using the Goodreads books dataset.

The goal of this mini project is to create two recommendation systems:

1. Popularity-Based Recommender
2. Content-Based Recommender

---

## Dataset

The dataset contains information about books from Goodreads.

The main columns are:

- `bookID`: Unique ID for each book
- `title`: Title of the book
- `authors`: Author of the book
- `average_rating`: Average rating given by users
- `isbn`: ISBN-10 number
- `isbn13`: ISBN-13 number
- `language_code`: Language of the book
- `num_pages`: Number of pages
- `ratings_count`: Number of ratings received
- `text_reviews_count`: Number of text reviews
- `publication_date`: Publication date of the book
- `publisher`: Publisher of the book

---

## Project Steps

### Step 1: Read the Dataset

The dataset is loaded into a Pandas DataFrame.

Extra spaces in column names are removed because the dataset contains a column named `num_pages` with extra spaces.

---

### Step 2: Explore the Dataset

The dataset is checked using:

- `.head()`
- `.shape`
- `.columns`
- `.isnull().sum()`

This helps understand the structure of the data before building the recommender system.

---

### Step 3: Popularity-Based Recommender

The popularity-based recommender recommends books based on a weighted rating formula.

A simple average rating is not always reliable because a book with only a few ratings may have a very high rating.

To solve this problem, the weighted rating formula considers:

- Average rating of the book
- Number of ratings received
- Mean rating of all books
- Minimum number of ratings required

Books with both high ratings and many ratings are ranked higher.

---

### Step 4: Content-Based Recommender

The content-based recommender recommends books similar to a selected book.

In this project, the `authors` column is used as the content feature.

The method uses:

- TF-IDF Vectorizer
- Cosine Similarity

TF-IDF converts author names into numerical vectors.

Cosine similarity compares books and finds the most similar ones.

For example, if the user selects a Harry Potter book, the system recommends books by the same or similar author information.

---

## Libraries Used

- pandas
- numpy
- scikit-learn

---

## How to Run the Project

First, install the required libraries:

```bash
pip install -r requirements.txt
