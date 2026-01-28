# movierecommendationsystem



# 🎬 Movie Recommendation System using Machine Learning

This project implements a **content-based movie recommendation system** that recommends similar movies based on textual information such as genres and movie descriptions. The system uses **NLP techniques, TF features, and cosine similarity** to generate recommendations.

---

## 🚀 Project Overview

The goal of this system is to recommend movies that are similar to a selected movie from a dataset. This is achieved using:

- Text preprocessing
- Feature vectorization using **CountVectorizer**
- Computing similarity scores using **Cosine Similarity**
- Ranking movies based on similarity distance

---

## 📁 Dataset

The dataset contains 10,000 movies with attributes:

| Column | Description |
|--------|-------------|
| `id` | Unique movie identifier |
| `title` | Movie title |
| `genre` | List of genres |
| `overview` | Movie plot |
| `popularity` | Popularity score |
| `vote_average` | Rating |
| `vote_count` | Number of votes |
| `release_date` | Release year |
| `original_language` | Language code |

---

## 🧩 System Components

| Component | Description |
|----------|-------------|
| Vectorizer | Extracts key tokens from text |
| Similarity Engine | Computes pairwise similarity |
| Ranking Module | Sorts recommendations |
| Inference | Returns final suggestions |

---

## 📦 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- NLP (Bag-of-Words)
- Cosine Similarity

---

## 🧪 Results

The system successfully recommends movies with similar themes and narrative structure.

---

## 🔧 Usage / How to Run

```bash
pip install -r requirements.txt
python app.py
