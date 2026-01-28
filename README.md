# 🎬Movie Recommendation System using Machine Learning

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

Click the link below to download the CSV file:

[Download dataset.csv](./dataset.csv)

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
![View](V1.png)
![View](V1.png)

---

## 🔧 Usage / How to Run

```bash
pip install -r requirements.txt
python app.py

---

### Full Code + Outputs + Line-by-Line Comments

```python
# Import required libraries
# pandas → for data loading and manipulation
# numpy → for numerical operations (arrays)
import pandas as pd 
import numpy as np

# Load the movie dataset from CSV file
# This reads the file 'dataset.csv' into a pandas DataFrame called 'movies'
movies = pd.read_csv('dataset.csv')

# Display first 5 rows to understand the data structure
# Output shows columns and sample data
movies.head()
# Output:
   id                          title                   genre original_language                                           overview  popularity release_date  vote_average  vote_count
0   278     The Shawshank Redemption              Drama,Crime                en  Framed in the 1940s for the double murder of h...      94.075   1994-09-23           8.7       21862
1 19404  Dilwale Dulhania Le Jayenge     Comedy,Drama,Romance                hi  Raj is a rich, carefree, happy-go-lucky second...      25.408   1995-10-19           8.7        3731
2   238                 The Godfather              Drama,Crime                en  Spanning the years 1945 to 1955, a chronicle o...      90.585   1972-03-14           8.7       16280
3   424             Schindler's List        Drama,History,War                en  The true story of how businessman Oskar Schind...      44.761   1993-12-15           8.6       12959
4   240        The Godfather: Part II              Drama,Crime                en  In the continuing saga of the Corleone crime f...      57.749   1974-12-20           8.6        9811

# Show column names
# Helps confirm which features are available
movies.columns
# Output:
Index(['id', 'title', 'genre', 'original_language', 'overview', 'popularity',
       'release_date', 'vote_average', 'vote_count'],
      dtype='object')

# Show dataset summary (types, non-null counts, memory usage)
# Important to check for missing values
movies.info()
# Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 10000 non-null  int64  
 1   title              10000 non-null  object 
 2   genre              9997 non-null   object 
 3   original_language  10000 non-null  object 
 4   overview           9987 non-null   object 
 5   popularity         10000 non-null  float64
 6   release_date       10000 non-null  object 
 7   vote_average       10000 non-null  float64
 8   vote_count         10000 non-null  int64  
dtypes: float64(2), int64(2), object(5)
memory usage: 703.3+ KB

# Create a new column 'tags' by concatenating genre and overview
# This combines the textual features we will use for similarity
# No space added between — simple concatenation is enough for vectorization
movies['tags'] = movies['genre'] + movies['overview']

# Check updated DataFrame (tags column now exists)
movies.head()
# Output:
   id                          title                   genre original_language                                           overview  popularity release_date  vote_average  vote_count                                              tags
0   278     The Shawshank Redemption              Drama,Crime                en  Framed in the 1940s for the double murder of h...      94.075   1994-09-23           8.7       21862              Drama,CrimeFramed in the 1940s for the double ...
1 19404  Dilwale Dulhania Le Jayenge     Comedy,Drama,Romance                hi  Raj is a rich, carefree, happy-go-lucky second...      25.408   1995-10-19           8.7        3731     Comedy,Drama,RomanceRaj is a rich, carefree, h...
2   238                 The Godfather              Drama,Crime                en  Spanning the years 1945 to 1955, a chronicle o...      90.585   1972-03-14           8.7       16280              Drama,CrimeSpanning the years 1945 to 1955, a ...
3   424             Schindler's List        Drama,History,War                en  The true story of how businessman Oskar Schind...      44.761   1993-12-15           8.6       12959        Drama,History,WarThe true story of how busines...
4   240        The Godfather: Part II              Drama,Crime                en  In the continuing saga of the Corleone crime f...      57.749   1974-12-20           8.6        9811              Drama,CrimeIn the continuing saga of the Corle...

# Create a working DataFrame with only needed columns
new_df = movies[['id','title','genre','overview','tags']]

new_df.head()
# Output:
   id                          title                   genre                                           overview                                              tags
0   278     The Shawshank Redemption              Drama,Crime  Framed in the 1940s for the double murder of h...              Drama,CrimeFramed in the 1940s for the double ...
1 19404  Dilwale Dulhania Le Jayenge     Comedy,Drama,Romance  Raj is a rich, carefree, happy-go-lucky second...     Comedy,Drama,RomanceRaj is a rich, carefree, h...
2   238                 The Godfather              Drama,Crime  Spanning the years 1945 to 1955, a chronicle o...              Drama,CrimeSpanning the years 1945 to 1955, a ...
3   424             Schindler's List        Drama,History,War  The true story of how businessman Oskar Schind...        Drama,History,WarThe true story of how busines...
4   240        The Godfather: Part II              Drama,Crime  In the continuing saga of the Corleone crime f...              Drama,CrimeIn the continuing saga of the Corle...

# Drop genre & overview columns — we only need 'tags' now
new_df = new_df.drop(columns=['genre','overview'])

new_df.head()
# Output:
   id                          title                                              tags
0   278     The Shawshank Redemption              Drama,CrimeFramed in the 1940s for the double ...
1 19404  Dilwale Dulhania Le Jayenge     Comedy,Drama,RomanceRaj is a rich, carefree, h...
2   238                 The Godfather              Drama,CrimeSpanning the years 1945 to 1955, a ...
3   424             Schindler's List        Drama,History,WarThe true story of how busines...
4   240        The Godfather: Part II              Drama,CrimeIn the continuing saga of the Corle...

# Import vectorizer to convert text into numerical vectors
from sklearn.feature_extraction.text import CountVectorizer

# Create vectorizer: use top 10,000 words, remove English stop words
cv = CountVectorizer(max_features=10000, stop_words='english')

# Show vectorizer object
cv
# Output:
CountVectorizer(max_features=10000, stop_words='english')

# Safe loading & vectorization block (useful in notebooks)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

if 'movies' not in locals():
    movies = pd.read_csv('dataset.csv')

if 'tags' not in movies.columns:
    movies['tags'] = movies['genre'].fillna('') + movies['overview'].fillna('')

new_df = movies[['id','title','tags']]

cv = CountVectorizer(max_features=10000,stop_words='english')

# Transform tags into count vectors (Bag of Words)
vec = cv.fit_transform(new_df['tags'].values.astype('U')).toarray()

vec
# Output (sample):
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]])

# Check vector shape (rows = movies, columns = words)
vec.shape
# Output: (10000, 10000)

# Import similarity function
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise cosine similarity between all movies
sim = cosine_similarity(vec)

sim
# Output (sample):
array([[1.        , 0.06253054, 0.05802589, ..., 0.07963978, 0.07597372,
        0.03798686],
       [0.06253054, 1.        , 0.08980265, ..., 0.        , 0.        ,
        0.        ],
       [0.05802589, 0.08980265, 1.        , ..., 0.02541643, 0.03636965,
        0.        ],
       ...,
       [0.07963978, 0.        , 0.02541643, ..., 1.        , 0.03327792,
        0.03327792],
       [0.07597372, 0.        , 0.03636965, ..., 0.03327792, 1.        ,
        0.04761905],
       [0.03798686, 0.        , 0.        , ..., 0.03327792, 0.04761905,
        1.        ]])

# Find Shawshank Redemption (index 0)
new_df[new_df['title']=='The Shawshank Redemption']
# Output:
   id                          title                                              tags
0   278     The Shawshank Redemption              Drama,CrimeFramed in the 1940s for the double ...

# Sort similarity scores for Shawshank (index 0)
dist = sorted(list(enumerate(sim[0])),reverse=True, key = lambda vec: vec[1])

dist
# Output (full sorted list – top portion):
[(0, np.float64(1.0000000000000002)),
 (3709, np.float64(0.23539595453459988)),
 (3649, np.float64(0.22019275302527214)),
 (9006, np.float64(0.20751433915982243)),
 (2605, np.float64(0.20100756305184245)),
 (4068, np.float64(0.20100756305184245)),
 (6156, np.float64(0.20100756305184245)),
 (698, np.float64(0.19894589252079753)),
 (7324, np.float64(0.19767387315371682)),
 (1009, np.float64(0.19069251784911848)),
 (884, np.float64(0.1899342940993966)),
 (2963, np.float64(0.18802535827258876)),
 (7478, np.float64(0.1877810107252081)),
 (2991, np.float64(0.1860968420796942)),
 (715, np.float64(0.18582615562066462)),
 (7271, np.float64(0.18556740475630137)),
 (9520, np.float64(0.18148850216015694)),
 (2120, np.float64(0.17978662999019787)),
 (4201, np.float64(0.17912443020795962)),
 (9718, np.float64(0.17766726362967541)),
 ... (continues for all 10000 movies)

# Print top 5 titles (including self)
for i in dist[0:5]:
  print(new_df.iloc[i[0]].title)
# Output:
The Shawshank Redemption
Anything for Her
The Woodsman
The Getaway
Pusher II

# Final recommendation function
def recommend(movies):
  index = new_df[new_df['title']== movies].index[0]
  distance = sorted(list(enumerate(sim[index])),reverse=True, key = lambda vec: vec[1])
  for i in distance[0:5]:
    print(new_df.iloc[i[0]].title)

# Test function
recommend("Iron Man")
# Output:
Iron Man
Mazinger Z: Infinity
Justice League Dark
Iron Man 3
The Colony
