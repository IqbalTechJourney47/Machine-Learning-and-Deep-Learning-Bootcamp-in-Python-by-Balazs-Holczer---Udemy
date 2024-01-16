from sklearn.feature_extraction.text import TfidfVectorizer
# Convert a collection of raw documents to a matrix of TF-IDF features.

vec = TfidfVectorizer()

# How to decide whether the text and sentences are similar or not
tfidf = vec.fit_transform(['I like machine learning and clustering algorithms',
                           'Apples, oranges and any kind of fruits are healthy',
                           'Is it feasible with machine learning algorithms?',
                           'My family is happy because of the healthy fruits'])

# fit_transform(raw_documents[, y])
# Learn vocabulary and idf, return document-term matrix.

print(tfidf.A)

# similarity matrix
# with as many columns and as many rows as the number of sentences we have
print((tfidf*tfidf.T).A)