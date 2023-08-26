from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# training_data.data[0] - training dataset with the first index, we have with index 0
# [:10] - first 10 lines
# print('\n'.join(training_data.data[10].split('\n')[:30]))
# print('Target is: ', training_data.target_names[training_data.target[10]])

# we just count the word occurences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
print(count_vector.vocabulary_)

# tokenizing is going to count the occurences of given words in the sentence
# This is my home town!
# This 1(ocuurence 1)
# is 1
# my 1
# home 1
# town 1

# its going to end up with document term matrix for the first article, second article and so on
# instead of sentences, here we are dealing with concrete articles with lots of sentences

# it returns a dictionary with key value pairs
# where value is the number of occurences and key is the given word itself

# we transform the word occurences into tf-idf
# TfidfVectorizer = CountVectorizer + TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

# TfidfTransformer is going to transform countVectorizer into a tf-idf vectorizer

# counting the occurence according to tokenizing is not going to work fine because we should get rit of
# these words that do not contain much information
# better approach is tf-idf approach

# x_train_counts - it is basically the fitted countVectorizer on the training data
# Now, this will be trained tf-idf

print(x_train_tfidf)

# At the beginning, we had a huge amount of text sentences and words
# But of course, machine learning algorithms can operate on numbers
# So, somehow, we have to transform these sentences, words and texts into numerical value

# Now, we can use multinomial gaussian naive bayes algorithm - MultinomialNB

model = MultinomialNB().fit(x_train_tfidf, training_data.target)

# we are going to fit the MultinomialNB classifier as far as the training tfidf is concerned
# and the training data that target

new = [# 'This has going to do with church or religion',
    'This has going to do with Masjid or religion',
    'Software engineering is getting hotter and hotter nowadays']

new = ['My favourite topic has something to do with quantum physics and quantum mechanics',
       'This has going to do with Masjid or religion',
       'Software engineering is getting hotter and hotter nowadays']

# First, we have to transform these sentences into numerical value.

# So, we have to use tokenizer and tfidf transformer in order to get the numerical representations.

x_new_counts = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

# Then, we can make prediction with the model
# The model is the MultinomialNB classifier

# The model that predicts on x_new_tfidf, this x_new_tfidf is the numerical representation of these past sentences.

predicted = model.predict(x_new_tfidf)

# Now, as we have multiple past sentences
# we are going to make the for loop
# As you can see, there's going to be the document basically the sentence itself
# and the category in zip new and predicted

# First of all, we are going to print the document
# and the document doc is going to be this sentence 'This has going to do with church or religion'
# and this sentence and the predicted is going to be the prediction.
# predicted = model.predict(x_new_tfidf)

# So, basically what is the category of these sentences
# 'This has going to do with church or religion', 'Software engineering is getting hotter and hotter nowadays'
# What we have to do?
# training_data, target_names, category
# because we have already made predictions here
# predicted = model.predict(x_new_tfidf)
# Here for doc, category in zip(new, predicted):
# we are just iterate through this 1-D array

# print(predicted)  this predicted is going to store 2 values,
# one category for the above 1st sentence ans the 2nd category for the above 2nd sentence

print(predicted)

for doc, category in zip(new, predicted):
    print('%r --------> %s' % (doc, training_data.target_names[category]))
    # print(values, sep, end, file, flush)

# 'My favourite topic has something to do with quantum physics and quantum mechanics', - physics, alt.atheism
# prediction is not going to work fine
# because in the categories we have defined, there is no physics topic
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# So the algorithm hadn't considered the physics related texts

# Naive Bayes classifier is a supervised learning algorithm
# which means we need the labels, we need the target values

# So basically what is the topic of a given text and we are going to deal with K means clustering

# A clustering algorithm is able to find the similar topics as far as the different texts are concerned
# which means that we can cluster different texts without targeting the values and label,
# which is an extremely powerful technique
