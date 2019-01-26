from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


# Retrieve the data


datafiles = ["raw_data/amazon_cells_labelled.txt",
             "raw_data/imdb_labelled.txt",
             "raw_data/yelp_labelled.txt", ]

data = []

for file in datafiles:
    with open(file, "r") as txt_file:
        data += txt_file.read().split('\n')


# Clean the data


data = [line.split('\t') for line in data if len(line.split('\t')) == 2 and line.split("\t")[1] != '']


# Split the data into features and labels


train_features = [item[0] for item in data]
train_labels = [int(item[1]) for item in data]


# Encode features using CountVectorizer


countvectorizer = CountVectorizer(binary='true')
train_features = countvectorizer.fit_transform(train_features)  # train_features is now a sparse matrix


# Create and train model


bernoulli_model = BernoulliNB().fit(train_features, train_labels)


# Create a simple test for accuracy


test_sentences = [("This TV show is bad", 0),
                  ("I cannot stand this food", 0),
                  ("When will they learn?", 0),
                  ("Which is why I'll never go back", 0),
                  ("Can we agree this place stinks?", 0),
                  ("Quality food at a quality price", 1),
                  ("I cannot get enough of this food!", 1),
                  ("Exceptional experience!", 1),
                  ("No better place in the area!", 1),
                  ("Cheap food, but tasted great!", 1)]


count = 0
total_sentences = len(test_sentences)
for item in test_sentences:
    prediction = int(bernoulli_model.predict(countvectorizer.transform([item[0]])))
    if prediction == item[1]:
        count += 1

print("accuracy: {}%".format((count/total_sentences)*100))


# Export model for use in CLI tool

joblib.dump(countvectorizer, "pickled_objects/countvectorizer.pkl")
joblib.dump(bernoulli_model, "pickled_objects/bernoulli_model.pkl")
