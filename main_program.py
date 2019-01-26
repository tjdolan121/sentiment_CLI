#!/usr/bin/env python3
"""CLI Tool for analyzing sentiment of a sentence.

   Accepts: A user-inputted sentence as a string.

   Returns: A string indicating whether the sentence
    has a "good" or "bad" sentiment """


from sklearn.externals import joblib
import sys


sys.path.append('/pickled_objects')


def load_pickles():
    countvectorizer = joblib.load("pickled_objects/countvectorizer.pkl")
    bernoulli_model = joblib.load("pickled_objects/bernoulli_model.pkl")
    return countvectorizer, bernoulli_model


def analyze_sentence():
    countvectorizer = load_pickles()[0]
    bernoulli_model = load_pickles()[1]
    while True:
        sentence = input("Enter a sentence to be analyzed, or type exit to leave: ")
        if sentence == "exit":
            exit(1)
        prediction = int(bernoulli_model.predict(countvectorizer.transform([sentence]))[0])
        if prediction == 1:
            prediction = "good"
        else:
            prediction = "bad"
        print("'{}' has a '{}' sentiment".format(sentence, prediction))


if __name__ == "__main__":
    print("Loading model, please wait a moment...")
    analyze_sentence()
