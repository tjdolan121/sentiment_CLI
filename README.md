# ML API: Simple CLI tool for Sentiment Analysis
![sentiment_CLI](static/Screenshot.png?raw=true "sentiment_CLI")
## Enter a sentence, see if it leans "good" or "bad".

#### PURPOSE: I am exploring different ways to deliver machine learning products to end users.

#### In this project, I built a simple CLI tool for sentiment analysis as a first step in this exploration.
#### The model used is a simple Naive Bayes classifier from Scikit-learn.
#### My next venture will be integrating this model into a Flask application.
##### As always, feedback and contributions are welcome!
<hr>

### USAGE:

NOTE: This tool requires Python 3

STEP 1: Clone the project (in the terminal): ```git clone https://github.com/tjdolan121/sentiment_CLI.git```

STEP 2: Create a new virtual environment: ```virtualenv venv```

STEP 3: Activate the virtual environment: ```source venv/bin/activate```

STEP 4: Navigate to the project directory, "Sentiment_CLI"

STEP 5: Install requirements: ```pip install -r requirements.txt```

STEP 6: Run the main program: ```python3 main_program.py```
<hr>

### STRUCTURE:
##### ```main_program.py```: The primary application
##### ```model_creation.py```: Demonstrates how the NB model was constructed.
##### ```/pickled_objects```: Where the NB model and CountVectorizer objects are stored, to be retrieved by main_program.py at runtime.
##### ```/raw_data```: Datasets used for NB model
<hr>

### References
###### Dataset: 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015:
https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

###### Inspiration: Pluralsight's "How to Think about Machine Learning Algorithms" by Swetha Kolalapudi
https://www.pluralsight.com/courses/machine-learning-algorithms