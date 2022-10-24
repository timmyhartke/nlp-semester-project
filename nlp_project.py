import csv
import re
import sys
import json

import nltk.lm
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.linear_model import LogisticRegression


def get_file_information(input_file):
    file_info = []
    file = open(input_file)
    data = json.load(file)
    for datum in data["data"]:
        file_info.append([datum["headline"], datum["is_sarcastic"]])

    return file_info

def get_words(sentence):
    # This uses a modified version of the regex I created for assignment 1 to get all 'words'.
    # I am including mentions and hashtags as unique words, so #hello, @hello, and hello are three distinct words.
    # I am not counting numbers as words
    word_re = "@?#?[A-Za-z]+-?[A-Za-z]+"
    words = re.findall(word_re, sentence)
    # This is also a convenient place to standardize our case, and I'm choosing lowercase
    index = 0
    while index < len(words):
        words[index] = words[index].lower()
        index = index + 1
    return words


def get_unique_words(sentences):
    # This is a helper function that gets all unique words in a list of sentences
    unique_words = []
    for sentence in sentences:
        for word in sentence:
            if word not in unique_words:
                unique_words.append(word)
    return unique_words


def train_LR_model(all_data):
    model = SklearnClassifier(LogisticRegression())
    features = generate_features(all_data, True)

    model.train(features)

    return model


def test_LR_model(all_data, model):

    features = generate_features(all_data, False)
    classifications = model.classify_many(features)
    probabilities = model.prob_classify_many(features)

    # write results to new file called 'lr_outputfile.tsv'
    with open('lr_outputfile.tsv', 'w+', encoding='utf-8') as file:
        output = []
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        headers = ['id', 'tweet', 'offensive_probability', 'classification']
        writer.writerow(headers)
        index = 0
        for line in all_data:
            output.append([line[0], line[1], probabilities[index].prob(1), classifications[index]])
            index = index + 1

        writer.writerows(output)


def generate_features(sentences, is_training_data):
    features = []
    number_of_words = 0
    for sentence in sentences:
        for word in get_words(sentence[0]):
            number_of_words = number_of_words + 1

        sentence_features = dict(
            words=number_of_words,
        )
        if is_training_data:
            # For training data, we want to know whether the tweet is offensive or not
            features.append([sentence_features, sentence[1]])
        else:
            # For testing data, we just want the features
            features.append(sentence_features)

    return features


def main():
    input_file = sys.argv[1]
    file_info = get_file_information(input_file)
    model = train_LR_model(file_info)
    test_LR_model(file_info, model)



main()
