import csv
import re
import sys
import json

import nltk.lm
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.linear_model import LogisticRegression

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_numeric, remove_stopwords, stem_text


def get_file_information(input_file):
    file_info = []
    file = open(input_file)
    data = json.load(file)
    for datum in data["data"]:
        file_info.append([datum["headline"], datum["is_sarcastic"]])

    return file_info


def get_word2vec_model(file):
    sentences = get_sentences(file)

    # There are some significant decisions being made in these two lines. First, I've set vector size to 300 because
    # from doing a bit of research that seems to be the optimal vector size. I've set window to 2, which is small,
    # but I kept getting unexpected results when using a larger window. I don't know if this is because the results were
    # actually weird or if I was letting my own bias creep too much into the decision. I used a min_count of 10 so that
    # rare words were excluded from the models
    model = Word2Vec(sentences=sentences, vector_size=300, window=2, min_count=10, workers=4)


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
        headers = ['headline', 'sarcastic_probability', 'classification', 'actual_classification']
        writer.writerow(headers)
        index = 0
        total_sarcastic = 0
        total_not_sarcastic = 0
        accurate = 0
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        while index < len(all_data):
            output.append([all_data[index][0], probabilities[index].prob(1), classifications[index], all_data[index][1]])
            if classifications[index] == all_data[index][1]:
                accurate = accurate + 1
                if all_data[index][1] == 0:
                    tn = tn + 1
                else:
                    tp = tp + 1
            else:
                if classifications[index] == 0:
                    fn = fn + 1
                else:
                    fp = fp + 1
            if classifications[index] == 1:
                total_sarcastic = total_sarcastic + 1
            else:
                total_not_sarcastic = total_not_sarcastic + 1
            index = index + 1
        print("True Positive: " + str(tp))
        print("True Negative: " + str(tn))
        print("False Positive: " + str(fp))
        print("False Negative: " + str(fn))
        print("Sarcastic: " + str(total_sarcastic / index))
        print("Accuracy: " + str(accurate / index))
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
            # For training data, we want to know whether the tweet is sarcastic or not
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
