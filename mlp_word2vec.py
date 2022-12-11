import csv
import re
import sys
import json

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_numeric, remove_stopwords, \
    strip_punctuation, strip_short, stem_text
from gensim.similarities import Similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier

word2vec = None
all_data = []


def get_file_information(input_file):
    file_info = []
    file = open(input_file)
    data = json.load(file)
    for datum in data["data"]:
        file_info.append([datum["headline"], datum["is_sarcastic"]])

    return file_info


def train_MLP_model(num_layers=1):
    sentences = get_sentences()
    is_sarcastic = get_sarcastic_attribute()
    global word2vec
    word2vec = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=10, workers=4)
    embeddings = get_mean_word2vec_vectors(sentences)

    model = MLPClassifier(hidden_layer_sizes=(100 for x in range(0, num_layers)), solver='lbfgs',
                          max_iter=750)
    model.fit(embeddings, is_sarcastic)

    return model


def test_MLP_model(model):
    global all_data

    sentences = get_sentences()
    embeddings = get_mean_word2vec_vectors(sentences)
    classifications = model.predict(embeddings)
    probabilities = model.predict_proba(embeddings)

    # write results to new file called 'mlp_outputfile.tsv'
    with open('mlp_outputfile.tsv', 'w+', encoding='utf-8') as file:
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
        for line in all_data:
            output.append([line[0], probabilities[index][1], classifications[index], line[1]])
            if classifications[index] == line[1]:
                accurate = accurate + 1
                if line[1] == 0:
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
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = 2 * (precision * recall) / (precision + recall)
        print("True Positive: " + str(tp))
        print("True Negative: " + str(tn))
        print("False Positive: " + str(fp))
        print("False Negative: " + str(fn))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("f1score: " + str(f1score))
        print("Sarcastic: " + str(total_sarcastic / index))
        print("Accuracy: " + str(accurate / index))
        writer.writerows(output)


def get_mean_word2vec_vectors(sentences):
    # We get the mean word2vec vectors for each sentence and return them in an array
    embeddings = []
    for sentence in sentences:
        sentence_embeddings = []
        for word in sentence:
            try:
                sentence_embeddings.append(word2vec.wv[word])
            # If the word is not in the word2vec model, we ignore it
            except KeyError:
                continue
        sentence_average_embedding = [0 for x in range(0, 100)]
        for embedding in sentence_embeddings:
            for index in range(0, len(embedding)):
                sentence_average_embedding[index] = sentence_average_embedding[index] + embedding[index]
        if len(sentence) > 0:
            for index in range(0, len(sentence_average_embedding)):
                sentence_average_embedding[index] = sentence_average_embedding[index] / len(sentence)
        embeddings.append(sentence_average_embedding)

    return embeddings


def get_sentences():
    global all_data
    all_sentences = []
    for line in all_data:
        all_sentences.append(get_words(line[0]))
    return all_sentences


def get_words(sentence):
    # we want all our words to be lowercase, we are getting rid of numbers, removing short
    # words, and stemming words. The stemming results in some words looking a bit weird
    filters = [lambda x: x.lower(), strip_numeric, strip_punctuation, strip_short, stem_text, remove_stopwords]
    filtered_words = preprocess_string(sentence, filters)
    return filtered_words


def get_sarcastic_attribute():
    # This is a helper function that returns an array of sarcastic attributes items from the training file
    global all_data
    is_sarcastic = []
    for line in all_data:
        is_sarcastic.append(line[1])

    return is_sarcastic


def main():
    global all_data
    input_file = sys.argv[1]
    all_data = get_file_information(input_file)
    model = train_MLP_model()
    test_MLP_model(model)


main()
