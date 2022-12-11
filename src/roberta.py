import csv
import re
import sys
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def get_file_information(input_file):
    file_info = []
    file = open(input_file)
    data = json.load(file)
    for datum in data["data"]:
        file_info.append([datum["headline"], datum["is_sarcastic"]])

    return file_info


def run_pretrained_roberta_model(data, url):
    tokenizer = AutoTokenizer.from_pretrained(url)
    model = AutoModelForSequenceClassification.from_pretrained(url)
    classification = []
    x = 0
    # Change this as needed to use a larger or smaller batch size
    batch_size = 1000
    # Change this as needed to use more or less of the input data
    total = 5000
    sentences = []
    for datum in data:
        sentences.append(datum[0])
    while x < total:
        # We need to pass our sentences in batches because otherwise the computer this is running on will likely run
        # out of memory
        if x + batch_size < len(sentences):
            sentences_batch = sentences[x:x + batch_size]
        else:
            # We need to account for if our batch size puts us over the total size of sentences
            sentences_batch = sentences[x:len(sentences)-1]
        tokenized_text = tokenizer(sentences_batch, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**tokenized_text).logits.argmax(-1)
        predicted_token_class_ids = logits
        for y in range(0, len(predicted_token_class_ids)):
            # This is a bit weird, but we need to extract the actual classification from the Tensor, and there's not
            # a built-in way to do that
            classification.append(int(str(predicted_token_class_ids[y]).split("(")[1].split(")")[0]))
        x = x + batch_size
    write_results('roberta-outputfile', data, classification, total)


def write_results(file_name, data, classification, total):
    # write results to a new .tsv new file
    with open(file_name + '.tsv', 'w+', encoding='utf-8') as file:
        output = []
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        headers = ['headline', 'classification', 'actual_classification']
        writer.writerow(headers)
        index = 0
        total_sarcastic = 0
        total_not_sarcastic = 0
        accurate = 0
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        while index < total:
            output.append([data[index][0], classification[index], data[index][1]])
            if classification[index] == data[index][1]:
                accurate = accurate + 1
                if data[index][1] == 0:
                    tn = tn + 1
                else:
                    tp = tp + 1
            else:
                if classification[index] == 0:
                    fn = fn + 1
                else:
                    fp = fp + 1
            if classification[index] == 1:
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


def main():
    input_file = sys.argv[1]
    file_info = get_file_information(input_file)
    run_pretrained_roberta_model(file_info, "jkhan447/sarcasm-detection-RoBerta-base-newdata")


main()
