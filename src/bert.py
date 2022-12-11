import csv
import re
import sys
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification

config = {
  "_name_or_path": "bert-base-uncased",
  "architectures": [
    "BertForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "problem_type": "single_label_classification",
  "torch_dtype": "float32",
  "transformers_version": "4.24.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}


def get_file_information(input_file):
    file_info = []
    file = open(input_file)
    data = json.load(file)
    for datum in data["data"]:
        file_info.append([datum["headline"], datum["is_sarcastic"]])

    return file_info


def run_bert_model(data, url):
    tokenizer = AutoTokenizer.from_pretrained(url)
    model = AutoModelForSequenceClassification.from_pretrained(url, config=config)
    model.intermediate_size = 3073
    model.aaaaa = 5
    classification = []
    x = 0
    # Change this as needed to use more or less of the input data
    total = 5000
    sentences = []
    for datum in data:
        sentences.append(datum[0])
    while x < total:
        tokenized_text = tokenizer(sentences[x], padding=True, truncation=True, max_length=256, return_tensors="pt")
        output = model(**tokenized_text)
        probs = output.logits.softmax(dim=-1).tolist()[0]
        confidence = max(probs)
        classification.append(probs.index(confidence))
        x = x + 1
    write_results('bert-outputfile', data, classification, total)


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
    run_bert_model(file_info, "helinivan/english-sarcasm-detector")


main()
