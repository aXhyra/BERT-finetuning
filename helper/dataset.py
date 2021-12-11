from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import requests


class Dataset:
    dataset = None
    tokenized_dataset = None

    def __init__(self, task, tokenizer_name, s_key1="text", s_key2=None):
        self.dataset_name = "tweet_eval"
        self.task = task
        self.s_key1 = s_key1
        self.s_key2 = s_key2
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.dataset = load_dataset(self.dataset_name, task)
        self.tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True)
        self.n_classes = np.max(self.dataset["validation"]["label"]) + 1
        self.labels = {}
        self.retrieve_labels()

    def retrieve_labels(self):

        r = requests.get(
            "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/" + self.task + "/mapping.txt")
        tmp = r.text.split("\n")
        for el in tmp:
            if len(el) > 1:
                tmp2 = el.split("\t")
                self.labels[tmp2[0]] = tmp2[1]

    def preprocess_function(self, dataset):
        if self.s_key2 is None:
            return self.tokenizer(dataset[self.s_key1], truncation=True)
        return self.tokenizer(dataset[self.s_key1], dataset[self.s_key2], truncation=True)
