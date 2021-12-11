from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
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

    def get_partition_classes_distribution(self, partition):
        class_occurrences = [0] * self.n_classes
        if partition == "all":
            for partition in self.dataset:
                for example in self.dataset[partition]:
                    class_occurrences[int(example["label"])] += 1
        else:
            for example in self.dataset[partition]:
                class_occurrences[int(example["label"])] += 1

        return class_occurrences

    def plot_classes_distribution(self, partition, title="Classes Distribution"):
        dist = self.get_partition_classes_distribution(partition)
        classes = list(self.labels.values())
        for i in range(len(classes)):
            print(classes[i] + ": " + str(dist[i]))
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_title(title)
        ax.bar(classes, dist)
        plt.show()


if __name__ == '__main__':
    dataset = Dataset("tweet_eval", "bert-base-uncased", s_key1="text")
    dataset.plot_classes_distribution("train", "Emotion, training class distribution")
    dataset.plot_classes_distribution("validation", "Emotion, validation class distribution")
    dataset.plot_classes_distribution("test", "Emotion, test class distribution")
    dataset.plot_classes_distribution("all", "Emotion, class distribution")
