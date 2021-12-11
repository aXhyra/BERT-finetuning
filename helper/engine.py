import numpy as np
import torch
from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
from datasets import load_metric
import wandb


class Engine:
    model_checkpoint = "distilbert-base-uncased"

    @staticmethod
    def compute_metrics(eval_pred):
        metric = load_metric("f1")
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")

    @staticmethod
    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
        }

    def __init__(self, data, args, device="cuda:0", model=None):
        self.model = None
        self.args = args
        self.trainer = None
        self.best_run = None
        self.dataset = data
        self.results = None
        self.device = device
        if model is not None:
            self.load_model(model)

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(Engine.model_checkpoint, id2label=self.dataset.labels,
                                                                  num_labels=self.dataset.n_classes, return_dict=True)

    def load_model(self, model):
        pass

    def load_trainer(self, use_init=False):
        if use_init:
            self.trainer = Trainer(
                model_init=self.model_init,
                args=self.args,
                train_dataset=self.dataset.tokenized_dataset['train'],
                eval_dataset=self.dataset.tokenized_dataset['validation'],
                tokenizer=self.dataset.tokenizer,
                compute_metrics=Engine.compute_metrics,
            )
        else:
            self.trainer = Trainer(
                model=self.model,
                args=self.args,
                train_dataset=self.dataset.tokenized_dataset['train'],
                eval_dataset=self.dataset.tokenized_dataset['validation'],
                tokenizer=self.dataset.tokenizer,
                compute_metrics=Engine.compute_metrics,
            )

    def load_train_args(self, opt_name="test", lr=2e-5, epochs=4, batch_size=16, push_to_hub=False, seed=0):
        self.args = TrainingArguments(
            opt_name,
            seed=seed,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            push_to_hub=push_to_hub,
            metric_for_best_model="f1",
            report_to="wandb",
            save_total_limit=1,
            run_name=opt_name,
        )

    def train(self, epochs, seed=0, opt_name="test", use_given_args=True):
        if not use_given_args:
            self.load_train_args(opt_name,
                                 self.best_run.hyperparameters["learning_rate"],
                                 epochs,
                                 self.best_run.hyperparameters["per_device_train_batch_size"],
                                 True, seed)
            self.load_trainer(True)
        else:
            self.load_trainer(False)

        self.results = self.trainer.train()
        self.trainer.push_to_hub()
        wandb.finish()

    def evaluate(self):
        if self.trainer is None:
            print("[!] Training required")
            return
        self.trainer.evaluate()

    def hyperparameter_search(self, n_trials=5):
        if self.args is None:
            print("[!] TraininArgument object is required")
            return -1
        self.load_trainer(True)
        self.best_run = self.trainer.hyperparameter_search(n_trials=n_trials, direction="maximize",
                                                           hp_space=self.my_hp_space)
        wandb.finish()

    def load_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.dataset.tokenizer = tokenizer
        self.model.to(self.device)

    def predict(self, input_text):
        input_text_tokenized = self.dataset.tokenizer(input_text, truncation=True, padding=True, return_tensors="pt")
        if self.device is not None:
            input_text_tokenized.to(self.device)
        prediction = self.model(**input_text_tokenized).logits

        result = torch.softmax(prediction, dim=1)
        result = np.argmax(result.tolist())

        return self.dataset.labels[str(result)]
