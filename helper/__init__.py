import requests

from .service_helper import LoginHelper
from .engine import Engine
from .dataset import Dataset


def retrieve_hyperparameter_config(project):
    lr = None
    batch_size = None
    r = requests.get(f"https://huggingface.co/{project}/raw/main/README.md")
    tmp = r.text.split("\n")
    for line in tmp:
        if "learning_rate" in line:
            lr = float(line.split(": ")[1])
        if "batch_size" in line:
            batch_size = int(line.split(": ")[1])

    return lr, batch_size