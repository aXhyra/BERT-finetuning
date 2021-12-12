# Finetuning DistilBERT for sentiment analysis on tweets
## Advanced Topics in Machine Learning, 2021. 
### Marco Latella, Pietro Tropeano, Alind Xhyra

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/aXhyra/BERT-finetuning/blob/master/notebook.ipynb
        :alt: Open In Colab
        

The focus of this project is to fine-tune a DistilBERT  model over 4 different classification tasks on the tweet-eval dataset.
 
To perform the experimental phase, 3 different helper classes were created. It is possible to find such classes in the helper folder.
- Service helper, to handle the login process in the used services 
- Dataset helper, to load and work with different datasets
- Engine helper, to perform optimization and training on the chosed models

## More informations

To further readings please refer to the report present in the home of the project.

## Used services

| Package | README |
| ------ | ------ |
| Hugging Face | https://huggingface.co|
| Wandb | https://wandb.ai/site|
