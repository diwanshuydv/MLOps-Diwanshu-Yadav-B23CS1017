import os
import json
import gzip
import random
import pickle
import requests
import torch
from transformers import DistilBertTokenizerFast

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))

def prepare_datasets(model_name='distilbert-base-cased', max_length=512):
    genre_url_dict = {
        'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
        'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
        'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
        'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
        'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
        'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
        'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
        'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
    }

    if os.path.exists('genre_reviews_dict.pickle'):
        with open('genre_reviews_dict.pickle', 'rb') as f:
            genre_reviews_dict = pickle.load(f)
    else:
        genre_reviews_dict = {}
        for genre, url in genre_url_dict.items():
            print(f'Loading reviews for genre: {genre}')
            # Loading smaller subset to run quickly
            genre_reviews_dict[genre] = load_reviews(url, head=5000, sample_size=500)
        with open('genre_reviews_dict.pickle', 'wb') as f:
            pickle.dump(genre_reviews_dict, f)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for _genre, _reviews in genre_reviews_dict.items():
        _reviews = random.sample(_reviews, min(len(_reviews), 500))
        split_idx = int(len(_reviews) * 0.8)
        
        for _review in _reviews[:split_idx]:
            train_texts.append(_review)
            train_labels.append(_genre)
        for _review in _reviews[split_idx:]:
            test_texts.append(_review)
            test_labels.append(_genre)

    unique_labels = sorted(list(set(train_labels)))
    label2id = {label: id for id, label in enumerate(unique_labels)}
    id2label = {id: label for label, id in label2id.items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    return train_dataset, test_dataset, label2id, id2label, tokenizer
