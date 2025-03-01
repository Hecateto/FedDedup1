from torch.utils.data import Dataset
import torch
import numpy as np
import csv
import os
from local_config import *
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset, concatenate_datasets

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def jokes_to_list(dataset_path="/kaggle/input/sonnets/"):
    short_jokes_path = os.path.join(dataset_path, "shortjokes.csv")

    joke_list = []

    with open(short_jokes_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            joke_str = f"{row[1]}"
            joke_list.append(joke_str)

    # shuffle the list
    np.random.shuffle(joke_list)

    return joke_list[:50000]


def imdb_to_list(dataset_path="/kaggle/input/sonnets/"):
    imdb_path = os.path.join(dataset_path, "imdb.csv")

    imdb_list = []

    with open(imdb_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            imdb_str = f"{row[0]}"
            imdb_list.append(imdb_str)

    np.random.shuffle(imdb_list)

    return imdb_list[:50000]


def rotten_to_list(dataset_path="/kaggle/input/sonnets/"):
    dataset = load_dataset("rotten_tomatoes")

    combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

    rotten_list = []

    for data in combined_dataset:
        rotten_list.append(data["text"])

    np.random.shuffle(rotten_list)

    if MODEL_NAME == "gpt2-large":
        return rotten_list[:5000]
    else:
        return rotten_list[:50000]


def haiku_to_list(dataset_path="/kaggle/input/sonnets/"):
    haiku_path = os.path.join(dataset_path, "haiku.csv")

    haiku_list = []

    with open(haiku_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            haiku_str = f"{row[2]}"
            haiku_list.append(haiku_str)

    np.random.shuffle(haiku_list)

    if MODEL_NAME == "gpt2-large":
        return haiku_list[:5000]
    else:
        return haiku_list[:50000]


def shakespeare_to_list(dataset_path="/kaggle/input/sonnets/"):
    dataset = load_dataset("Trelis/tiny-shakespeare")

    train_set = dataset["train"]
    test_set = dataset["test"]
    combined_set = concatenate_datasets([train_set, test_set])

    shakespeare_list = []

    for data in combined_set:
        shakespeare_list.append(data["Text"])

    np.random.shuffle(shakespeare_list)

    return shakespeare_list[:50000]


def sonnets_to_list(dataset_path="/kaggle/input/sonnets/"):
    sonnets_path = os.path.join(dataset_path, "sonnets.csv")

    sonnets_list = []

    with open(sonnets_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            sonnet_str = f"{row[2]}"
            sonnets_list.append(sonnet_str)

    sonnets_list = sonnets_list[1:]

    np.random.shuffle(sonnets_list)

    return sonnets_list[:50000]


def poetry_to_list(dataset_path="/kaggle/input/sonnets/"):
    poetry_path = os.path.join(dataset_path, "poetry.csv")

    poetry_list = []

    with open(poetry_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            poetry_str = f"{row[2]}"
            poetry_list.append(poetry_str)

    poetry_list = poetry_list[1:]
    np.random.shuffle(poetry_list)

    return poetry_list[:50000]


def get_text_dataset(dataset_name):
    if dataset_name == "Jokes":
        data = jokes_to_list()
    elif dataset_name == "IMDB":
        data = imdb_to_list()
    elif dataset_name == "Rotten":
        data = rotten_to_list()
    elif dataset_name == "Haiku":
        data = haiku_to_list()
    elif dataset_name == "Shakespeare":
        data = shakespeare_to_list()
    elif dataset_name == "Sonnets":
        data = sonnets_to_list()
    elif dataset_name == "Poetry":
        data = poetry_to_list()

    return data


class TextDataset(Dataset):

    def __init__(self, data, tokenizer, sample_weights=None):
        self.tokenizer = tokenizer

        self.input_ids = []
        self.attn_masks = []
        self.sample_weights = []

        self.data = data

        for i, text in enumerate(self.data):
            encodings_dict = tokenizer(
                BOS_TOKEN + text + EOS_TOKEN,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
            )

            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

            if sample_weights is not None:
                self.sample_weights.append(sample_weights[i])
            else:
                self.sample_weights.append(1.0)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.sample_weights[idx]


# FedAvg Local Training
def train_client(
        client_model,
        client_data_loader,
        client_id,
        round,
        sample_weights=None,
):
    total_train_loss = 0

    client_model.train()
    client_model.to(device)

    client_optimizer = AdamW(client_model.parameters(), lr=LEARNING_RATE)

    training_steps = ROUNDS * EPOCHS * len(client_data_loader)
    last_epoch = (round - 1) * EPOCHS * len(client_data_loader)
    warmup_steps = int(training_steps * 0.1)
    client_scheduler = get_linear_schedule_with_warmup(
        client_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )

    # dummy scheduler steps since for future rounds
    for _ in range(last_epoch):
        client_scheduler.step()

    for epoch in range(EPOCHS):

        epoch_loss = 0

        for idx, data in enumerate(client_data_loader):
            input_ids = data[0].to(device)
            attn_masks = data[1].to(device)

            if sample_weights is not None:
                batch_weights = torch.tensor(sample_weights[idx]).to(device)
            else:
                batch_weights = None

            client_model.zero_grad()

            outputs = client_model(
                input_ids=input_ids, attention_mask=attn_masks, labels=input_ids
            )

            loss = outputs[0]

            if batch_weights is not None:
                loss = (loss * batch_weights).mean()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            loss.backward()
            clip_grad_norm_(client_model.parameters(), 1.0)
            client_optimizer.step()
            client_scheduler.step()

        avg_epoch_loss = epoch_loss / len(client_data_loader)

        total_train_loss += avg_epoch_loss

    total_train_loss /= EPOCHS

    print(
        f"\n\nClient {client_id} in round {round} has average training loss of {total_train_loss}"
    )

    return total_train_loss


def train_client_amp(
        client_model,
        client_data_loader,
        client_id,
        round,
        sample_weights=None,
):

    client_optimizer = AdamW(client_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    training_steps = EPOCHS * len(client_data_loader)
    scheduler = get_linear_schedule_with_warmup(client_optimizer,
                                                num_warmup_steps=min(100, int(training_steps * 0.1)),
                                                num_training_steps=training_steps)
    scaler = torch.cuda.amp.GradScaler()

    client_model.train()
    total_train_loss = 0

    for epoch in range(EPOCHS):
        for batch_idx, (inputs, labels, weights) in enumerate(client_data_loader):
            input_ids, labels = inputs.to(device), labels.to(device)
            weights = torch.tensor(weights).to(device)

            client_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = client_model(input_ids=input_ids).logits
                loss_per_sample = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
                loss_per_sample = loss_per_sample.view(labels.size()).mean(dim=1)
                if weights is not None:
                    loss = (loss_per_sample * weights).mean()
                else:
                    loss = loss_per_sample.mean()
                # outputs = client_model(input_ids=input_ids, labels=labels)
                # loss = outputs[0]
                # if weights is not None:
                #     loss = (loss * weights).mean()
            scaler.scale(loss).backward()
            clip_grad_norm_(client_model.parameters(), 5.0) # Gradient clipping

            scaler.step(client_optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

    total_train_loss /= (len(client_data_loader) * EPOCHS)

    print(
        f"\n\nClient {client_id} in round {round} has average training loss of {total_train_loss}"
    )

    return total_train_loss

def compute_test_perplexity(model, test_data_loader):
    model.eval()
    model.to(device)

    total_loss = 0

    with torch.no_grad():
        for idx, data in enumerate(test_data_loader):
            input_ids = data[0].to(device)
            attn_masks = data[1].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attn_masks, labels=input_ids
            )

            loss = outputs[0]

            total_loss += loss.item()

    avg_loss = total_loss / len(test_data_loader)

    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()