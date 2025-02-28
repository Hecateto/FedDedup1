import torch
import numpy as np
import random
import math

from torch import nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, random_split
from ep_mpd import MultiPartyDeduplicator, EgPsiType, EgPsiDataType
import os
import time

from config import *
from utils_w import TextDataset, train_client_amp  , compute_test_perplexity, get_text_dataset
from copy import deepcopy

os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# set all seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


# 添加适配器
class Adapter(nn.Module):
    def __init__(self, dim, reduction_factor=8):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction_factor)
        self.up = nn.Linear(dim // reduction_factor, dim)
        nn.init.kaiming_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(nn.GELU()(self.down(x)))


def add_adapter_layers(model):
    for layer in model.transformer.h:
        # 获取隐藏维度
        hidden_dim = layer.mlp.c_fc.weight.shape[1]

        # 正确添加适配器模块
        layer.register_module("adapter", Adapter(hidden_dim))

        # 保存原始前向传播函数
        original_forward = layer.forward

        # 定义新的前向传播
        def new_forward(self, hidden_states, *args, **kwargs):
            # 原始前向传播
            outputs = original_forward(hidden_states, *args, **kwargs)

            # 在注意力之后添加适配器
            attn_output = outputs[0]
            adapted_output = self.adapter(attn_output)

            # 返回修改后的输出
            return (adapted_output,) + outputs[1:]

        # 更新层的前向传播方法
        layer.forward = new_forward
    return model


tokenizer = GPT2Tokenizer.from_pretrained(
    MODEL_NAME, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN
)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model = add_adapter_layers(model)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

if not os.path.exists(MODEL_CACHE):
    os.mkdir(MODEL_CACHE)

torch.save(
    model.state_dict(),
    os.path.join(MODEL_CACHE, f"global_{DATASET}.pt"),
)

############ Load dataset ############

full_data = get_text_dataset(dataset_name=DATASET)

# Original Dataset might have duplicates. Since we are artificially planting duplicates to observe their effect, clean up the original dataset
full_data = list(set(full_data))

train_data, test_data = random_split(full_data, [1 - TEST_RATIO, TEST_RATIO])

test_set = TextDataset(data=test_data, tokenizer=tokenizer)

############ Implant duplicates in dataset ############

if DUPLICATE_RATE > 0:
    train_data_len = len(train_data)
    num_duplicates = int(DUPLICATE_RATE * train_data_len)
    duplicate_indices = np.random.choice(train_data_len, num_duplicates, replace=True)
    train_data = torch.utils.data.ConcatDataset(
        [
            train_data,
            torch.utils.data.Subset(train_data, duplicate_indices),
        ]
    )

############ Split dataset among clients ############

splits = [1 / CLIENTS] * CLIENTS
if sum(splits) != 1:
    splits[-1] += 1 - sum(splits)
client_data = random_split(train_data, splits)

client_data_list = []

for data in client_data:
    client_data_list.append(list(data))

############ 不去重，分配权重（对数加权） ############
from collections import defaultdict
frequency_dict = defaultdict(int)
for data in client_data_list:
    for text in data:
        frequency_dict[text] += 1
epsilon = 1e-6
weights = {text: 1 / (math.log(freq + 1) + epsilon) for text, freq in frequency_dict.items()}


client_sample_weights = [
    [weights.get(text, 1.0) for text in client_data_list[i]]
    for i in range(CLIENTS)
]


client_datasets = [
    TextDataset(data=client_data_list[i],
                tokenizer=tokenizer,
                sample_weights=client_sample_weights[i],
            )
    for i in range(CLIENTS)
]


client_data_loaders = [
    DataLoader(client_datasets[i],
               batch_size=BATCH_SIZE,
               shuffle=True,
               collate_fn=lambda batch: (
                    torch.stack([x[0] for x in batch]),
                    torch.stack([x[1] for x in batch]),
                    [x[2] for x in batch],
               ),         # 加权
        )
    for i in range(CLIENTS)
]

############ FedAvg FL TRAINING STARTS ############

best_model = None
best_loss = math.inf
best_round = -1

best_ppl = math.inf
patience = 3

for round in range(ROUNDS):
    client_models = []
    client_sizes = [len(data) for data in client_data_list]
    total_size = sum(client_sizes)

    # Train the clients
    avg_loss = 0
    for client in range(CLIENTS):

        model.load_state_dict(torch.load(f"{MODEL_CACHE}/global_{DATASET}.pt"))

        start = time.time()
        client_loss = train_client_amp(
            model,
            client_data_loaders[client],
            client,
            round,
            sample_weights=client_sample_weights[client],
        )
        end = time.time()
        print(f"Client {client} round {round} took {end - start} seconds\n")

        avg_loss += client_loss
        torch.save(model.state_dict(), f"{MODEL_CACHE}/client_{client}_{DATASET}.pt")

    avg_loss /= CLIENTS

    print("\n\nAverage loss after round {} is {}\n".format(round, avg_loss))
    print("Average loss after round {} is {}\n".format(round, avg_loss))

    # Aggregate all client models and average them to get the new global model
    global_state = deepcopy(model.state_dict())

    start = time.time()
    for k in global_state:
        global_state[k] = sum(
            torch.load(f"{MODEL_CACHE}/client_{client}_{DATASET}.pt")[k] * client_sizes[client] / total_size
            for i in range(CLIENTS)
        )
    end = time.time()
    print(f"Aggregation took {end - start} seconds in round {round} \n")

    model.load_state_dict(global_state)
    torch.save(
        global_state,
        f"{MODEL_CACHE}/global_{DATASET}.pt",
    )

    # Evaluate the global model
    # Early Stopping
    current_ppl = compute_test_perplexity(model, DataLoader(
        TextDataset(test_data, tokenizer),
        batch_size=16
    ))
    if current_ppl < best_ppl:
        best_ppl, best_round = current_ppl, round
        best_model = deepcopy(model.state_dict())
        patience = 3
        torch.save(model.state_dict(), f"{MODEL_PATH}/best_model_{DATASET}.pt")
    else:
        patience -= 1
        if patience == 0:
            print(f"Early stopping at round {round}")
            break

############ Evaluate on test dataset ############

model.load_state_dict(best_model)
model.eval()

with torch.no_grad():
    inputs = tokenizer(BOS_TOKEN, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_return_sequences=5,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


test_ppl = compute_test_perplexity(model, DataLoader(
    TextDataset(test_data, tokenizer),
    batch_size=16
))

print(f"Test Perplexity: {test_ppl:.2f}")


