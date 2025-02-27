# LLM Configuration (优化后使用轻量级模型)
import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


MODEL_NAME = "distilgpt2"  # 更改为4位量化的轻量模型
USE_LORA = True  # 启用LoRA高效微调
LOAD_IN_4BIT = True  # 启用4位量化

CLIP_GRAD = True
MAX_GRAD_NORM = 1.0
DYNAMIC_PADDING = True



# 量化配置参数
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = False
BNB_4BIT_QUANT_TYPE = "nf4"

# 训练参数优化
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数（等效batch_size=16）
LEARNING_RATE = 5e-5  # Adafactor优化器适用该学习率
MAX_SEQ_LEN = 128  # 缩短序列长度以适应显存限制

# LoRA微调参数
LORA_RANK = 8  # LoRA矩阵秩
LORA_ALPHA = 32  # LoRA缩放系数
LORA_DROPOUT = 0.05  # 防止过拟合

# 系统优化参数
USE_GRADIENT_CHECKPOINTING = True  # 梯度检查点技术
DYNAMIC_PADDING = True  # Windows系统需启用动态填充

# 联邦学习基础配置
CLIENTS = 2
ROUNDS = 5
EPOCHS = 5
DUPLICATE_RATE = 0.3
TEST_RATIO = 0.2
SEED = 123

# 数据集配置
DATASET = "Sonnets"  # Haiku, Jokes, Rotten, IMDB, Shakespeare, Sonnets, Poetry

# 特殊token设置（保持与distilgpt2兼容）
EOS_TOKEN = "<|endoftext|>"
BOS_TOKEN = "<|startoftext|>"
PAD_TOKEN = "<|pad|>"

# EP-MPD去重配置
USE_EPMPD = True
TYPE = 1  # Type can be 1 or 2

# 路径配置
CACHE_PATH = "/scratch/vad5173"
MODEL_PATH = "trained_models"
MODEL_CACHE = "models_cache"