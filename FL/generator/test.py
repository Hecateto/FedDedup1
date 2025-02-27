import numpy as np
import pandas as pd
import os

import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset  # 新增关键库

# 数据加载
data = pd.read_csv("../datasets/sonnets.csv")
texts = data['Variation Text'].astype(str).tolist()

# 创建 HuggingFace Dataset 对象
dataset = Dataset.from_dict({"text": texts})  # 必须包装成特定结构

# 分词处理
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # 为 GPT-2 设置填充标记



def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt"  # 返回 PyTorch 张量格式
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # 移除原始文本列
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 使用因果语言建模 (CLM)
    pad_to_multiple_of= 8  # 为了梯度累积，填充到8的倍数
)

# 模型加载
model = AutoModelWithLMHead.from_pretrained("distilgpt2",
                                            low_cpu_mem_usage=True,
                                            torch_dtype=torch.float16)  # 低内存模式

import math
from transformers import Trainer, TrainingArguments

# 新增：自定义评估指标计算函数
def compute_metrics(eval_pred):
    # 确保正确处理不同格式的 eval_pred
    try:
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred
    except Exception as e:
        raise ValueError(f"无法解析 eval_pred: {str(e)}")

    # 转换 numpy 数组为张量
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # # 验证张量维度
    # if logits.dim() == 3:  # (batch, seq_len, vocab_size)
    #     logits = logits[:, :-1, :]  # 对齐标签
    #     labels = labels[:, 1:]  # 预测下一个token
    max_seq_len = min(logits.size(1), labels.size(1) + 1)  # +1处理序列偏移
    logits = logits[:, :max_seq_len - 1, :]  # 安全截断
    labels = labels[:, 1:max_seq_len]  # 匹配截断

    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)

    # 重塑张量
    loss = torch.nn.functional.cross_entropy(
        logits.float(),
        labels.long(),
        # reduction="mean",
        ignore_index=-100
    )

    return {
        "perplexity": math.exp(loss.item()),
        "eval_loss": loss.item()
    }

# 修改训练参数配置
training_args = TrainingArguments(
    output_dir="../results",
    evaluation_strategy="steps",
    eval_steps=1000,
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=500,
    gradient_accumulation_steps=2,
    # 新增以下配置
    load_best_model_at_end=True,
    metric_for_best_model="perplexity",  # 使用困惑度作为模型选择依据
    greater_is_better=False,  # 困惑度越低越好

    fp16=False,  # 启用混合精度训练
    # fp16_full_eval=True,  # 全量混合精度评估
    gradient_checkpointing=True,  # 激活梯度检查点
    per_device_eval_batch_size=1,  # 评估批次设为1
    eval_accumulation_steps=4,  # 评估时梯度累积
    optim="adafactor"  # 使用内存优化型优化器
)

# 修改训练器初始化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # 绑定指标计算函数
)

from transformers import TrainerCallback

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = state.log_history[-1]
        if "eval_perplexity" in metrics:
            print(f"\n评估困惑度: {metrics['eval_perplexity']:.2f}")

# 添加回调到训练器
trainer.add_callback(PerplexityCallback())



# 执行训练和评估
train_result = trainer.train()
# eval_result = trainer.evaluate()

# 正确获取困惑度
# print(f"最终困惑度: {eval_result['perplexity']}")
# print(f"最终bleu: {eval_result['bleu']}")



from transformers import pipeline
# Instantiate a text generation pipeline
text_generator = pipeline("text-generation",
                          model=model,
                          tokenizer=tokenizer,
                          device=0 if torch.cuda.is_available() else -1)

# Generate text
prompt = "Shall I compare thee to a summer's day?"
generated_texts = text_generator(
    prompt,
    max_length=500,  # 最大生成长度
    temperature=0.85,  # 控制创造性 (0-1，越大越随机)
    top_k=50,         # 限制候选词数量
    top_p=0.90,       # 限制累积概率
    repetition_penalty=1.2,  # 控制重复度
    do_sample=True,   # 使用采样
    no_repeat_ngram_size=2,  # 避免重复2-gram
    num_return_sequences=3  # 生成3个不同版本
)
# 输出结果
for i, seq in enumerate(generated_texts):
    print(f"生成结果 {i+1}:\n{seq['generated_text']}\n")


# 保存模型
trainer.save_model("distilgpt2-sonnets")
# 保存分词器
tokenizer.save_pretrained("distilgpt2-sonnets")
# 保存训练参数
training_args.save("distilgpt2-sonnets")
