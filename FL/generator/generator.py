import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载已保存的模型和分词器
model_path = "../distilgpt2-sonnets"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelWithLMHead.from_pretrained(model_path)

# 创建文本生成管道
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # 自动检测GPU
)

# 使用示例
prompt = "Shall I compare thee to a summer's day?"
prompt1 = "Desire for growth in loveliest of beings"
generated_texts = text_generator(
    prompt1,
    max_length=300,  # 最大生成长度
    temperature=0.85,  # 控制创造性 (0-1，越大越随机)
    top_k=50,         # 限制候选词数量
    num_return_sequences=3  # 生成3个不同版本
)

# 输出结果
for i, seq in enumerate(generated_texts):
    print(f"生成结果 {i+1}:\n{seq['generated_text']}\n")
