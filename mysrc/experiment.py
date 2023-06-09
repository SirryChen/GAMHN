"""
import torch
import numpy as np
print(torch.cuda.is_available())

from transformers import pipeline

p_dim = 3
ppp = np.array([[1,2,3],[4,5,6]])

feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=3,
                           padding=True,
                           truncation=True, max_length=50, add_special_tokens=True)

text = 'I am sirry chen'
result = torch.tensor(feature_extract(text))
total_word_tensor = None
for k, each_word_tensor in enumerate(result[0]):
    if k == 0:
        total_word_tensor = each_word_tensor
    else:
        total_word_tensor += each_word_tensor
print(result.shape[1])
total_word_tensor /= result.shape[1]
print(total_word_tensor.shape)


import torch
from transformers import BertModel, BertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_weights = 'bert-base-uncased'
model = BertModel.from_pretrained(pretrained_weights)
tokenizer = BertTokenizer.from_pretrained(pretrained_weights, use_fast=True)
text = ["Here is some text to encode", "I am a student from chengdu", "I want to go home now"]
inputs = tokenizer(text,
                   max_length=64,
                   padding='max_length',
                   return_tensors="pt",
                   truncation=True)  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
print(inputs['input_ids'])
inputs.to(device)
model.to(device)
outputs = model(inputs['input_ids'])
print(outputs.pooler_output)


# print(outputs.last_hidden_state.shape)
# print(outputs.pooler_output.shape)
# print(outputs.last_hidden_state[0].shape)
# print(outputs.pooler_output[0].shape)


import json

# 读取 JSON 文件并打印
with open('/data1/botdet/datasets/Twibot-20/node.json', 'r') as f:
    data = f.read()
print("原始 JSON 数据：")
# print(data)
try:
    fixed_json = json.dumps(data, indent=4)
    json.loads(fixed_json)

except ValueError as e:
    print("Error: 生成修正后JSON失败！ {}".format(e))

else:

    # 输出修正后JSON数据到控制台以及保存到新文件中
    print("\n修改后 JSON 数据：")
    # print(fixed_json)

    location = fixed_json.find("t7869759")
    final_str = fixed_json[location:]
    with open('/data1/botdet/datasets/Twibot-20/mini_dataset/fixed_example.txt', 'w') as f:
        f.write(final_str)
"""

import pandas as pd
with open('/data1/botdet/datasets/Twibot-20/node.json', 'r') as f:
    data = f.read()

# data = data[:-1]
print(data[-200:])
fixed_data = data + r'\n"}]'
print(fixed_data[-200:])


with open('/data1/botdet/datasets/Twibot-20/fixed_node.json', 'w') as f:
    f.write(fixed_data)

# test
pd.read_json('/data1/botdet/datasets/Twibot-20/fixed_node.json')
