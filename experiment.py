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
"""
"""
import torch
from transformers import BertModel, BertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_weights = 'bert-base-uncased'
model = BertModel.from_pretrained(pretrained_weights)
tokenizer = BertTokenizer.from_pretrained(pretrained_weights, use_fast=True, hidden_size=128)
model.to(device)
tokenizer.to(device)
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
print(outputs.pooler_output.shape)

from transformers import DistilBertTokenizer, DistilBertModel
import torch

# 加载DistilBERT模型和tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# 定义要处理的文本数据
text = "DistilBERT is a lighter version of BERT, which uses fewer parameters and requires less computational resources."

# 使用tokenizer对文本进行编码处理
inputs = tokenizer(text, return_tensors='pt')

# 使用DistilBERT模型获取文本特征向量
outputs = model(**inputs)

# 获取特征向量的最后一层
last_hidden_state = outputs.last_hidden_state

# 获取特征向量的平均值
mean_pooling = torch.mean(last_hidden_state, dim=1)

# 输出特征向量的形状
print(mean_pooling.shape)

# 使用MobileBERT模型进行前向传递，得到…[omitted]




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


# import pandas as pd
# with open('/data1/botdet/datasets/Twibot-20/node.json', 'r') as f:
#     data = f.read()
#
# # data = data[:-1]
# print(data[-200:])
# fixed_data = data + r'\n"}]'
# print(fixed_data[-200:])
#
#
# with open('/data1/botdet/datasets/Twibot-20/fixed_node.json', 'w') as f:
#     f.write(fixed_data)
#
# # test
# pd.read_json('/data1/botdet/datasets/Twibot-20/fixed_node.json')


# import json
# a = {'1':[1,1],'2':[2,2],'3':[3,3]}
# key = list(a.keys())
#
# with open('experiment.json', 'w') as f:
#     json.dump(a, f, indent=4)
# with open('experiment.json', 'r') as f:
#     text = json.load(f)
#     print(text)

import tensorflow_hub as hub
import torch


module_path = "/data1/botdet/universal-sentence-encoder_4"
model = hub.load(module_path)
sentences = ["Hello world.", "How are you?"]
embeddings = model(sentences)
print(embeddings)
print(type(embeddings))
print(embeddings.shape)
np_embeddings = embeddings.numpy()
torch_embeddings = torch.from_numpy(np_embeddings)
print(type(torch_embeddings))
print(torch_embeddings.shape)


import ijson
import json
from tqdm import tqdm

path = r"/data1/botdet/datasets/Twibot-20/"
# with open(path + 'fixed_node.json') as file:
#     texts = ijson.items(file, 'item')
#     for text in texts:
#         if text['id'] == 't965250':
#             print(text['text'])
#             print('yes')
#             break

# with open('/data1/botdet/GAMHN-master/mysrc/' + 'node_attribute.json', 'r') as fp:
#     feature_dic = json.load(fp)      # 所有node的特征属性，不等长
#     feature_name_list = list(feature_dic.keys())
#     if 't965250' in feature_name_list:
#         print('yes')

tweet_id_library = []
tweet_library = []
with open(path + 'fixed_node.json') as file:
    texts = ijson.items(file, 'item')
    for text in tqdm(texts, desc='loading text properties'):
        if not text['id'].find('t') == -1:  # 推文对象
            if text['text'] is None:
                continue
            if text['id'] == 't1344357':
                print('yes')
                print(text['text'])
            tweet_id_library.append(text['id'])
            tweet_library.append(text['text'])
        else:
            continue

tweet_id_library = []
tweet_library = []
with open(path + 'fixed_node.json') as file:
    texts = ijson.items(file, 'item')
    for text in tqdm(texts, desc='loading text properties'):
        if not text['id'].find('t') == -1:  # 推文对象
            if text['text'] is None:
                continue
            tweet_id_library.append(text['id'])
            tweet_library.append(text['text'])
        else:
            continue
count = 0
convert_length = 1000
convert_step = len(tweet_library) // convert_length + 1
for i in tqdm(range(convert_step)):
    with open(path + "USE_predata/USE_temp_data{}.json".format(i), 'r') as f:
        temp_texts_feature_dict = json.load(f)
        for key in temp_texts_feature_dict.keys():
            count += 1
            if key == 't965250':
                print('yes')
                break
print(count)


import json
with open('node_attribute.json', 'r') as fp:
    node_attributes = json.load(fp)
    for node_id in node_attributes.keys():
        node_attribute_value = node_attributes[node_id]
        if len(node_attribute_value) != 64:
            print(node_id)
            print(len(node_attribute_value))


try:
    a = 0/0
except Exception as e:
    print(e)
    pass
print('i love nanjing ')


# import pickle
#
# with open('temp_data/one_batch.pickle', 'rb') as f:
#     index2word, train_pairs, neighbors, num_nodes, sub_feature_dic, neighbors_features = pickle.load(f)

import ijson
from tqdm import tqdm
import pandas as pd
import json


node_dict = {}
path = r"/data1/botdet/datasets/Twibot-20/"
with open(path + 'fixed_node.json') as file:
    texts = ijson.items(file, 'item')
    for text in tqdm(texts, desc='loading text properties'):
        if not text['id'].find('u') == -1:
            node_dict[text['id']] = text['id']

data = pd.read_csv(path + "label.csv", header=0)
user_label = {}
count = 0
flag = True
for _, user in data.iterrows():
    user_label[user['id']] = int(user['label'] == 'bot')
    count += 1

print('node count:{}'.format(len(list(node_dict.keys()))))
print('label count:{}'.format(count))


import pandas as pd
from tqdm import tqdm

path = r"/data1/botdet/datasets/Twibot-20/"
split = pd.read_csv(path + 'split.csv', index_col='id')
test_split_id = split.groupby('split').groups['train'].tolist()
print(test_split_id)

# print(len(test_split_id))
#
# count = 0
# edges = pd.read_csv(path + 'edge.csv')  # 存储用于测试的用户id
# for _, edge in tqdm(edges.iterrows()):
#     if edge['source_id'] in test_split_id:
#         print(edge['source_id'])
#         count += 1
#
# print(count)
"""

# import json
#
# with open('pre_data/node_attribute.json', 'r') as fp:
#     nodes = json.load(fp)
#     for node in nodes.keys():
#         if node.find('t') != -1:
#             print(nodes[node])

from sklearn.metrics import accuracy_score

a= [1,0,0,0,1]
# b=[0.8,0.9,0.4,0.1,0.2]
b=[1,1,0,0,1]
print(accuracy_score(a,b))