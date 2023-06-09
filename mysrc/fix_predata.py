import pandas as pd
from tqdm import tqdm
import ijson
from datetime import datetime
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
import json

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# FIXME change path
path = r"/data1/botdet/datasets/Twibot-20/"
properties_number = 15
properties_list = ['created_at', 'description', 'entities', 'location',
                   'pinned_tweet_id', 'profile_image_url', 'protected',
                   'url', 'username', 'verified', 'withheld',
                   {'public_metrics': ['followers_count', 'following_count', 'tweet_count', 'listed_count']}]


# 输出：{edge_type1:[(node1,node2),()],  }  str|str|str
def edge_build():
    split = pd.read_csv(path + 'split.csv', index_col='id')
    train = split.groupby('split').groups['train']
    train_split_id = train.tolist()        # 存储用于训练的用户id

    chunk_edges = pd.read_csv(path + 'edge.csv', chunksize=10000)   # chunksize分批处理
    edge_type_id_id = dict()
    for edges in tqdm(chunk_edges, desc='edge building'):
        for _, edge in edges.iterrows():
            if edge['source_id'] not in train_split_id:
                continue
            if edge['relation'] not in edge_type_id_id.keys():
                edge_type_id_id[edge['relation']] = list()
            edge_type_id_id[edge['relation']].append((edge['source_id'], edge['target_id']))

    with open('edge.json', 'w') as f:
        json.dump(edge_type_id_id, f, indent=4)

    return edge_type_id_id


def test_edge_build(data_type='test'):
    assert data_type in ['test', 'valid'], 'data type error'
    split = pd.read_csv(path + 'split.csv')
    test_split_id = split.groupby('split').groups[data_type].tolist()

    # TODO 添加label判断，分为正反（通过负采样）
    chunk_edges = pd.read_csv(path + 'edge.csv', chunksize=10000)  # 存储用于测试的用户id
    true_edge_type_id_id = dict()
    false_edge_type_id_id = dict()
    for edges in tqdm(chunk_edges):
        for _, edge in edges.iterrows():
            if edge['source_id'] not in test_split_id:
                continue
            if edge['relation'] not in true_edge_type_id_id.keys():
                true_edge_type_id_id[edge['relation']] = list()
            true_edge_type_id_id[edge['relation']].append((edge['source_id'], edge['target_id']))

    with open(data_type + '_edge', 'w') as fp:
        json.dump(true_edge_type_id_id, fp, indent=4)

    return true_edge_type_id_id


# 输入：账号创建时间
# 输出：活跃时间（从创建到2020/9/28的时间）
def calc_activate_days(created_at):
    created_at = created_at.strip()
    create_date = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
    crawl_date = datetime.strptime('2020 09 28 +0000', '%Y %m %d %z')
    delta_date = crawl_date - create_date
    return delta_date.days


# 输入：用户属性列表、属性个数、推文向量特征维度
# 输出：{user_id1:[property1, property2...],...}, 各个属性均已经归一化，属性列表长度填充至与text向量同宽
def users_feature(properties_list, properties_number, feature_dim):
    with open(path + 'node.json') as file:
        users = ijson.items(file, 'item')
        users_feature_dict = dict()
        flag = 0
        for user in tqdm(users, desc='loading user properties'):
            flag += 1
            if flag >= 8000000:
                break
            if user['id'].find('u') == -1:      # 寻找用户节点
                continue
            users_feature_dict[user['id']] = list()
            for property in properties_list:
                if isinstance(property, dict):
                    for count_property in property["public_metrics"]:
                        if user["public_metrics"][count_property] is None:
                            users_feature_dict[user['id']].append(0)
                        else:
                            users_feature_dict[user['id']].append(user["public_metrics"][count_property])
                elif user[property] is None:          # 属性为空则补0
                    users_feature_dict[user['id']].append(0)
                elif property in ['withheld', 'url', 'profile_image_url',       # bool属性值，非None就输入1
                              'pinned_tweet_id', 'entities', 'location']:
                    users_feature_dict[user['id']].append(1)
                elif property in ['verified', 'protected']:
                    users_feature_dict[user['id']].append(int(user[property] == 'True'))
                elif property in ['description', 'username']:
                    users_feature_dict[user['id']].append(len(user[property].strip()))
                elif property in ['created_at']:
                    users_feature_dict[user['id']].append(calc_activate_days(user[property].strip()))
            # 属性值数目小于规定值时报错
            assert len(users_feature_dict[user['id']]) == properties_number, \
                'user:{}, properties_number:{} < {}'.format(user['id'], len(users_feature_dict[user['id']]), properties_number)

    print('user properties z-score...')     # 数据z-score归一化
    properties = []
    users_id = list(users_feature_dict.keys())
    for user_id in users_id:
        properties.append(users_feature_dict[user_id])

    properties = np.array(properties)
    for column in range(properties.shape[1]):
        mean = np.mean(properties[:, column])  # 求平均值
        std = np.std(properties[:, column])  # 求标准差
        if std == 0:
            properties[:, column] = mean
        else:
            properties[:, column] = (properties[:, column] - mean) / std  # z-score归一化

    properties = np.pad(properties, ((0, 0), (0, feature_dim - properties_number)), mode='constant', constant_values=0)    # 用0扩充至与text特征向量同宽
    final_properties = properties.tolist()
    for index, user_id in enumerate(users_id):
        users_feature_dict[user_id] = final_properties[index]

    return users_feature_dict


# 输出：{text_id1: text1_feature, text_id2: text2_feature ...}
def texts_feature():
    pretrained_weights = 'bert-base-uncased'
    model = BertModel.from_pretrained(pretrained_weights)
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights, use_fast=True)
    tweet_id_library = []
    tweet_library = []
    with open(path + 'fixed_node.json') as file:
        texts = ijson.items(file, 'item')
        for text in tqdm(texts, desc='loading text properties'):
            if not text['id'].find('t') == -1:      # 推文对象
                if text['text'] is None:
                    continue
                tweet_id_library.append(text['id'])
                tweet_library.append(text['text'])
            else:
                continue
    convert_length = 100
    convert_step = len(tweet_library) // convert_length + 1
    front = 0
    back = convert_length - 1
    text_feature_dim = 0
    with open("result.json", 'w') as f:
        for i in tqdm(range(convert_step)):
            input_tensor = tokenizer(tweet_library[front:back], max_length=32, padding='max_length',
                                     return_tensors="pt", truncation=True)  # 编码推文
            tweet_rep = model(input_tensor['input_ids']).pooler_output
            text_feature_dim = tweet_rep.shape[1]
            texts_feature_dict = dict(zip(tweet_id_library[i], tweet_rep.tolist()))
            json.dump(texts_feature_dict, f)
            front += convert_length
            back = min(back + convert_length, len(tweet_library) + 1)
    with open("result.json", 'r') as f:
        final_texts_feature_dict = json.load(f)   # 获取所有推文表示
    return final_texts_feature_dict, text_feature_dim

# 输出：user_label {user_id:label, ...}
def get_user_label():
    data = pd.read_csv(path + "label.csv", header=0)
    user_label = {}
    for _, user in data.iterrows():
        user_label[user['id']] = int(user['label'] == 'bot')

    with open('user_label.json', 'w') as f:
        json.dump(user_label, f, indent=4)


if __name__ == '__main__':
    """
    edge_type_id_id = edge_build()
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    texts_feature_dict, feature_dim = texts_feature()
    users_feature_dict = users_feature(properties_list, properties_number, feature_dim)
    feature_dict = dict(**users_feature_dict, **texts_feature_dict)     # 合并字典

    with open('node_attribute.json', 'w') as fp:
        json.dump(feature_dict, fp, indent=4)

    get_user_label()
