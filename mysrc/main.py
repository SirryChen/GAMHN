import math
import json
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import functional as F
from numpy import random
from torch.nn.parameter import Parameter

from utils import *


# 输出一个batch的信息  源节点列表，目标节点列表，关系列表，每个源节点的所有邻居列表
def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size   # 一共n_batches个batch，向下取整，所以要加一个batch_size

    for idx in range(n_batches):            # 第idx个batch
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])       # [index_i,...]
            y.append(pairs[index][1])       # [index_j,...]
            t.append(pairs[index][2])       # [layer_id,...] 其实是edge_type_i, index_i与index_j节点间边的关系集合
            neigh.append(neighbors[pairs[index][0]])    # .append([ edge_type1[node_index1,node_index2...], edge_type2[ ],...  ])
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)    # 返回一个batch_size


class GATNEModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, feature_dim
    ):
        super(GATNEModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a


        self.feature_dim = feature_dim
        self.embed_trans = Parameter(torch.FloatTensor(feature_dim, embedding_size))
        self.u_embed_trans = Parameter(torch.FloatTensor(edge_type_count, feature_dim, embedding_u_size))

        self.trans_weights = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size))
        self.trans_weights_s1 = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, dim_a))
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))

        self.reset_parameters()     # 初始化参数列表

    def reset_parameters(self):         # 参数初始化，uniform_([-1,1]均匀分布),  normal_(均值=0，标准差=std 的正态分布),
        self.embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.u_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    # 输入：
    # train_inputs:[node_index1,node_index2...]  (1*batch_size) 源节点集合
    # train_types:[layer_id1,layer_id2...] (1*batch_size) 源节点与目标节点间关系集合
    # node_neigh:[ [edge_type1[node_index1,node_index2...], edge_type2[ ],...  ],  ] (batch_size*edge_type_count*neighbor_sample)  batch_size个源节点对应的所有关系下的节点集合
    def forward(self, train_inputs, train_types, node_neigh, features):
        # 节点特征属性嵌入  (batch_size*feature_dim) * (feature_dim*embedding_size)->(batch_size*embedding_size)
        node_embed = torch.mm(features[train_inputs], self.embed_trans)
        # (batch_size*edge_type_count*neighbor_sample*feature_dim) * (edge_type_count*embedding_u_size*embedding_size)
        node_embed_neighbors = torch.einsum('bijk,akm->bijam', self.features[node_neigh], self.u_embed_trans)
        node_embed_tmp = torch.diagonal(node_embed_neighbors, dim1=1, dim2=3).permute(0, 3, 1, 2)
        node_type_embed = torch.sum(node_embed_tmp, dim=2)

        trans_w = self.trans_weights[train_types]
        trans_w_s1 = self.trans_weights_s1[train_types]
        trans_w_s2 = self.trans_weights_s2[train_types]

        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)

        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed      # 节点关于关系r的表示 (batch_size*dimensions)


class Classifier(nn.Module):
    def __init__(self, in_size, head_num, out_size=1, hidden_size=128):
        super(Classifier, self).__init__()
        self.head_num = head_num
        self.semantic_attention_layers = nn.ModuleList()
        # multi-head attention
        for i in range(head_num):
            self.semantic_attention_layers.append(
                nn.Sequential(
                    nn.Linear(in_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, out_size, bias=False)
                )
            )

    def forward(self, user_relation_rep):
        output = None
        for i in range(self.head_num):
            relation_weight = self.semantic_attention_layers[i](user_relation_rep).mean(0)
            norm_relation_weight = torch.softmax(relation_weight, dim=0)
            norm_relation_weight = norm_relation_weight.expand((user_relation_rep.shape[0],) + norm_relation_weight.shape)
            if output is None:
                output = (norm_relation_weight * user_relation_rep).sum(1)
            else:
                output += (norm_relation_weight * user_relation_rep).sum(1)

        return output / self.head_num


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.Tensor(
                [(math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1) for k in range(num_nodes)]
            ), dim=0,)

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


# 输入：network_data = training_data_by_type{edge_type1:[(node1,node2),(node3,node4),...],...}
#      feature_dic {node_i:[node1_feature],node_j:[node2_feature],...}
def train_model(network_data, feature_dic):
    # vocab {node_i:class(节点出现的次数，节点index),...}
    # index2word [node_i,...] 按节点出现次数降序排列，列表索引与vocab中index一致
    # train_pairs [  (index1,index6,layer_id), (index2,index7,layer_id)... ] layer_id是边类型编号
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema, file_name, args.window_size, args.num_workers, args.walk_file)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)         # 节点个数
    edge_type_count = len(edge_types)   #节点类型数量
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # neighbors[ node_index1[ edge_type1[node_index, ],... ],  node_index2[ edge_type2[] ],...  ]
    # 每个节点在每种关系上的列表 增减为 固定长度neighbor_samples
    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)

    features = None
    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])    # 特征维度
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        features = torch.FloatTensor(features).to(device)       # 行数：node_index，一行为一个节点的特征向量

    model = GATNEModel(
        num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    )
    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)

    classifier = Classifier(in_size=embedding_size, head_num=1)
    classifier_loss = nn.CrossEntropyLoss()

    model.to(device)
    nsloss.to(device)
    classifier.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), 'lr': 1e-4},
         {"params": nsloss.parameters(), 'lr': 1e-4},
         {"params": classifier.parameters(), 'lr': 1e-3}],
    )

    best_score = 0
    test_score = (0.0, 0.0, 0.0)
    patience = 0
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        batches = get_batches(train_pairs, neighbors, batch_size)   # 源节点列表，目标节点列表，关系列表，每个源节点的所有邻居列表

        data_iter = tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0
        stop_representation_learning = False
        for i, data in enumerate(data_iter):
            optimizer.zero_grad()
            embs = model(data[0].to(device), data[2].to(device), data[3].to(device),)   # 源节点、关系、源节点所有邻居(neighbors中信息)
            if not stop_representation_learning:
                loss = nsloss(data[0].to(device), embs, data[1].to(device))
                loss.backward()

            data_label = [user_label[index2word[index]] for index in data[0]]   # 获取本batch用户标签
            data_label = torch.tensor(data_label, dtype=torch.long)
            predict = classifier(embs)
            class_loss = classifier_loss(predict, data_label)
            class_loss.backword()
            optimizer.step()

            avg_loss += loss.item()

            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))

        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)]).to(device)
            train_types = torch.tensor(list(range(edge_type_count))).to(device)
            node_neigh = torch.tensor(
                [neighbors[i] for _ in range(edge_type_count)]
            ).to(device)
            node_emb = model(train_inputs, train_types, node_neigh)
            for j in range(edge_type_count):
                final_model[edge_types[j]][index2word[i]] = (
                    node_emb[j].cpu().detach().numpy()
                )

        valid_aucs, valid_f1s, valid_prs = [], [], []
        test_aucs, test_f1s, test_prs = [], [], []
        for i in range(edge_type_count):
            if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    valid_true_data_by_edge[edge_types[i]],
                    valid_false_data_by_edge[edge_types[i]],
                )
                valid_aucs.append(tmp_auc)
                valid_f1s.append(tmp_f1)
                valid_prs.append(tmp_pr)

                tmp_auc, tmp_f1, tmp_pr = evaluate(
                    final_model[edge_types[i]],
                    testing_true_data_by_edge[edge_types[i]],
                    testing_false_data_by_edge[edge_types[i]],
                )
                test_aucs.append(tmp_auc)
                test_f1s.append(tmp_f1)
                test_prs.append(tmp_pr)
        print("representation learning:")
        print("\tvalid auc:", np.mean(valid_aucs))
        print("\tvalid pr:", np.mean(valid_prs))
        print("\tvalid f1:", np.mean(valid_f1s))

        average_auc = np.mean(test_aucs)
        average_f1 = np.mean(test_f1s)
        average_pr = np.mean(test_prs)

        cur_score = np.mean(valid_aucs)
        if cur_score > best_score:
            best_score = cur_score
            test_score = (average_auc, average_f1, average_pr)
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Representation Learning Early Stopping")
                stop_representation_learning = True

    return test_score


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input      # FIXME change the file path in utils.py
    print(args)
    # if args.features is not None:
    #     feature_dic = load_feature_data(args.features)
    # else:
    #     feature_dic = None

    # training_data_by_type = load_training_data(file_name + "/train.txt")
    # valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
    #     file_name + "/valid.txt"
    # )
    # testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
    #     file_name + "/test.txt"
    # )

    training_data_by_type = json.load(file_name + 'egde.json')
    feature_dic = json.load(file_name + 'node_attribute.json')
    user_label = json.load(file_name + 'user_label.json')


    average_auc, average_f1, average_pr = train_model(training_data_by_type, feature_dic)

    print("Overall ROC-AUC:", average_auc)
    print("Overall PR-AUC", average_pr)
    print("Overall F1:", average_f1)
