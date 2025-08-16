import array
import gzip
import json
import os
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import pickle

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


np.random.seed(123)

# Define base path as a variable
base_path = '/data/ZZC/workspace/'

datasets = 'Toys'

folder = f'{base_path}DiffMM-main-SentimentScore/Datasets/{datasets}/'
name = f'{datasets}'
core = 5
bert_path = f'{base_path}DiffMM-main-SentimentScore/Datasets/sentence-bert/stsb-roberta-large/'
bert_model = SentenceTransformer(bert_path)

if not os.path.exists(folder + '%d-core'%core):
    os.makedirs(folder + '%d-core'%core)


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

print("----------parse metadata----------")
if not os.path.exists(folder + "meta-data/meta.json"):
    with open(folder + "meta-data/meta.json", 'w') as f:
        for l in parse(folder + 'meta-data/' + "meta_%s.json.gz"%(name)):
            f.write(l+'\n')

print("----------parse data----------")
if not os.path.exists(folder + "meta-data/%d-core.json" % core):
    with open(folder + "meta-data/%d-core.json" % core, 'w') as f:
        for l in parse(folder + 'meta-data/' + "reviews_%s_%d.json.gz"%(name, core)):
            f.write(l+'\n')

print("----------load data----------")
jsons = []
for line in open(folder + "meta-data/%d-core.json" % core).readlines():
    jsons.append(json.loads(line))

print("----------Build dict----------")
items = set()
users = set()
for j in jsons:
    items.add(j['asin'])
    users.add(j['reviewerID'])
print("n_items:", len(items), "n_users:", len(users))

n_users = len(users)
n_items = len(items)

item2id = {}
with open(folder + '%d-core/item_list.txt'%core, 'w') as f:
    for i, item in enumerate(items):
        item2id[item] = i
        f.writelines(item+'\t'+str(i)+'\n')

user2id = {}
with open(folder + '%d-core/user_list.txt'%core, 'w') as f:
    for i, user in enumerate(users):
        user2id[user] = i
        f.writelines(user+'\t'+str(i)+'\n')

# 初始化一个空列表来存储情绪得分
all_sentiment_scores = []
# 初始化空评论计数器
empty_comment_count = 0

# 初始化VADER情感分析器
analyzer = SentimentIntensityAnalyzer()

ui = defaultdict(list)
for j in jsons:
    u_id = user2id[j['reviewerID']]
    i_id = item2id[j['asin']]
    comment = j['reviewText']
    sentiment_score = None
    
    # 检查评论是否为空
    if not comment.strip():
        empty_comment_count += 1
    else:
        # 使用VADER分析评论的情感
        sentiment_scores = analyzer.polarity_scores(comment)
        # 使用compound分数，将-1到1的范围转换为0到1
        sentiment_score = (sentiment_scores['compound'] + 1) / 2

    # 将情绪得分添加到列表中
    all_sentiment_scores.append(sentiment_score)
    ui[u_id].append([i_id, sentiment_score])

# 计算all_sentiment_scores的平均值，对ui和all_sentiment_scores中的none值进行更改
# 计算所有非None情感得分的平均值
average_score = sum(score for score in all_sentiment_scores if score is not None) / len([score for score in all_sentiment_scores if score is not None])

# 遍历ui字典，替换None值为平均值
for u_id, item_list in ui.items():
    for item in item_list:
        i_id, sentiment_score = item
        if sentiment_score is None:
            # 替换None值为平均值
            item[1] = average_score

print(f'average_score: {average_score}  empty_comment_count: {empty_comment_count}')
# 注意：上面的代码直接修改了ui字典中的列表项。如果你不希望修改原始数据，
# 你应该创建一个新的数据结构来存储替换后的值。

# average_score: 0.8398378811639535  empty_comment_count: 24 Clothing
# average_score: 0.7945318509618706  empty_comment_count: 1071 Electronics

with open(folder + '%d-core/user-item-dict.json'%core, 'w') as f:
    f.write(json.dumps(ui))


print("----------Split Data----------")
train_json = {}
val_json = {}
test_json = {}
for u, items in ui.items():
    if len(items) < 10:
        testval = np.random.choice(len(items), 2, replace=False)
    else:
        testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

    test = testval[:len(testval)//2]
    val = testval[len(testval)//2:]
    train = [i for i in list(range(len(items))) if i not in testval]
    train_json[u] = [items[idx] for idx in train]
    val_json[u] = [items[idx] for idx in val.tolist()]
    test_json[u] = [items[idx] for idx in test.tolist()]

with open(folder + '%d-core/train.json'%core, 'w') as f:
    json.dump(train_json, f)
with open(folder + '%d-core/val.json'%core, 'w') as f:
    json.dump(val_json, f)
with open(folder + '%d-core/test.json'%core, 'w') as f:
    json.dump(test_json, f)

print("----------Split Over----------")

print("----------创建交互矩阵----------")

train_file = f'{base_path}DiffMM-main-SentimentScore/Datasets/{datasets}/5-core/train.json'
train = json.load(open(train_file))
train_mat = np.zeros((n_users, n_items))


for uid, train_items in train.items():
    if len(train_items) == 0:
        continue
    uid = int(uid)
    for item, score in train_items:
        item = int(item)
        train_mat[uid, item] = score
        # train_mat[uid, item] = 1

# 获取矩阵的行数、列数和非零元素的坐标及值
row, col = np.nonzero(train_mat)  # 获取非零元素的行和列索引
data = train_mat[row, col]  # 获取非零元素的值

# 创建coo_matrix
sparse_matrix = coo_matrix((data, (row, col)), shape=train_mat.shape)

# 使用with语句打开文件，并自动处理文件的打开和关闭
with open(f'{base_path}DiffMM-main-SentimentScore/Datasets/{datasets}/trnMat.pkl', 'wb') as file:
    # 使用pickle.dump()函数将矩阵数据序列化后写入文件
    pickle.dump(sparse_matrix, file)

print(f"Matrix has been saved to trnMat.pkl")

# 处理验证集
val_file = f'{base_path}DiffMM-main-SentimentScore/Datasets/{datasets}/5-core/val.json'
val = json.load(open(val_file))
val_mat = np.zeros((n_users, n_items))

for uid, val_items in val.items():
    if len(val_items) == 0:
        continue
    uid = int(uid)
    for item, _ in val_items:  # 忽略score值
        item = int(item)
        val_mat[uid, item] = 1  # 直接赋值为1

# 创建验证集稀疏矩阵
val_row, val_col = np.nonzero(val_mat)
val_data = val_mat[val_row, val_col]
val_sparse = coo_matrix((val_data, (val_row, val_col)), shape=val_mat.shape)

with open(f'{base_path}DiffMM-main-SentimentScore/Datasets/{datasets}/valMat.pkl', 'wb') as f:
    pickle.dump(val_sparse, f)

# 处理测试集
test_file = f'{base_path}DiffMM-main-SentimentScore/Datasets/{datasets}/5-core/test.json'
test = json.load(open(test_file))
test_mat = np.zeros((n_users, n_items))

for uid, test_items in test.items():
    if len(test_items) == 0:
        continue
    uid = int(uid)
    for item, _ in test_items:  # 忽略score值
        item = int(item)
        test_mat[uid, item] = 1  # 直接赋值为1

# 创建测试集稀疏矩阵
test_row, test_col = np.nonzero(test_mat)
test_data = test_mat[test_row, test_col]
test_sparse = coo_matrix((test_data, (test_row, test_col)), shape=test_mat.shape)

with open(f'{base_path}DiffMM-main-SentimentScore/Datasets/{datasets}/tstMat.pkl', 'wb') as f:
    pickle.dump(test_sparse, f)

print("Validation and test matrices have been saved")

jsons = []
with open(folder + "meta-data/meta.json", 'r') as f:
    for line in f.readlines():
        jsons.append(json.loads(line))

print("----------Text Features----------")
raw_text = {}
for json in jsons:
    if json['asin'] in item2id:
        string = ' '
        if 'categories' in json:
            for cates in json['categories']:
                for cate in cates:
                    string += cate + ' '
        if 'title' in json:
            string += json['title']
        if 'brand' in json:
            string += json['brand']
        if 'description' in json:
            string += json['description']
        raw_text[item2id[json['asin']]] = string.replace('\n', ' ')
texts = []
with open(folder + '%d-core/raw_text.txt'%core, 'w') as f:
    for i in range(len(item2id)):
        f.write(raw_text[i] + '\n')
        texts.append(raw_text[i] + '\n')
sentence_embeddings = bert_model.encode(texts)
assert sentence_embeddings.shape[0] == len(item2id)
np.save(folder+'text_feat.npy', sentence_embeddings)


print("----------Image Features----------")
def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10).decode('UTF-8')
        if asin == '': break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()

data = readImageFeatures(folder + 'meta-data/' + "image_features_%s.b" % name)
feats = {}
avg = []
for d in data:
    if d[0] in item2id:
        feats[int(item2id[d[0]])] = d[1]
        avg.append(d[1])
avg = np.array(avg).mean(0).tolist()

ret = []
for i in range(len(item2id)):
    if i in feats:
        ret.append(feats[i])
    else:
        ret.append(avg)

assert len(ret) == len(item2id)
np.save(folder+'image_feat.npy', np.array(ret))

print("----------Over Over----------")

