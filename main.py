import random

import numpy as np
import pandas as pd
from collections import Counter
import pkuseg
import torch
from torch import nn
from torchtext.legacy.data import BucketIterator, Field, TabularDataset
from lstm import LSTM

SEED = 721
batch_size = 32
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)

seg = pkuseg.pkuseg()


def tokenizer(text):
    return seg.cut(text)


TEXT = Field(sequential=True, tokenize=tokenizer, fix_length=35)
POS = Field(sequential=False, use_vocab=False)

FIELD = [('label', None), ('content', TEXT), ('pos', POS)]

df = TabularDataset(
    path='./data/news.csv', format='csv',
    fields=FIELD, skip_header=True)

TEXT.build_vocab(df, min_freq=3, vectors='glove.6B.50d')

train, valid = df.split(split_ratio=0.7, random_state=random.seed(SEED))

train_iter, valid_iter = BucketIterator.splits(
    (train, valid),
    batch_sizes=(batch_size, batch_size),
    device=device,
    sort_key=lambda x: len(x.content),
    sort_within_batch=False,
    repeat=False
)

model = LSTM(len(TEXT.vocab), 64, 128).to(device)

import torch.optim as optim
import torch.nn.functional as F

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = F.cross_entropy
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# -----------------------------------模型训练--------------------------------------
epochs = 100
stop = 20
cnt = 0
best_valid_acc = float('-inf')
model_save_path = './model/torchtext.pkl'

for epoch in range(epochs):
    loss_one_epoch = 0.0
    correct_num = 0.0
    total_num = 0.0

    for i, batch in enumerate(train_iter):
        model.train()
        pos, content = batch.pos, batch.content
        # 进行forward()、backward()、更新权重
        optimizer.zero_grad()
        pred = model(content)
        loss = criterion(pred, pos)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        total_num += pos.size(0)
        # 预测有多少个标签是预测中的，并加总
        correct_num += (torch.argmax(pred, dim=1) == pos).sum().float().item()
        loss_one_epoch += loss.item()

    loss_avg = loss_one_epoch / len(train_iter)

    print("Train: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".
          format(epoch + 1, epochs, loss_avg, correct_num / total_num))

    # ---------------------------------------验证------------------------------
    loss_one_epoch = 0.0
    total_num = 0.0
    correct_num = 0.0

    model.eval()
    for i, batch in enumerate(valid_iter):
        pos, content = batch.pos, batch.content
        pred = model(content)
        pred.detach()
        # 计算loss

        # 统计预测信息
        total_num += pos.size(0)
        # 预测有多少个标签是预测中的，并加总
        correct_num += (torch.argmax(pred, dim=1) == pos).sum().float().item()

    # 学习率调整
    scheduler.step()

    print('valid Acc:{:.2%}'.format(correct_num / total_num))

    # 每个epoch计算一下验证集准确率如果模型效果变好，保存模型
    if (correct_num / total_num) > best_valid_acc:
        print("超过最好模型,保存")
        best_valid_acc = (correct_num / total_num)
        torch.save(model.state_dict(), model_save_path)
        cnt = 0
    else:
        cnt = cnt + 1
        if cnt > stop:
            # 早停机制
            print("模型基本无变化，停止训练")
            print("训练集最高准确率为%.2f" % best_valid_acc)
            break
