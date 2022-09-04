import torch
import time
import pandas as pd
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from utils import BertConfig
import torch.nn.functional as F


# 进行token,预处理
def preprocessing_for_bert(text):
    input_ids = []
    attention_masks = []

    for sent in text:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # 预处理语句
            add_special_tokens=True,  # 加 [CLS] 和 [SEP]
            max_length=MAX_LEN,  # 截断或者填充的最大长度
            padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
            return_attention_mask=True,  # 返回 attention mask
            truncation=True
        )

        # 把输出加到列表里面
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # 把list转换为tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


class BertClassifier(nn.Module):
    def __init__(self, out_dim, res):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_choice, return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(768, out_dim, batch_first=True, dropout=0.5, bidirectional=True, num_layers=2)
        self.linear = nn.Sequential(
            nn.Linear(2 * out_dim, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, res)
        )

    def forward(self, input_ids, attention_mask):
        # 开始搭建整个网络了
        # 输入
        enbedding, pooled_output = self.bert(input_ids=input_ids,
                                             attention_mask=attention_mask)

        # outputs = self.drop(pooled_output)
        out, (h, c) = self.lstm(enbedding)
        out = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        out = self.linear(out)

        return out


def initialize_model():
    """
    初始化我们的bert，优化器还有学习率，epochs就是训练次数
    """
    # 初始化我们的Bert分类器
    bert_classifier = BertClassifier(128, 5)
    # 用GPU运算
    bert_classifier.to(device)
    # 创建优化器
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=2e-5,  # 默认学习率
                      )
    criterion = F.cross_entropy  # 交叉熵
    return bert_classifier, optimizer, criterion


def train(model, train_dataloader, valid_dataloader, epochs, path):
    best_loss = 1000
    best_acc = 0
    for epoch_i in range(epochs):
        print("epoch %d" % (epoch_i + 1))

        time_begin = time.time()
        train_loss = []

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)
            loss = criterion(logits, b_labels)
            train_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        time_elapsed = time.time() - time_begin
        avg_train_loss = sum(train_loss) / len(train_loss)
        print("训练集 Loss: %.2f 时间: %.2f" % (avg_train_loss, time_elapsed))
        # ---------------------------------------验证------------------------------
        model.eval()
        valid_accuracy = []
        valid_loss = []
        best_acc = 0.0
        cnt, stop = 0, 10
        time_begin = time.time()

        for batch in valid_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            loss = criterion(logits, b_labels.long())
            valid_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()

            accuracy = (preds == b_labels).cpu().numpy().mean() * 100

            valid_accuracy.append(accuracy)

        time_elapsed = time.time() - time_begin
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_accuracy = sum(valid_accuracy) / len(valid_accuracy)
        print("验证集 Loss: %.2f 验证集 Acc: %.2f 时间: %.2f" % (valid_loss, valid_accuracy, time_elapsed))

        if best_loss > valid_loss:
            best_loss = valid_loss
            best_acc = valid_accuracy
            torch.save(model.state_dict(), path)  # save entire net
            print('保存loss最小模型')

    print("验证集最高准确率为%.2f" % best_acc)


def test(Config, model, test_iter):
    # 测试-------------------------------
    model.load_state_dict(torch.load(Config.save_path))
    model.eval()
    t_a = time.time()
    total_acc, total_loss = 0, 0
    for i, batch in enumerate(test_iter):
        with torch.no_grad():

            model.zero_grad()
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            outputs = model(b_input_ids, b_attn_mask)
            pos = b_labels.to(Config.device)
            loss = criterion(outputs, pos)

        true = pos.data.cpu()
        predict = torch.max(outputs.data, 1)[1].cpu()
        total_loss += float(loss.item())
        total_acc += torch.eq(predict, true).sum().float().item()

    total_acc = total_acc / len(test_iter.dataset)
    total_loss = total_loss / len(test_iter)

    t_b = time.time()
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%},  Time: {2:>7.2}'
    print(msg.format(total_loss, total_acc, t_b - t_a))


config = BertConfig()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(config.seed)

model_choice = './hfl/chinese-bert-wwm'

tokenizer = BertTokenizer.from_pretrained(model_choice)

MAX_LEN = 35

data_train = pd.read_csv('./data/news_train.csv')
x_train, y_train = data_train['comment'].to_numpy(), data_train['pos'].to_numpy()

data_valid = pd.read_csv('./data/news_valid.csv')
x_valid, y_valid = data_valid['comment'].to_numpy(), data_valid['pos'].to_numpy()

data_test = pd.read_csv('./data/news_test.csv')
x_test, y_test = data_test['comment'].to_numpy(), data_test['pos'].to_numpy()

train_inputs, train_masks = preprocessing_for_bert(x_train)
valid_inputs, valid_masks = preprocessing_for_bert(x_valid)
test_inputs, test_masks = preprocessing_for_bert(x_test)
train_labels = torch.tensor(y_train)
valid_labels = torch.tensor(y_valid)
test_labels = torch.tensor(y_test)

# 创建DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=config.batch_size)
valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=config.batch_size)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=config.batch_size)

net, optimizer, criterion = initialize_model()
print("Start training and validating:")
train(net, train_dataloader, valid_dataloader, config.num_epochs, config.save_path)
print("Start testing:")
test(config, net, test_dataloader)
