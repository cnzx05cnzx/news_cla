import torch
import time
import pandas as pd
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

SEED = 721
batch_size = 32
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)


data = pd.read_csv('./data/news.csv')

x = data['comment'].to_numpy()
y = data['pos'].to_numpy()

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=SEED)

model_choice = './hfl/chinese-bert-wwm'

tokenizer = BertTokenizer.from_pretrained(model_choice)


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
MAX_LEN = 35
batch_size = 32

# 在train，valid上运行 preprocessing_for_bert 转化为指定输入格式
train_inputs, train_masks = preprocessing_for_bert(x_train)
valid_inputs, valid_masks = preprocessing_for_bert(x_valid)
train_labels = torch.tensor(y_train)
valid_labels = torch.tensor(y_valid)

# 创建DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=batch_size)


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

    return bert_classifier, optimizer


# 实体化loss function
criterion = nn.CrossEntropyLoss()  # 交叉熵


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
            print('保存最好效果模型')

        print("\n")
    print("验证集最高准确率为%.2f" % best_acc)


t_epoch = 10
net, optimizer = initialize_model()
save_path = './model/try_bert_params.pkl'
print("Start training and validating:")
train(net, train_dataloader, valid_dataloader, t_epoch, save_path)
