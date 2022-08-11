import pandas as pd
import pkuseg
import torch
from torchtext.vocab import build_vocab_from_iterator
from lstm import LSTM
from torch.utils.data import DataLoader, Dataset
from torchtext.transforms import Truncate, PadTransform
from utils import LstmConfig
import torch.optim as optim
import torch.nn.functional as F


class TextCNNDataSet(Dataset):
    def __init__(self, data, data_targets):
        self.content = data
        self.pos = data_targets

    def __getitem__(self, index):
        return self.content[index], self.pos[index]

    def __len__(self):
        return len(self.pos)


def tokenizer(text):
    return seg.cut(text)


def yield_tokens(data_iter):
    for _, text in data_iter.iterrows():
        yield tokenizer(text['comment'])


def collate_batch(batch):
    label_list, text_list = [], []
    truncate = Truncate(max_seq_len=20)  # 截断
    pad = PadTransform(max_length=20, pad_value=vocab['<pad>'])
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        text = text_pipeline(_text)
        text = truncate(text)
        text = torch.tensor(text, dtype=torch.int64)
        text = pad(text)
        text_list.append(text)

    label_list = torch.tensor(label_list, dtype=torch.int64)

    text_list = torch.vstack(text_list)
    return label_list.to(config.device), text_list.to(config.device)


def train(train_fig):
    epochs = train_fig.num_epochs
    stop = train_fig.early_stop
    cnt = 0
    best_valid_acc = float('-inf')
    model_save_path = train_fig.save_path

    for epoch in range(epochs):
        loss_one_epoch = 0.0
        correct_num = 0.0
        total_num = 0.0
        model.train()
        for i, batch in enumerate(train_loader):
            pos, content = batch[0], batch[1]
            # 进行forward()、backward()、更新权重
            optimizer.zero_grad()
            pred = model(content)
            loss = criterion(pred, pos)
            loss.backward()
            optimizer.step()

            total_num += pos.size(0)
            correct_num += (torch.argmax(pred, dim=1) == pos).sum().float().item()
            loss_one_epoch += loss.item()

        loss_avg = loss_one_epoch / len(train_iter)

        print("Train: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".
              format(epoch + 1, epochs, loss_avg, correct_num / total_num))

        # ---------------------------------------验证------------------------------
        total_num = 0.0
        correct_num = 0.0

        model.eval()
        for i, batch in enumerate(valid_loader):
            pos, content = batch[0], batch[1]
            pred = model(content)
            pred.detach()

            total_num += pos.size(0)
            correct_num += (torch.argmax(pred, dim=1) == pos).sum().float().item()

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
                print("验证集最高准确率为%.2f" % best_valid_acc)
                break


if __name__ == '__main__':
    config = LstmConfig()
    torch.manual_seed(config.seed)
    seg = pkuseg.pkuseg()

    train_iter = pd.read_csv('./data/news_train.csv')
    valid_iter = pd.read_csv('./data/news_valid.csv')

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), min_freq=5, specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    train_iter = TextCNNDataSet(list(train_iter['comment']), list(train_iter['pos']))
    valid_iter = TextCNNDataSet(list(valid_iter['comment']), list(valid_iter['pos']))

    train_loader = DataLoader(train_iter, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_iter, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    model = LSTM(len(vocab), 64, 128).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = F.cross_entropy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train(config)
