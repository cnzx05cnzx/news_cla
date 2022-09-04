from torchdata.datapipes.iter import FileOpener, IterableWrapper, FileLister
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
import pandas as pd
import pkuseg
import torch
from torchtext.vocab import build_vocab_from_iterator
from lstm import LSTM
from torch.utils.data import DataLoader
from torchtext.transforms import Truncate, PadTransform


def tokenizer(text):
    return seg.cut(text)


def yield_tokens(data_iter):
    for data in data_iter:
        yield tokenizer(data['content'])


def build_datapipes(path):
    def row_processer(row):
        return {"content": row[0], "label": row[1]}

    datapipe = dp.iter.FileLister(path)
    datapipe = dp.iter.FileOpener(datapipe, encoding='utf-8')
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)

    datapipe = datapipe.shuffle()
    datapipe = datapipe.map(row_processer)
    return datapipe


def collate_batch(batch):
    text_list, label_list = [], []
    truncate = Truncate(max_seq_len=20)  # 截断
    pad = PadTransform(max_length=20, pad_value=vocab['<pad>'])
    for ba in batch:
        _text, _label = ba['content'], ba['label']

        label_list.append(label_pipeline(_label))
        text = text_pipeline(_text)
        text = truncate(text)
        text = torch.tensor(text, dtype=torch.int64)
        text = pad(text)
        # print(text)
        text_list.append(text)

    label_list = torch.tensor(label_list, dtype=torch.int64)

    text_list = torch.vstack(text_list)
    return label_list.to('cpu'), text_list.to('cpu')


if __name__ == '__main__':
    seg = pkuseg.pkuseg()
    train_iter = build_datapipes('./data/news_test.csv')

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    dl = DataLoader(train_iter, batch_size=4, shuffle=True, collate_fn=collate_batch)
    for step, d in enumerate(dl):
        print(d)
