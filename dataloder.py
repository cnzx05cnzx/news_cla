import random
import time
import torch
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def seed_init(seed):
    random.seed(seed)
    np.random.seed(seed)


def write_csv(path, header, datax, datay):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for x, y in zip(datax, datay):
            writer.writerow([x, y])


def make_file(path):
    df = pd.read_csv(path)
    df['pos'] = df['pos'].astype('str')

    x_train, x_test, y_train, y_test = train_test_split(df['comment'], df['pos'], test_size=0.4)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5)
    header = ['comment', 'pos']
    write_csv('./data/news_train.csv', header, x_train, y_train)
    write_csv('./data/news_valid.csv', header, x_valid, y_valid)
    write_csv('./data/news_test.csv', header, x_test, y_test)


if __name__ == '__main__':
    seed_init(721)
    make_file('./data/news.csv')
