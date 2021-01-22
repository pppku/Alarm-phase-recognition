"""Prepare data for train and test

    author: jimmy li
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def read_csv(path):
    """Read csv file"""
    data = pd.read_csv(path)
    # data = data.drop_duplicates(subset=['B', 'E'], keep='first', inplace=False)
    # data = data.reset_index(drop=True)
    x = data['E']
    y = data['B']

    return x, y


def sklearn_tfidf(x):
    process_data = x.tolist()

    tv = CountVectorizer()
    word_frequency = tv.fit_transform(process_data)

    transform = TfidfTransformer()
    tfidf = transform.fit_transform(word_frequency)

    return tfidf.toarray()


def main():
    """Main function"""
    # read csv
    inside_x, inside_y = read_csv("inside-log.csv")
    dmz_x, dmz_y = read_csv("dmz-log.csv")

    # get train_set size
    inside_size = inside_x.shape[0]
    dmz_size = dmz_x.shape[0]

    # 合并两个dataframe 忽略原始编号
    all_data = pd.concat([inside_x, dmz_x], ignore_index=True)
    all_data = sklearn_tfidf(all_data)

    # 拆分TF-IDF后的数据
    inside_x = all_data[0:inside_size, :]
    dmz_x = all_data[inside_size:inside_size + dmz_size, :]

    # 合并x y
    inside = np.zeros((inside_size, inside_x[0].size + 1))
    dmz = np.zeros((dmz_size, dmz_x[0].size + 1))

    for i in range(inside_size):
        inside[i] = np.append(inside_x[i], inside_y[i])
    for i in range(dmz_size):
        dmz[i] = np.append(dmz_x[i], dmz_y[i])

    # 转换为npy格式
    np.save("./train/train_data.npy", inside)
    np.save("./test/test_data.npy", dmz)

    print("<prepare.py> Prepare Done! </prepare.py>")


if __name__ == "__main__":
    main()
    # x, y = read_csv("inside-log.csv")
    # print(x)
    # print(y)
