import pandas as pd
import numpy as np
import logging

from tokenization_kobert import KoBertTokenizer
from keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)


def read_data(args):
    train_data = pd.read_csv(args.train_file, sep='\t')
    test_data = pd.read_csv(args.test_file, sep='\t')

    return train_data, test_data


# def process(data):
    # print(tokenizer.cls_token)
    # texts = ["[CLS] " + str(text) + " [SEP]" for text in data['document']]
    # label = train_data['label'].values
    # return texts, label


def tokenize(data):
    # print(tokenizer.cls_token)
    texts = ["[CLS] " + str(text) + " [SEP]" for text in data['document']]
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    print(tokenizer.cls_token)
    print(111)
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=128, dtype="long", truncating="post", padding="post")
    # print(input_ids)


# def load(input_ids):
#     for seq in input_ids:
#         seq_mask = [int()]


# if __name__ == "__main__":
#     train_data, test_data = read_data()
#     train_texts, train_analysis = process(data=train_data)
#     tokenize(texts=train_texts)
