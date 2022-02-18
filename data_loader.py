import pandas as pd
import numpy as np
import logging
import json
import copy

from tokenization_kobert import KoBertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class Feature(object):
    """ features of data """
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        # self.mode = mode
        self.labels = labels


def read_data(args):
    train_data = pd.read_csv(args.train_file, sep='\t')
    test_data = pd.read_csv(args.test_file, sep='\t')

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)
    # 혹시 스탑워드 추가시 다음과 같이 관리
    # (data['description'].dropna()
    #  .apply(lambda x: [item for item in x if item not in stop_words]))
    return train_data, test_data


def process(data, max_seq):

    input_ids = []

    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    texts = data['document']
    #이 사이에 전처리해주는 함수 하나 추가해도 좋을 거 같아.

    for text in texts:
        tokens = tokenizer.tokenize(text)
        if len(tokens) - max_seq > 2:
            tokens = tokens[:(max_seq-2)]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_ids.append(input_id)
    input_ids = pad_sequences(input_ids, maxlen=max_seq, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for input_id in input_ids:
        seq_mask = [int(id > 0) for id in input_id]
        attention_masks.append(seq_mask)

    labels = data['label'].values

    feature = Feature(input_ids=input_ids, attention_masks=attention_masks, labels=labels)

    return feature


