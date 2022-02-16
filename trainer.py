import os
import logging
import time
import tqdm import tqdm, trange

import numpy as np
import torch
import torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from utils import set_device

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, test_dataset=None):
        self.args=args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = set_device()
        self.model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)

    def train(self):
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(self.train_dataset.input_ids,
                                                                                            self.train_dataset.labels,
                                                                                            random_state=2021,
                                                                                            test_size=0.1)
        train_masks, validation_masks, _, _ = train_test_split(self.train_dataset.attention_masks,
                                                               self.train_dataset.input_ids,
                                                               random_state=2021,
                                                               test_size=0.1)
        train_inputs = torch.tensor(train_inputs)
        train_labels = torch.tensor(train_labels)
        train_masks = torch.tensor(train_masks)
        validation_inputs = torch.tensor(validation_inputs)
        validation_labels = torch.tensor(validation_labels)
        validation_masks = torch.tensor(validation_masks)

        # batch size 어떻게 정하는지, GPU에 fit 하는지 확인하는 방법
        batch_size = 64

        #https://subinium.github.io/pytorch-dataloader/

        #tensor dataset의 경우 class로 뺄 수 있을 거 같음. __getitem__
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), #Adam을 안쓰는경우는 뭘까
                          lr=2e-5,  # 학습률
                          eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                          )

        epochs = 4
        total_steps = len(train_dataloader) * epochs
        #.get_cosine_schedule_with_warmup 이건 어떤걸 학습할때  쓰이는 걸까?
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        self.model.zero_grad()
        t0 = time.time()
        total_loss = 0
        self.model.train()
