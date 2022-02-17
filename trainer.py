import os
import logging
import time
import tqdm

import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from utils import set_device, get_accuracy
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, train_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = set_device()
        self.model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)

    def train(self):

        train_inputs = torch.tensor(self.train_dataset.input_ids)
        train_labels = torch.tensor(self.train_dataset.labels)
        train_masks = torch.tensor(self.train_dataset.attention_masks)

        batch_size = 32 # config로 빼야지

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(),
                          lr=5e-5,  # 학습률
                          eps=2e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                          )

        epochs = 3
        total_loss = 0
        p_iteration = 500
        total_correct = 0
        total_len = 0
        total_steps = len(train_loader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        self.model.train()

        for epoch in range(epochs):
            for step, batch in enumerate(train_loader):
                if step and step % p_iteration == 0:
                    print('[Epoch {}/{}] Iteration {} -> Accuracy: {:.3f}'.format(epoch + 1,
                                                                                  epochs,
                                                                                  step,
                                                                                  total_correct / total_len))
                    total_len = 0
                    total_correct = 0

                batch = tuple(b.to(self.device) for b in batch)
                input_ids, attention_mask, labels = batch

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                loss, logits = outputs

                pred = torch.argmax(F.softmax(logits), dim=1)
                correct = pred.eq(labels)
                total_correct += correct.sum().item()
                total_len += len(labels)

                total_loss += loss.item()

                loss.backward()

                clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()

                self.model.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        print("\n   Average training loss: {0:.2f}".format(avg_train_loss))

    def eval(self):

        self.model.eval()

        test_inputs = torch.tensor(self.test_dataset.input_ids)
        test_labels = torch.tensor(self.test_dataset.labels)
        test_masks = torch.tensor(self.test_dataset.attention_masks)

        batch_size = 32 # config로 빼야지

        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for step, batch in enumerate(test_loader):

            batch = tuple(b.to(self.device) for b in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)

            # 로스 구함
            logits = outputs[0]

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy = get_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("")
        print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))


