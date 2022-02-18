import argparse

from utils import init_logger, set_seed
from data_loader import read_data, process
from trainer import Trainer
import os

def main(args):
    init_logger()
    set_seed()

    train_data, test_data = read_data(args)
    train_dataset = process(data=train_data, max_seq=128)
    test_dataset = process(data=test_data, max_seq=128)

    trainer = Trainer(args, train_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default="data/ratings_train.txt")
    parser.add_argument("--test_file", default="data/ratings_test.txt")
    parser.add_argument("--lr", default="10e-3", help="learning rate")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    os.environ["TORCH_CUDA_ARCH_LIST"] = '7.5'

    main(args)