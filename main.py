import argparse

from utils import init_logger, set_device, set_seed
from data_loader import read_data, tokenize


def main(args):
    init_logger()
    set_seed()

    train_data, test_data = read_data(args)
    tokenize(train_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default="data/ratings_train.txt")
    parser.add_argument("--test_file", default="data/ratings_test.txt")
    parser.add_argument("--lr", default="10e-3", help="learning rate")

    args = parser.parse_args()

    main(args)