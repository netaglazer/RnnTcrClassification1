import train
import os
import shutil
import json
import argparse
import logging
import nni
import time
from datetime import datetime
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loader import MyDataset, load_data
import utils
from model import CharacterLevelCNN
from focal_loss import FocalLoss
_logger = logging.getLogger("rnn_tcr")

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300


def prepare(args):
    parser = argparse.ArgumentParser(
        'Character Based CNN for text classification')
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--label_column', type=str, default='hla')
    parser.add_argument('--text_column', type=str, default='tcr')
    parser.add_argument('--chunksize', type=int, default=50000)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--encoding', type=str, default='utf-8')
    parser.add_argument('--balance', type=int, default=0, choices=[0, 1])
    parser.add_argument('--alphabet', type=str ,default="ARNDCQEGHILKMFPSTWYV")
    parser.add_argument('--sep', type=str, default=',')
    parser.add_argument('--steps', nargs='+', default=['lower'])
    parser.add_argument('--group_labels', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ignore_center', type=int, default=1, choices=[0, 1])
    parser.add_argument('--label_ignored', type=int, default=None)
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--use_sampler', type=int ,default=0, choices=[0, 1])
    parser.add_argument('--dropout_input', type=float, default=args['dropout_input'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    if args['optimizer'] == 'sgd':
        parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd')
    if args['optimizer'] == 'Adam':
        parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--l2_regularization', type=float, default=args['l2_regularization'])
    parser.add_argument('--learning_rate', type=float, default= args['lr'])
    parser.add_argument('--number_of_characters', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=25) ##לבדוק
    parser.add_argument('--extra_characters', type=str, default='')
    parser.add_argument('--class_weights', type=int, default=0, choices=[0, 1])
    parser.add_argument('--focal_loss', type=int, default=0, choices=[0, 1])
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--scheduler', type=str, default='step', choices=['clr', 'step'])
    parser.add_argument('--min_lr', type=float, default=1.7e-3)
    parser.add_argument('--max_lr', type=float, default=1e-2)
    parser.add_argument('--stepsize', type=float, default=4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--early_stopping', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, choices=[0, 1], default=1)
    parser.add_argument('--workers', type=int, default=1)
    #     parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--log_f1', type=int, default=1, choices=[0, 1])
    parser.add_argument('--flush_history', type=int, default=1, choices=[0, 1])
    parser.add_argument('--output', type=str, default='./models/')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--model_type', type=str, default=args['model_type'])
    # parser.add_argument('--momentum', type=float, default=args['momentum'])

    run_args = parser.parse_args(args=[])
    return run_args


if __name__ == "__main__":

    try:
        RCV_CONFIG = nni.get_next_parameter()
    except Exception as exception:
        _logger.exception(exception)
        raise
    args = prepare(RCV_CONFIG)
    # args = prepare({'optimizer': 'Adam', 'lr': 0.001, 'model_type': 'small', 'dropout_input': 0.4, 'l2_regularization':0.001})
    train.run(args)