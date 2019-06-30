import argparse
import os 
import glob
import pandas as pd
import numpy as np
from dataset import CDTBDataset, partition, Collator
from preprocessor import Preprocessor

def main(args,k_fold=10,batch_size=2):

    if args.make_dataset:
        preprocessor = Preprocessor()
        trainDataEDU,trainDataTrans,trainDataRlat = preprocessor.make_dataset('./train','csv',oversample=True)
        testDataEDU,testDataTrans,testDataRlat = preprocessor.make_dataset('./test','csv',oversample=False) 

    if args.train_edu:
        from model_dir.model_edu import ModelEDU
        model = ModelEDU(trainDataEDU, testDataEDU, embedding_dim=768, tagset_size=2, batch_size=batch_size, k_fold=k_fold)
        model.train()

    if args.train_trans:
        from model_dir.model_trans import ModelTrans
        model = ModelTrans(trainDataTrans, testDataTrans, embedding_dim=768, tagset_size=2, batch_size=batch_size, k_fold=k_fold)
        model.train()

    if args.train_rlat:
        from model_dir.model_rlat import ModelRlat
        model = ModelRlat(trainDataRlat, testDataRlat, embedding_dim=768, tagset_size_center=3,tagset_size_relation=4, batch_size=batch_size, k_fold=k_fold)
        model.train()


def parse():
    parser = argparse.ArgumentParser(description="Discourse Parsing with shfit reduce method")
    parser.add_argument('--make_dataset',action='store_true',help='whether make training dataset')
    parser.add_argument('--train_edu', action='store_true', help='whether train edu segmenter')
    parser.add_argument('--train_trans', action='store_true', help='whether train shift & reduce parsing')
    parser.add_argument('--train_rlat', action='store_true', help='whether train labeling for merged node')
    parser.add_argument('--test', action='store_true', help='whether test performance')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    main(args)

