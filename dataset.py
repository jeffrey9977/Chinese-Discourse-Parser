import pandas
import numpy
import json
import glob
from torch.utils.data import Dataset

from pytorch_pretrained_bert.tokenization import BertTokenizer


def partition(samples, n):
    '''divide samples into n partitions'''
    assert(len(samples) > n)  # there are more samples than partitions
    assert(n > 0)  # there is at least one partition
    size = len(samples) // n
    for i in range(0, len(samples), size):
        yield samples[i:i+size]


class Collator(object):
    def __init__(self, train_edu, train_trans, train_rlat):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.train_edu = train_edu
        self.train_trans = train_trans
        self.train_rlat = train_rlat

    def __call__(self, batch):
        for sample in batch:

            sample['sent1'] = self.tokenizer.tokenize(sample['sent1'])
            sample['sent2'] = self.tokenizer.tokenize(sample['sent2'])
            if not self.train_rlat:
                sample['sent3'] = self.tokenizer.tokenize(sample['sent3'])

            while len(sample['sent1']) > 510:
                sample['sent1'].pop()
            while len(sample['sent2']) > 510:
                sample['sent2'].pop()
            if not self.train_rlat:
                while len(sample['sent3']) > 510:
                    sample['sent3'].pop()

            sample['sent1'].insert(0, '[CLS]')
            sample['sent1'].append('[SEP]')
            sample['sent2'].insert(0, '[CLS]')
            sample['sent2'].append('[SEP]')
            if not self.train_rlat:
                sample['sent3'].insert(0, '[CLS]')
                sample['sent3'].append('[SEP]')

            # convert to index
            sample['sent1'] = self.tokenizer.convert_tokens_to_ids(
                sample['sent1'])
            sample['sent2'] = self.tokenizer.convert_tokens_to_ids(
                sample['sent2'])
            if not self.train_rlat:
                sample['sent3'] = self.tokenizer.convert_tokens_to_ids(
                    sample['sent3'])

        # calculate the shape of batch (sent1,sent2,sent3)
        allSent1_len = [len(sample['sent1']) for sample in batch]
        allSent2_len = [len(sample['sent2']) for sample in batch]
        if not self.train_rlat:
            allSent3_len = [len(sample['sent3']) for sample in batch]

        longestSent1_sent = max(allSent1_len)
        longestSent2_sent = max(allSent2_len)
        if not self.train_rlat:
            longestSent3_sent = max(allSent3_len)

        data_size = len(batch)

        # prepare data (do padding)
        # sample_id = numpy.ones((data_size, 1)) * 0
        padded_data1 = numpy.ones((data_size, longestSent1_sent)) * 0
        input_mask1 = numpy.ones((data_size, longestSent1_sent)) * 0
        # token_type_ids = numpy.ones((data_size, longest_sent)) * 0
        padded_data2 = numpy.ones((data_size, longestSent2_sent)) * 0
        input_mask2 = numpy.ones((data_size, longestSent2_sent)) * 0

        if not self.train_rlat:
            padded_data3 = numpy.ones((data_size, longestSent3_sent)) * 0
            input_mask3 = numpy.ones((data_size, longestSent3_sent)) * 0

        label = numpy.ones((data_size, 1)) * 0
        if self.train_rlat:
            center = numpy.ones((data_size, 1)) * 0   

        # copy over the actual sequences
        for idx, sample in enumerate(batch):
            # sample_id[idx, 0] = sample['id']
            sent1 = sample['sent1']
            sent2 = sample['sent2']
            if not self.train_rlat:
                sent3 = sample['sent3']
            padded_data1[idx, 0:allSent1_len[idx]] = sent1
            padded_data2[idx, 0:allSent2_len[idx]] = sent2
            if not self.train_rlat:
                padded_data3[idx, 0:allSent3_len[idx]] = sent3
           
            input_mask1[idx, 0:allSent1_len[idx]] = [1] * len(sent1)
            input_mask2[idx, 0:allSent2_len[idx]] = [1] * len(sent2)
            if not self.train_rlat:
                input_mask3[idx, 0:allSent3_len[idx]] = [1] * len(sent3)
            # token_type_ids[idx, 0:all_len[idx]] = [
            #     0] * len(title1) + [1] * len(title2)
            label[idx, 0] = sample['label']
            if self.train_rlat:
                center[idx, 0] = sample['center']

        if not self.train_rlat:
            return padded_data1, input_mask1, padded_data2, input_mask2, padded_data3, input_mask3, label
        else:
            return padded_data1, input_mask1, padded_data2, input_mask2, label,center

class CDTBDataset(Dataset):
    def __init__(self, data: list, train_edu: bool, train_trans: bool, train_rlat: bool,label_map: dict,center_map:dict):
        # required to map the labels to integer values
        self.data = data
        self.label_map = label_map
        self.center_map = center_map
        self.train_edu = train_edu
        self.train_trans = train_trans
        self.train_rlat = train_rlat
        # self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train_edu:
            sample = {
                'sent1': self.data[idx][0],
                'sent2': self.data[idx][1],
                'sent3': self.data[idx][2],
                'label': self.label_map[self.data[idx][3]],
            }
        if self.train_trans:
            sample = {
                'sent1': self.data[idx][0],
                'sent2': self.data[idx][1],
                'sent3': self.data[idx][2],
                'label': self.label_map[self.data[idx][3]],
            }
        if self.train_rlat:

            sample = {
                'sent1': self.data[idx][0],
                'sent2': self.data[idx][1],
                'label': self.label_map[self.data[idx][2].split('_')[0]],
                'center': self.center_map[self.data[idx][2].split('_')[1]],
            }

        return sample


