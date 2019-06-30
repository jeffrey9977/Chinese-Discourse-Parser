import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as Data

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

import copy
from tqdm import tqdm
from dataset import CDTBDataset, partition, Collator

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

class _ModelEDU(nn.Module):

    def __init__(self, embedding_dim, tagset_size, batch_size):
        super(_ModelEDU,self).__init__()

        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size = tagset_size
        self.batch_size = batch_size
        # BERT
        self.bert = BertModel.from_pretrained('bert-base-chinese').cuda()
        # self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(embedding_dim*3, tagset_size)

    def forward(self,input_ids1,input_ids2,input_ids3,input_mask1=None,input_mask2=None,input_mask3=None,labels=None):
        # bert embedding
        out1, pooled1 = self.bert(input_ids1, attention_mask=input_mask1, output_all_encoded_layers=False)
        out2, pooled2 = self.bert(input_ids2, attention_mask=input_mask2, output_all_encoded_layers=False)
        out3, pooled3 = self.bert(input_ids3, attention_mask=input_mask3, output_all_encoded_layers=False)
        # concat
        output = torch.cat([pooled1,pooled2,pooled3],-1)
        # reduce dim to label size
        logits = self.hidden2tag(output)
        # logits = F.softmax(logits,dim=2)
        return logits


class ModelEDU():
    def __init__(self, train_data, test_data, embedding_dim, tagset_size, batch_size,k_fold):

        self.train_data = train_data
        self.test_data = test_data
        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size = tagset_size
        self.batch_size = batch_size
        self.k_fold = k_fold   

        self.model = _ModelEDU(self.embedding_dim,self.tagset_size,self.batch_size).cuda()
        self.optimizer = optim.SGD(self.model.parameters(),lr= 5e-5)
        self.loss_function = nn.CrossEntropyLoss().cuda() # delete ignore_index = 0 



    def train(self):
        indices = list(range(len(self.train_data)))
        np.random.shuffle(indices)

        partitions = list(partition(indices, self.k_fold))

        train_idx = [idx for part in partitions[0:self.k_fold-1] for idx in part]
        valid_idx = partitions[self.k_fold-1]

        # randomly sample from only the indicies given
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        collate_fn = Collator(train_edu=True, train_trans=False, train_rlat=False)

        train_data = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn
        )

        valid_data = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            collate_fn=collate_fn
        )

        test_data = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )

        for epoch in range(10): 
            running_loss = 0.0
            self.model.train()
            i = 0

            trange = tqdm(enumerate(train_data),
                          total=len(train_data),
                          desc='modelEDU train')

            for step, (sent1, mask1, sent2, mask2, sent3, mask3, label) in trange:

                sent1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
                sent2_torch = torch.tensor(sent2,dtype=torch.long).cuda()
                sent3_torch = torch.tensor(sent3,dtype=torch.long).cuda()

                mask1_torch = torch.tensor(mask1,dtype=torch.long).cuda()
                mask2_torch = torch.tensor(mask2,dtype=torch.long).cuda()
                mask3_torch = torch.tensor(mask3,dtype=torch.long).cuda()                

                label_torch = torch.tensor([label], dtype=torch.long).cuda()

                self.model.zero_grad()

                score = self.model(sent1_torch.view(self.batch_size,-1),sent2_torch.view(self.batch_size,-1),sent3_torch.view(self.batch_size,-1),mask1_torch.view(self.batch_size,-1),mask2_torch.view(self.batch_size,-1),mask3_torch.view(self.batch_size,-1))

                loss = self.loss_function(score.view(self.batch_size,self.model.tagset_size),label_torch.view(self.batch_size))

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() 

                trange.set_postfix(
                    {'loss' : '{0:1.5f}'.format(running_loss / (step + 1))}
                )

            print("\n")
            print('[%d] loss: %.5f' %
                  (epoch + 1, running_loss * self.batch_size / len(train_data)))

            with torch.no_grad():
                self.test_accuracy("train", train_data)
            with torch.no_grad():
                self.test_accuracy("valid", valid_data)
            with torch.no_grad():
                self.test_accuracy("test", test_data)

            torch.save(self.model.state_dict(),'saved_model/model_edu.pkl.{}'.format(epoch))

    def test(self):
        self.model.load_state_dict(torch.load("saved_model/model_edu.pkl.8")) # load pretrained model
        self.model_1.eval()

        with torch.no_grad():
            self.test_accuracy("train", train_data)
        with torch.no_grad():
            self.test_accuracy("valid", valid_data)
        with torch.no_grad():
            self.test_accuracy("test", test_data)

    def test_accuracy(self,phase,data):
        total = n_corrects = n_wrongs = count = 0 

        trange = tqdm(enumerate(data),
                      total=len(data),
                      desc=phase)

        self.model.eval()
        for step, (sent1, mask1, sent2, mask2, sent3, mask3, label) in trange:
            # if step == 10:
            #     break
            sent1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
            sent2_torch = torch.tensor(sent2,dtype=torch.long).cuda()
            sent3_torch = torch.tensor(sent3,dtype=torch.long).cuda()

            mask1_torch = torch.tensor(mask1,dtype=torch.long).cuda()
            mask2_torch = torch.tensor(mask2,dtype=torch.long).cuda()
            mask3_torch = torch.tensor(mask3,dtype=torch.long).cuda()

            label_torch = torch.tensor([label], dtype=torch.long).cuda()

            score = self.model(sent1_torch.view(self.batch_size,-1),sent2_torch.view(self.batch_size,-1),sent3_torch.view(self.batch_size,-1),mask1_torch.view(self.batch_size,-1),mask2_torch.view(self.batch_size,-1),mask3_torch.view(self.batch_size,-1))
            
            max_score, idx = torch.max(score, 1)

            for j in range(0, len(idx)):
                if idx[j] == label_torch.view(-1)[j]:
                    n_corrects += 1
                else:
                    n_wrongs += 1
            total += len(idx)

        print("\n")
        print(total," ",n_corrects," ",n_wrongs)
        acc = float(n_corrects)/float(total)
        acc *= 100
        print("the accuracy of "+ phase + " data is: ",acc,"%")

