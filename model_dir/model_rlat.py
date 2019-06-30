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

tag_to_ix_relation = {'causality':0,'coordination':1,'transition':2,'explanation':3}
tag_to_ix_center = {'1':0,'2':1,'3':2}

class _ModelRlat(nn.Module):

    def __init__(self, embedding_dim, tagset_size_center,tagset_size_relation, batch_size):
        super(_ModelRlat,self).__init__()

        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size_center = tagset_size_center # center label size
        self.tagset_size_relation = tagset_size_relation # relation label size
        self.batch_size = batch_size
        # BERT
        self.bert = BertModel.from_pretrained('bert-base-chinese').cuda()
        # self.dropout = nn.Dropout(0.1)
        self.hidden2tag_center = nn.Linear(embedding_dim*2, self.tagset_size_center)
        self.hidden2tag_relation = nn.Linear(embedding_dim*2, self.tagset_size_relation)

    def forward(self,input_ids1,input_ids2,input_mask1=None,input_mask2=None,labels=None):
        # bert embedding
        out1, pooled1 = self.bert(input_ids1, attention_mask=input_mask1, output_all_encoded_layers=False)
        out2, pooled2 = self.bert(input_ids2, attention_mask=input_mask2, output_all_encoded_layers=False)
        # concat
        output = torch.cat([pooled1,pooled2],-1)
        # reduce dim to label size
        center = self.hidden2tag_center(output)
        relation = self.hidden2tag_relation(output)
        # logits = F.softmax(logits,dim=2)
        return center,relation


class ModelRlat():
    def __init__(self, train_data, test_data, embedding_dim, tagset_size_center,tagset_size_relation, batch_size,k_fold):

        self.train_data = train_data
        self.test_data = test_data
        self.embedding_dim = embedding_dim  # 768(bert)
        self.tagset_size_center = tagset_size_center # center label size
        self.tagset_size_relation = tagset_size_relation # relation label size
        self.batch_size = batch_size
        self.k_fold = k_fold   

        self.model = _ModelRlat(self.embedding_dim,self.tagset_size_center,self.tagset_size_relation,self.batch_size).cuda()
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

        collate_fn = Collator(train_edu=False, train_trans=False, train_rlat=True)

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
                          desc='modelRlat train')

            for step, (sent1, mask1, sent2, mask2, relation, center) in trange:
                # if step == 10:
                #     break
                sent1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
                sent2_torch = torch.tensor(sent2,dtype=torch.long).cuda()

                mask1_torch = torch.tensor(mask1,dtype=torch.long).cuda()
                mask2_torch = torch.tensor(mask2,dtype=torch.long).cuda()

                relation_torch = torch.tensor([relation], dtype=torch.long).cuda()
                center_torch = torch.tensor([center], dtype=torch.long).cuda()

                self.model.zero_grad()

                center,relation = self.model(sent1_torch.view(self.batch_size,-1),sent2_torch.view(self.batch_size,-1),mask1_torch.view(self.batch_size,-1),mask2_torch.view(self.batch_size,-1))

                relation_loss = self.loss_function(center.view(self.batch_size,self.model.tagset_size_relation),relation_torch.view(self.batch_size))
                center_loss = self.loss_function(relation.view(self.batch_size,self.model.tagset_size_center),center_torch.view(self.batch_size))

                loss = []
                loss.append(center_loss)
                loss.append(relation_loss)

                gradients = [torch.tensor(1.0).cuda() for _ in range(len(loss))]
                torch.autograd.backward(loss,gradients)            

                self.optimizer.step()

                running_loss_1 += loss[0].item()
                running_loss_2 += loss[1].item()

                trange.set_postfix(
                    {'center_loss' : '{0:1.5f}'.format(running_loss_1 / (step + 1)),
                     'relation_loss' : '{0:1.5f}'.format(running_loss_2 / (step + 1))
                    }
                )

            print("\n")
            print('[%d] loss of center: %.5f' %
                  (epoch + 1, running_loss_1 * self.batch_size / len(train_data)))
            print('[%d] loss of relation: %.5f' %
                  (epoch + 1, running_loss_2 * self.batch_size / len(train_data)))

            with torch.no_grad():
                self.test_accuracy("train", train_data)
            with torch.no_grad():
                self.test_accuracy("valid", valid_data)
            with torch.no_grad():
                self.test_accuracy("test", test_data)

            torch.save(self.model.state_dict(),'saved_model/model_rlat.pkl.{}'.format(epoch))

    def test(self):
        self.model.load_state_dict(torch.load("saved_model/model_rlat.pkl.8")) # load pretrained model
        self.model.eval()

        with torch.no_grad():
            self.test_accuracy("train", train_data)
        with torch.no_grad():
            self.test_accuracy("valid", valid_data)
        with torch.no_grad():
            self.test_accuracy("test", test_data)

    def test_accuracy(self,phase,data):
        l0 = l1 = l2 = l3 = 0
        t0 = t1 = t2 = t3 = 0
        total = n_corrects = n_wrongs = count = 0 

        trange = tqdm(enumerate(data),
                      total=len(data),
                      desc=phase)

        self.model.eval()
        for step, (sent1, mask1, sent2, mask2, sent3, mask3, relaton,center) in trange:
            # if step == 10:
            #     break
            sent1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
            sent2_torch = torch.tensor(sent2,dtype=torch.long).cuda()

            mask1_torch = torch.tensor(mask1,dtype=torch.long).cuda()
            mask2_torch = torch.tensor(mask2,dtype=torch.long).cuda()

            relation_torch = torch.tensor([relation], dtype=torch.long).cuda()
            center_torch = torch.tensor([center], dtype=torch.long).cuda()

            center,relation = self.model(sent1_torch.view(self.batch_size,-1),sent2_torch.view(self.batch_size,-1),mask1_torch.view(self.batch_size,-1),mask2_torch.view(self.batch_size,-1))
            
            max_score_relation, relation_idx = torch.max(relation, 1)
            max_score_center, center_idx = torch.max(center, 1)

            for j in range(0, len(relation_idx)):
                if relation_idx[j] == relation_torch.view(-1)[j] and center_idx[j] == center_torch.view(-1)[j]:
                    n_corrects += 1
                else:
                    n_wrongs += 1
            total += len(idx)

            for j in range(0, len(relation_idx)):
                if relation_idx[j] == 0:
                    t0 +=1
                if relation_idx[j] == 0:
                    t1 +=1
                if relation_idx[j] == 0:
                    t2 +=1
                if relation_idx[j] == 0:
                    t3 +=1

            for j in range(0, len(relation_idx)):
                if relation_torch.view(-1)[j] == 0:
                    l0 +=1
                if relation_torch.view(-1)[j] == 0:
                    l1 +=1
                if relation_torch.view(-1)[j] == 0:
                    l2 +=1
                if relation_torch.view(-1)[j] == 0:
                    l3 +=1

        print('causality = ',t0," ans = ",l0)
        print('coordination = ',t1," ans = ",l1)
        print('transition = ',t2," ans = ",l2)
        print('explanation = ',t3," ans = ",l3)

        print("\n")
        print(total," ",n_corrects," ",n_wrongs)
        acc = float(n_correct)/float(total)
        acc *= 100
        print("the accuracy of "+ phase + " data is: ",acc,"%")
