import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as Data

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

from anytree import Node,AnyNode,RenderTree,find,findall,LevelOrderIter
from anytree.exporter import DotExporter

from sklearn.metrics import precision_recall_fscore_support
from collections import deque
from tqdm import tqdm
import re
import copy
import glob
import os
import sys
import time
import math

from model_dir.model_edu import _ModelEDU
from model_dir.model_trans import _ModelTrans
from model_dir.model_rlat import _ModelRlat

from preprocessor import Preprocessor

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

preprocessor = Preprocessor()

tag_to_ix_1 = { "shift":0,"reduce":1}
tag_to_ix_center = {'1':0,'2':1,'3':2}
tag_to_ix_relation = {'causality':0,'coordination':1,'transition':2,'explanation':3} 

"""
    create gold leaf nodes 
"""
def createLeaf(p,info):
    leafnode_name_list = []
    for idx in range(len(p)):
        leafnode_name_list.append("s"+str(idx))

    leafnode_list = []
    for idx in range(len(p)):
        leafnode_list.append(Node(leafnode_name_list[idx],
            relation="",center="",leaf=True,pos=str(idx),sent=p[idx],is_edu=True))

    return leafnode_list

"""
    create predict leaf nodes
"""
def createPredictLeaf(p,tokenizer,model_3):

    stack = []
    queue = deque(p)

    while queue != deque([]):
        # number of elements in stack < 2 --> always do 'shift'
        if len(stack) < 2:
            stack.append(queue.popleft())
        # number of elements in stack >= 2
        elif queue != deque([]):
            # predict shift or reduce 
            # prepare 3 sentences for model input
            sent1 = tokenizer.tokenize(stack[len(stack)-2])
            sent2 = tokenizer.tokenize(stack[len(stack)-1])
            sent3 = tokenizer.tokenize(queue[0])
            # insert [CLS] and [SEP] to the sentence
            sent1.insert(0,'[CLS]')
            sent1.append('[SEP]')
            sent2.insert(0,'[CLS]')
            sent2.append('[SEP]')
            sent3.insert(0,'[CLS]')
            sent3.append('[SEP]')
            # convert to bert idx
            sent1 = tokenizer.convert_tokens_to_ids(sent1)
            sent2 = tokenizer.convert_tokens_to_ids(sent2)
            sent3 = tokenizer.convert_tokens_to_ids(sent3)
            # convert to tensor
            v1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
            v2_torch = torch.tensor(sent2,dtype=torch.long).cuda()
            v3_torch = torch.tensor(sent3,dtype=torch.long).cuda()
            # predict 'shift' or 'reduce'
            score = model_3(v1_torch.view(1,-1),v2_torch.view(1,-1),v3_torch.view(1,-1))

            max_score, idx = torch.max(score, 1)
            rev_tag_to_ix = {v:k for k,v in tag_to_ix_1.items()}  
            action = rev_tag_to_ix[idx.item()]

            if action == 'shift':
                stack.append(queue.popleft())
                continue
            elif action == 'reduce':
                new_sent = stack[len(stack)-2]+stack[len(stack)-1]
                stack.pop()
                stack.pop()
                stack.append(new_sent)
    # now all the EDUs have been determined
    # create node name for them
    leafnode_name_list = []
    for idx in range(len(stack)):
        leafnode_name_list.append("s"+str(idx))
    # convert EDUs to anytree type
    leafnode_list = []
    for idx in range(len(stack)):
        leafnode_list.append(Node(leafnode_name_list[idx],relation="",center="",leaf=True,pos="",sent=stack[idx],is_edu=True))

    leafnode_sent = []
    for i in range(len(leafnode_list)):
        leafnode_sent.append(leafnode_list[i].sent)

    return leafnode_sent,leafnode_list

"""
    build a predict tree based on predicted EDUs
"""
def buildPredictTree(p,tokenizer,model_1,model_2,model_3,s_list=None,info=None):
    # [v1 from stack, v2 from stack, v3 from queue ] -> label(shift or reduce)
    model_1.eval() # Transition 'shift' or 'reduce'
    model_2.eval() # Relation
    model_3.eval() # EDU

    # golden leaf nodes sentence
    # leafnode_sent = copy.deepcopy(p)
    # golden edu
    # leafnode_list = createLeaf(leafnode_sent,info)

    # end-to-end segemented edu 
    # when testing self-segmented performance
    leafnode_sent,leafnode_list = createPredictLeaf(p,tokenizer,model_3)

    node_list = []
    result = []
    stack = []
    queue = deque(leafnode_sent)
    # teminate when equals sentence of stack[-1] equals to whole paragraph
    terminal = ""
    for s in leafnode_sent:
        terminal += s
    # name of internal nodes
    node_name_list = []
    for idx in range(len(leafnode_sent)-1):
        node_name_list.append("n"+str(idx))

    # terminal condition
    count = 0
    # while stack[len(stack)-1] != terminal:
    while count < len(node_name_list):
        # number of elements in stack < 2 --> always do 'shift'
        if len(stack) < 2:
            stack.append(queue.popleft())
        # number of elements in stack >= 2
        else:
            # queue is empty --> reduce recursively till number of elements in stack == 1(root)
            if queue == deque([]):
                # predict their relation type directly cuz they will reduce to a node anyways
                # prepare 2 sentences for model input
                sent1 = tokenizer.tokenize(stack[len(stack)-2])
                sent2 = tokenizer.tokenize(stack[len(stack)-1])
                # insert [CLS] and [SEP] to the sentence
                sent1.insert(0,'[CLS]')
                sent1.append('[SEP]')
                sent2.insert(0,'[CLS]')
                sent2.append('[SEP]')
                # convert to bert idx
                sent1 = tokenizer.convert_tokens_to_ids(sent1)
                sent2 = tokenizer.convert_tokens_to_ids(sent2)

                v1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
                v2_torch = torch.tensor(sent2,dtype=torch.long).cuda()

                center,relation = model_2(v1_torch.view(1,-1),v2_torch.view(1,-1))
                rev_tag_to_ix_center = {v:k for k,v in tag_to_ix_center.items()}  
                rev_tag_to_ix_relation = {v:k for k,v in tag_to_ix_relation.items()}

                max_score, idx = torch.max(center, 1)
                predict_tag_center = rev_tag_to_ix_center[idx.item()]

                max_score, idx = torch.max(relation, 1)
                predict_tag_relation = rev_tag_to_ix_relation[idx.item()]

                label = predict_tag_relation+'_'+predict_tag_center

                node1 = node2 = None
                # if child(node1,node2) of this node is internal node
                for node in node_list:
                    if node.sent == stack[len(stack)-2]:
                        node1 = node
                    elif node.sent == stack[len(stack)-1]:
                        node2 = node
                # if child(node1,node2) of this node is internal node
                for node in leafnode_list:
                    if node.sent == stack[len(stack)-2]:
                        node1 = node
                    elif node.sent == stack[len(stack)-1]:
                        node2 = node

                if node1.leaf == True and node2.leaf == True: 
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                elif node1.leaf == True: 
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]
                elif node2.leaf == True: 
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                else:
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]

                node_list.append(Node(node_name_list[count],children=[node1,node2],relation=label.split("_")[0],center=label.split("_")[1],leaf=False,pos=pos,sent=node1.sent+node2.sent,is_edu=False))

                count += 1
                stack.pop()
                stack.pop()
                stack.append(node1.sent+node2.sent)

            else:
                # predict shift or reduce
                # prepare 3 sentences for model input
                sent1 = tokenizer.tokenize(stack[len(stack)-2])
                sent2 = tokenizer.tokenize(stack[len(stack)-1])
                sent3 = tokenizer.tokenize(queue[0])
                # insert [CLS] and [SEP] to the sentence
                sent1.insert(0,'[CLS]')
                sent1.append('[SEP]')
                sent2.insert(0,'[CLS]')
                sent2.append('[SEP]')
                sent3.insert(0,'[CLS]')
                sent3.append('[SEP]')
                # convert to bert idx
                sent1 = tokenizer.convert_tokens_to_ids(sent1)
                sent2 = tokenizer.convert_tokens_to_ids(sent2)
                sent3 = tokenizer.convert_tokens_to_ids(sent3)

                v1_torch = torch.tensor(sent1,dtype=torch.long).cuda()
                v2_torch = torch.tensor(sent2,dtype=torch.long).cuda()
                v3_torch = torch.tensor(sent3,dtype=torch.long).cuda()

                score = model_1(v1_torch.view(1,-1),v2_torch.view(1,-1),v3_torch.view(1,-1))

                rev_tag_to_ix = {v:k for k,v in tag_to_ix_1.items()}  

                max_score, idx = torch.max(score, 1)
                action = rev_tag_to_ix[idx.item()]

                if action == 'shift':
                    stack.append(queue.popleft())
                    continue
                elif action == 'reduce':

                    center,relation = model_2(v1_torch.view(1,-1),v2_torch.view(1,-1))

                    rev_tag_to_ix_center = {v:k for k,v in tag_to_ix_center.items()}  
                    rev_tag_to_ix_relation = {v:k for k,v in tag_to_ix_relation.items()}

                    max_score, idx = torch.max(center, 1)
                    predict_tag_center = rev_tag_to_ix_center[idx.item()]

                    max_score, idx = torch.max(relation, 1)
                    predict_tag_relation = rev_tag_to_ix_relation[idx.item()]

                    label = predict_tag_relation+'_'+predict_tag_center

                    node1 = node2 = None

                    # if child(node1,node2) of this node is internal node
                    for node in node_list:
                        if node.sent == stack[len(stack)-2]:
                            node1 = node
                        elif node.sent == stack[len(stack)-1]:
                            node2 = node
                    # if child(node1,node2) of this node is internal node
                    for node in leafnode_list:
                        if node.sent == stack[len(stack)-2]:
                            node1 = node
                        elif node.sent == stack[len(stack)-1]:
                            node2 = node

                    if node1.leaf == True and node2.leaf == True: 
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                    elif node1.leaf == True: 
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]
                    elif node2.leaf == True: 
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                    else:
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]

                    node_list.append(Node(node_name_list[count],children=[node1,node2],relation=label.split("_")[0],center=label.split("_")[1],leaf=False,pos=pos,sent=node1.sent+node2.sent,is_edu=False))

                    count += 1
                    stack.pop()
                    stack.pop()
                    stack.append(node1.sent+node2.sent)

    return node_list,leafnode_list

"""
    build gold tree 
"""
def buildGoldTree(p,s_list,info):
    leafnode_list = createLeaf(p,info)

    sentences = p.copy()
    relations = s_list.copy()

    node_list = []

    node_name_list = []
    for idx in range(len(s_list)):
        node_name_list.append("n"+str(idx))

    count = 0 

    while relations != []:
        for relation in relations:
            # bottom up
            splited_relation = relation.split("|") 
            if set(splited_relation) <= set(sentences): 

                new_sentence = ""

                for r in splited_relation:
                    new_sentence += r

                sent_info = info[info['Sentence']==relation]
                old_label = sent_info['label'].tolist()[0]
                label = preprocessor.changeRelationType(old_label.split("_")[0])  
                # find all children
                child_list = []
                builded_node_list = leafnode_list + node_list
                for r in splited_relation: 
                    for node in builded_node_list:
                        if node.sent == r:
                            child_list.append(node)

                node1 = child_list[0]
                node2 = child_list[-1]
                if node1.leaf == True and node2.leaf == True: 
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                elif node1.leaf == True: 
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]
                elif node2.leaf == True: 
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                else:
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]

                sentences.insert(sentences.index(splited_relation[-1])+1,new_sentence)
                node_list.append(Node(node_name_list[count],children=child_list,relation=label,center=old_label.split("_")[1],leaf=False,pos=pos,sent=new_sentence,old_relation=old_label.split("_")[0]))

                count += 1

                for r in splited_relation:
                    sentences.remove(r)

                relations.remove(relation)
                break

    return node_list

def F_measure_EDU(test,gold):
    PUNCs = (u'?', u'”', u'…', u'—', u'、', u'。', u'」', u'！', u'，', u'：', u'；', u'？')

    test_boundary = []
    gold_boundary = []

    test_sent = ""
    for node in test:
        test_sent += node.sent
        test_sent = test_sent[:-1]
        test_sent += "+"

    gold_sent = ""
    for node in gold:
        gold_sent += node.sent
        gold_sent = gold_sent[:-1]
        gold_sent += "+"

    for s in test_sent:
        if s in PUNCs:
            test_boundary.append(0)
        elif s in '+':
            test_boundary.append(1)

    for s in gold_sent:
        if s in PUNCs:
            gold_boundary.append(0)
        elif s in '+':
            gold_boundary.append(1)

    return test_boundary,gold_boundary

def main():

    model_3 = _ModelEDU(768,2,1).cuda()
    model_1 = _ModelTrans(768,2,1).cuda()
    model_2 = _ModelRlat(768,3,4,1).cuda()

    model_3.load_state_dict(torch.load("saved_model/model_edu.pkl.1")) # load pretrained model
    model_3.eval()
    model_1.load_state_dict(torch.load("saved_model/model_trans.pkl.8")) # load pretrained model
    model_1.eval()
    model_2.load_state_dict(torch.load("saved_model/model_rlat.pkl.4")) # load pretrained model
    model_2.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    p_total = r_total = p_merge = r_merge = p_sense = r_sense = p_center = r_center = p_join = r_join = 0

    edu_p_total = edu_r_total = edu_p_score = edu_r_score = 0

    test_edu_tag_list, gold_edu_tag_list = [], []

    macro_merge, macro_sense, macro_center, macro_join = [], [], [], []

    test_casuality, gold_casuality, test_coordination, gold_coordination = [], [], [], []
    test_transition, gold_transition, test_explanation, gold_explanation = [], [], [], []

    tmp_p_score = tmp_r_score = 0

    all_files = [f for f in sorted(glob.glob('{}/*.{}'.format('test','csv')))]
    # iterate through all training data(csv)

    trange = tqdm(enumerate(all_files),
                  total=len(all_files),
                  desc='testing')
    for i,file in trange:
        trange.set_description("Processing %s" % file)
        df = dataframe = pd.read_csv(file)

        # add label (relation type + center)
        df['label'] = df['RelationType'].map(str) + '_' + df['Center'].map(str)
        label = df[['label']]
        label = label['label'].tolist()

        for idx in set(df['p_id'].tolist()): 
            # same paragragh
            same_parent = df[df['p_id']==idx]
            # format : [ 's1|s2', 's3|s4',...... ]
            s_list = same_parent['Sentence'].tolist()

            if s_list == []:
                continue
            # prepareData : 把有並列關係的都轉成binary (三個以上的sentence可能連在一起)
            p = preprocessor.getGoldenEDU(s_list)

            # seperate them by delimiters
            new_p = preprocessor.getTrainEDU(p)


            with torch.no_grad():
                # predict_node_list,predict_leafnode_list = buildPredictTree(cky_edu,tokenizer,model_1,model_2,model_3,s_list,df)
                predict_node_list,predict_leafnode_list = buildPredictTree(new_p,tokenizer,model_1,model_2,model_3,s_list,df)
                # predict_node_list,predict_leafnode_list = buildPredictTree(p,tokenizer,model_1,model_2,model_3,s_list,df)
                gold_node_list = buildGoldTree(p,s_list,df) 

            leafnode_list = createLeaf(p,df)
            pred_list = predict_leafnode_list + predict_node_list
            gold_list = leafnode_list + gold_node_list

            # # print(RenderTree(f, style=AsciiStyle()).by_attr())
            # #draw all predict tree
            # DotExporter(pred_list[-1]).to_picture("pred.png")
            # #draw all gold tree
            # DotExporter(gold_list[-1]).to_picture("gold.png")

            # print("="*100)

            done = False
            while done != True:
                for node in LevelOrderIter(pred_list[-1]):
                    all_node = [node.name for node in LevelOrderIter(pred_list[-1])]
                    if node.relation == 'coordination' and node.center == '3':

                        modified = False
                        child_list = list(node.children)
                        child = child_list[0]

                        if child_list[0].relation == 'coordination' and child_list[0].center == '3':
                            #idx = child_list.index(child)
                            child_list[0:0] = list(child_list[0].children)
                            child_list.remove(child)

                            predict_node_list.remove(child)

                            node.children = child_list
                            modified = True
                        if modified == True:
                            break
                    if node.name == all_node[-1]:
                        done = True

            pred_list = predict_leafnode_list + predict_node_list
            gold_list = leafnode_list + gold_node_list

            # print(RenderTree(pred_list[-1]))

            # DotExporter(pred_list[-1]).to_picture("new_pred.png")
            # assert 1 == 0
            # draw all predict tree
            # DotExporter(pred_list[-1]).to_picture("tree_photo_predict_notbin/pred_"+str(counter)+"_"+str(idx)+".png")
            # # draw all gold tree
            # DotExporter(gold_list[-1]).to_picture("tree_photo_gold_notbin/gold_"+str(counter)+"_"+str(idx)+".png")

            """
                            PARSEVAL
            """
            # merge , sense , center , join
            r_total += len(gold_node_list)-1
            p_total += len(predict_node_list)-1
            # F measure edu
            test_boundary,gold_boundary = F_measure_EDU(predict_leafnode_list,leafnode_list)
            test_edu_tag_list.extend(test_boundary)
            gold_edu_tag_list.extend(gold_boundary)                

            for predict_node in predict_node_list[:-1]:
                for gold_node in gold_node_list[:-1]:
                    if (predict_node.sent == gold_node.sent):
                        p_merge += 1
                        r_merge += 1

            # sense
            for predict_node in predict_node_list[:-1]:
                for gold_node in gold_node_list[:-1]:
                    if (predict_node.sent == gold_node.sent) and (predict_node.relation == gold_node.relation):
                        p_sense += 1
                        r_sense += 1
            # center
            for predict_node in predict_node_list[:-1]:
                for gold_node in gold_node_list[:-1]:
                    if (predict_node.sent == gold_node.sent) and (predict_node.center == gold_node.center):
                        p_center += 1
                        r_center += 1

            # join
            for predict_node in predict_node_list[:-1]:
                for gold_node in gold_node_list[:-1]:
                    if (predict_node.sent == gold_node.sent) and (predict_node.relation == gold_node.relation) and (predict_node.center == gold_node.center):
                        p_join += 1
                        r_join += 1

    edu_result = precision_recall_fscore_support(test_edu_tag_list,gold_edu_tag_list,average="binary")
    
    print("edu     : f_score = ", round(edu_result[2],5)," %")

    p_score = p_merge / p_total
    r_score = r_merge / r_total
    # F-score = 2*P*R/(P+R)
    f_score = 2 * p_score * r_score / (p_score+r_score)
    print("merge   : f_score = ", round(f_score,5)," %")
    p_score = p_sense / p_total
    r_score = r_sense / r_total
    # F-score = 2*P*R/(P+R)
    f_score = 2 * p_score * r_score / (p_score+r_score)
    print("+sense  : f_score = ", round(f_score,5)," %")
    p_score = p_center / p_total
    r_score = r_center / r_total
    # F-score = 2*P*R/(P+R)
    f_score = 2 * p_score * r_score / (p_score+r_score)
    print("+center : f_score = ", round(f_score,5)," %")
    p_score = p_join / p_total
    r_score = r_join / r_total
    # F-score = 2*P*R/(P+R)
    f_score = 2 * p_score * r_score / (p_score+r_score)
    print("overall : f_score = ", round(f_score,5)," %")


if __name__ == '__main__':
    main()
