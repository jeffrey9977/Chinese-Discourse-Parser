import pandas
import numpy
import json
import glob
import copy
from collections import deque

from torch.utils.data import Dataset
from dataset import CDTBDataset

class Preprocessor():

    def __init__(self):
        pass

    def make_dataset(self,file_dir,extension,oversample=False):
        trainDataEDU = []
        trainDataTrans = []
        trainDataRlat = [] 
        self.num_data = 0

        all_files = [f for f in sorted(glob.glob('{}/*.{}'.format(file_dir,extension)))]
        # iterate through all training data(csv)
        for file in all_files:
            df = pandas.read_csv(file)
            # add label (relation type + center)
            df['label'] = df['RelationType'].map(str) + '_' + df['Center'].map(str)

            for idx in set(df['p_id'].tolist()): 
                # get a paragraph everytime

                paragraph = df[df['p_id']==idx]
                # format of s_list : [ 's1|s2', 's3|s4',...... ]
                paragraph = paragraph['Sentence'].tolist()

                if paragraph == []:
                    continue
                # generate golden EDU
                goldenEDU = self.getGoldenEDU(paragraph)
                # generate train EDU training data
                trainEDU = self.getTrainEDU(goldenEDU)

                # generate EDU training data
                self.genTrainData_edu(goldenEDU,trainEDU,trainDataEDU)
                # generate trans training data
                self.genTrainData_trans(goldenEDU,paragraph,trainDataTrans)

            self.genTrainData_rlat(df,trainDataRlat,oversample)

        print(len(trainDataEDU))
        print(len(trainDataTrans))
        print(len(trainDataRlat))

        return CDTBDataset(trainDataEDU, True, False, False, { "shift":0,"reduce":1 }, None) , \
               CDTBDataset(trainDataTrans, False, True, False, { "shift":0,"reduce":1 }, None) , \
               CDTBDataset(trainDataRlat, False, False, True, {'causality':0,'coordination':1,'transition':2,'explanation':3 }, {'1':0,'2':1,'3':2 })
        # assert 1 == 0
            # # generate second pass training data                   


    """ 
        segment the paragraph with golden standard '|'
        and store those sentences(EDU) in list p
    """
    def getGoldenEDU(self,paragraph):
        data = paragraph.copy()
        p = data[0].split("|")
        data.remove(data[0])

        while data != []:
            for sent in data:
                for sent2 in p:
                    if sent.replace("|","") == sent2:
                        data.remove(sent)
                        for idx,item in enumerate(sent.split("|")):
                            p.insert(p.index(sent2)+1+idx,item)
                        p.remove(sent2)

        return p

    """
        slice those EDUs into smaller units with pre-determined delimiters (PUNCs)
        and use them as training data for EDU segmenter
    """
    def getTrainEDU(self,goldenEDU):
        # PUNCs = (u'?', u'”', u'…', u'—', u'、', u'。', u'」', u'！', u'，', u'：', u'；', u'？')
        PUNCs = (u'?', u'”', u'…', u'、', u'。', u'」', u'！', u'，', u'：', u'；', u'？')

        paragraph = ''.join(goldenEDU)
        # seperate them based on PUNCs insteads of using gold standard
        trainEDU = []
        last_pos = 0
        for idx,char in enumerate(paragraph):
            if char in PUNCs:
                trainEDU.append(paragraph[last_pos:idx+1])
                last_pos = idx+1
        if last_pos != len(paragraph):
            trainEDU.append(paragraph[last_pos:len(paragraph)])

        return trainEDU

    def genTrainData_edu(self,gold,data,data_1):
        # [v1 from stack, v2 from stack, v3 from queue ] -> label(shift or reduce)

        # data : data segmented by delimiters
        # gold : gold edu

        tmp = copy.deepcopy(gold)

        stack = []
        queue = deque(data)

        while tmp != []:
            if len(stack) < 2: 
                stack.append(queue.popleft())

            elif queue == deque([]):
                break              
            else:
                reduce_flag = False
                sub_edu = stack[len(stack)-2] + stack[len(stack)-1]
                # reduce
                for gold_edu in tmp:
                    if sub_edu in gold_edu:
                        reduce_flag = True
                        if queue != deque([]):
                            data_1.append([ stack[len(stack)-2], stack[len(stack)-1], queue[0],"reduce" ])
                            # data_2.append([])  

                            # data_2.append([ stack[len(stack)-2], stack[len(stack)-1],"edu_1" ])

                        stack.pop()
                        stack.pop()
                        stack.append(sub_edu)
                        if sub_edu == gold_edu:
                            tmp.remove(sub_edu)
                # shift
                if reduce_flag == False:
                    data_1.append([ stack[len(stack)-2], stack[len(stack)-1], queue[0],"shift" ])
                    stack.append(queue.popleft())
        
        # for i in range(len(train_data_1)):
        #     print(train_data_1[i])
        # print('-'*100)
        return 

    def genTrainData_trans(self,p,s_list,train_data_1):

        # [v1 from stack, v2 from stack, v3 from queue ] -> label(shift or reduce)
        tmp = self.genBinaryAction(p,s_list)
        data = tmp.copy()

        stack = []
        queue = deque(p)

        while data != []:
            if len(stack) < 2: 
                stack.append(queue.popleft())
            else:
                relation = stack[len(stack)-2] + "|" + stack[len(stack)-1]
                du = stack[len(stack)-2] + stack[len(stack)-1]
                # reduce
                if relation in data:
                    # action_list.append("reduce")
                    if queue != deque([]): # make sure queue[0] has sentence
                        train_data_1.append([ stack[len(stack)-2], stack[len(stack)-1], queue[0],"reduce" ])
                    stack.pop()
                    stack.pop()
                    stack.append(du)
                    data.remove(relation)
                # shift
                else:
                    train_data_1.append([ stack[len(stack)-2], stack[len(stack)-1], queue[0],"shift" ])
                    stack.append(queue.popleft())
                    # action_list.append("shift")
        return 

    def genTrainData_rlat(self,df,trainDataRlat,oversample):

        info = df[['p_id','r_id','RelationType','Sentence','Center','label']]

        for idx in range(len(info)):

            sent = info.iloc[idx,3].split("|")
            # coordination 
            if len(sent) >= 3:
                for idx1 in range(len(sent)-1):
                    for idx2 in range(idx1+1,len(sent)):
                        newRelation = self.changeRelationType(info.iloc[idx,2])
                        trainDataRlat.append([sent[idx1],sent[idx2],newRelation + "_" + str(info.iloc[idx,4])])
                        self.num_data += 1
            else:
                newRelation = self.changeRelationType(info.iloc[idx,2])
                trainDataRlat.append([sent[0],sent[1],newRelation + "_" + str(info.iloc[idx,4])])
                self.num_data += 1 
                # over sample 
                if trainDataRlat[self.num_data-1][2].split("_")[0] != "coordination" and oversample:

                    data_label = trainDataRlat[self.num_data-1][2].split("_")
                    new_label = ""
                    if data_label[1] == "1":
                        new_label = data_label[0] + "_" + "2"
                    elif data_label[1] == "2":
                        new_label = data_label[0] + "_" + "1"
                    else:
                        new_label = data_label[0] + "_" + "3"

                    trainDataRlat.append([trainDataRlat[self.num_data-1][1],trainDataRlat[self.num_data-1][0],new_label])
                    self.num_data += 1         

    def genBinaryAction(self,p,s_list):
        # ex: 1|2|3   --> 1|2 , 12|3
        #     1|2|3|4 --> 1|2 , 12|3 , 123|4
        data = s_list.copy()

        for sent in data:
            if len(sent.split("|")) >= 3:
                nodes = sent.split("|")
                for idx in range(0,len(nodes)-1):
                    data.insert(data.index(sent)+1+idx,nodes[0]+"|"+nodes[1])
                    new_node = nodes[0]+nodes[1]
                    nodes.remove(nodes[1])
                    # if idx == 0:
                    nodes.remove(nodes[0])
                    nodes.insert(0,new_node)

                data.remove(sent)

        bin_s_list = data.copy()
        return bin_s_list

    def changeRelationType(self,relationType):

        if relationType in ['因果关系','背景关系','目的关系','条件关系','假设关系','推断关系'] :
            relationType = 'causality'
        elif relationType in ['例证关系','解说关系','总分关系','评价关系']:
            relationType = 'explanation'
        elif relationType in ['并列关系','顺承关系','对比关系','递进关系','选择关系']:
            relationType = 'coordination'
        elif relationType in ['让步关系','转折关系']:
            relationType = 'transition'

        return relationType


