import torch
from torch import nn
from torch.autograd import Variable
import torch.functional as F
import pickle
from collections import Counter

import numpy as np
import os

class Dataloader(object):

    def __init__(self, all_sent, params):
        self.batch_size = params['batch_size']
        self.use_cuda = params['use_cuda']
        self.test = params['test']
        self._iter_count = 0
        #tak this apart to the training script
        self.all_sent = all_sent
        self.t_dict, self.c_dict = self.build_dict()
        self._n_batch = len(self.t_dict)


    def build_dict(self):

        target_dict = dict()
        for i in self.all_sent.keys():
            target_dict[i] = len(target_dict)+1
        self.nt_words = len(target_dict)

        contex_dict = dict()
        contex_counter = Counter()
        for i in self.all_sent.values():
            joi = ''.join(i)
            for j in joi.split():
                contex_counter[j] += 1
        for i,j in contex_counter.items():
            if j>2:
                contex_dict[i] = len(contex_dict)+1
        contex_dict['UNK'] = 0
        self.nc_words = len(contex_dict)

        return target_dict, contex_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):

        def pad_to_longest(insts):
            # print('insts', insts)
            max_len = max([len(inst) for inst in insts])

            inst_data = np.array([inst + [0] * (max_len-len(inst))
                                  for inst in insts])
            inst_position = np.array([[pos+1 if not i==0 else 0 for pos, i in enumerate(inst)]
                                      for inst in inst_data])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)
            inst_position_tensor = Variable(
                torch.LongTensor(inst_position), volatile=self.test)

            #if self.use_cuda:
                #inst_data_tensor = inst_data_tensor.cuda()
                #inst_position_tensor = inst_position_tensor.cuda()

            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            tt = list(self.t_dict.keys())[batch_idx]
            tt_id = torch.LongTensor([self.t_dict[tt]]).expand(self.batch_size,1)
            #tt_id = tt_id.cuda()
            tt_id = Variable(tt_id, volatile=self.test)
            #if self.use_cuda:
                #try:
                    #tt_idc = tt_id.cuda()
                #except RuntimeError:
                    #print('cuda runtime error triggerd on')
                    #print('tt is', tt, 'wrong tt_id',tt_id) 
            ct = self.all_sent[tt]
            ct_id = []
            for i in ct:
                temp = []
                for j in i.strip('\n').split():
                    if j in self.c_dict.keys():
                        temp.append(self.c_dict[j])
                    else:
                        temp.append(self.c_dict['UNK'])
                ct_id.append(temp)
            # print('context is', ct_id)
            ct_data, ct_pos = pad_to_longest(ct_id)

            return tt_id, ct_data, ct_pos
        else:
            self._iter_count = 0
            raise StopIteration


    # def generate_batch(self):
    #     a_list = list(self.target_dict.keys())
    #     global t_index
    #     selected_tt = self.target_dict[a_list[t_index]]
    #     selected_ct = self.word2sent_index[a_list[t_index]]
    #     ct_id = []
    #     for i in selected_ct:
    #         ct_id.append(self.data2id[i])
    #     t_index += 1
    #     return selected_tt, ct_id




















