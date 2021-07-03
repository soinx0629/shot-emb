import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.modules.distance import CosineSimilarity

#def chec_nan(matrix):
   
def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu

def attend(query, key, vector, dk, neg=False):
    if query.data.dim() == 3:
        # xmm = torch.bmm if query.data.dim() == 3 else torch.mm
        sim = torch.bmm(query,key.transpose(1, 2))
    else:
        # print('query in 2', query.data.shape)
        sim = torch.mm(query, key.transpose(0, 1))
    # print('q shape', query.data.shape)
    score = F.softmax(sim) / np.power(dk, 0.5)
    # print('k shape', key.data.shape)
    if neg:
        score = F.softmax(-sim / np.power(dk, 0.5))
    if query.data.dim() == 3:
        attended = torch.bmm(score, vector)
    else:
        attended = torch.mm(score, vector).detach()
    # print(('att shape', attended.data.shape))
    return attended, score

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, param):
        super(Protonet, self).__init__()
        self.nt_words = param['nt_words']
        self.nc_words = param['nc_words']
        self.v_emb = param['v_emb']
        self.mem_siz = param['mem_siz']
        self.mem_foc = param['mem_foc']
        self.mem_up_r = param['mem_up_r']
        self.use_cuda = param['use_cuda']

        self.embed_contex = nn.Embedding(self.nc_words+1, self.v_emb)
        self.embed_target = nn.Embedding(self.nt_words+1, self.v_emb)
        self.cos = CosineSimilarity(dim=1)
        self.att_dropout = nn.Dropout(param['drop_rate'])
        self.tt = torch.cuda if self.use_cuda else torch
        print('model initilized')
        #self.init_mem(all_train_data)
 
    def init_mem(self, all_train_data):
        self.proto_memory = random_uniform((self.mem_siz, self.v_emb), -0.001, 0.001, cuda=False)
        print('initialize memory with context protocol ')
        for inde, batch in tqdm(enumerate(all_train_data)):
            tgt, ctx, _ = batch
            contex_ = self.embed_contex(ctx).detach()
            target_ = self.embed_target(tgt).detach()
            sim = torch.bmm(target_, contex_.transpose(1, 2))
            score = F.softmax(sim) / np.power(self.v_emb, 0.5)
            attended = torch.bmm(score, contex_).mean(0)
            self.proto_memory[inde] = attended.data
        self.prot_mem_v = Variable(self.proto_memory, requires_grad=False)

    def test_quote(self):
            print('this can be quoted')
    def mem_update(self, mem_att_score, proto_contex):
        if self.mem_foc > 0:
            top_sim_in_mem, top_inde = torch.topk(mem_att_score.data, self.mem_foc)
            #print('top sim in mem',top_sim_in_mem.shape)
            #print('proto shape', proto_contex.data.shape)
            top_sim_mem = self.proto_memory[top_inde.squeeze()]
            #print('top sim meme', top_sim_mem.shape)
            try:
                new_v = torch.mm(torch.t(top_sim_in_mem), proto_contex.data)
            except RuntimeError:
                print('mem',top_sim_mem)
                print('sim', top_sim_in_mem)
            #print('new v', new_v.shape)
            new_mem_v = F.normalize(top_sim_mem + new_v)
            self.proto_memory[top_inde.squeeze()] = new_mem_v

    def forward(self, target, contex, repe_end):
        contex = self.embed_contex(contex) # support
        # print('target', target.data)
        target = self.embed_target(target)
        #attend target to the contex
        attented_contex, _ = attend(target, contex, contex, dk=self.v_emb)
        #get the prototype
        proto_contex = attented_contex.squeeze().mean(0).unsqueeze(0)
        mem_contex, mem_att_score = attend(proto_contex, self.prot_mem_v,
                                           self.prot_mem_v, self.v_emb)
        #print(mem_att_score.data[:10])
        # print('type mem:', type(mem_contex))
        # print('type mem:', type(proto_contex))
        full_contex = torch.cat([attented_contex.squeeze(), proto_contex, mem_contex])
        #get the negative contex
        neg_mem_contex, mem_att_score_neg = attend(proto_contex, self.prot_mem_v,
                                                   self.prot_mem_v, self.v_emb, neg=True)
        #calculate loss
        # print('tt shape', target[0].data.shape, 'ct shape', full_contex.data.shape)
        conte_loss = self.cos(full_contex, target[0]).mean(0)
        # print('conte_loss', conte_loss.data)
        # print('loss shape', conte_loss.data.shape)
        neg_loss = -self.cos(neg_mem_contex, target[0])
        # print('neg_loss', neg_loss.data)
        loss = F.logsigmoid(conte_loss) +  F.logsigmoid(neg_loss)
        if repe_end:
            #print('for this batch, repe_ends start updating memory')
            self.mem_update(mem_att_score, proto_contex)

        return -loss


# @register_model('protonet_conv')
# def load_protonet_conv(**kwargs):
#     x_dim = kwargs['x_dim']
#     hid_dim = kwargs['hid_dim']
#     z_dim = kwargs['z_dim']
#
#     def conv_block(in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#     encoder = nn.Sequential(
#         conv_block(x_dim[0], hid_dim),
#         conv_block(hid_dim, hid_dim),
#         conv_block(hid_dim, hid_dim),
#         conv_block(hid_dim, z_dim),
#         Flatten()
#     )
#
#     return Protonet(encoder)
