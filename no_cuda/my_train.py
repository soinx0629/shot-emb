import time
import pickle
import argparse
import torch

from few_shot import Protonet
from dataloader import Dataloader
from tqdm import tqdm

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def train_epoch(model, train_data, optimizer, repe_n):

    model.train()
    total_loss = 0

    for inde, batch in tqdm(enumerate(train_data), mininterval=2):
        for repe in range(repe_n):
            repe_end = True if repe == repe_n - 1 else False
            tgt, ctx, ctx_pos = batch
            optimizer.zero_grad()

            loss = model(tgt, ctx, repe_end)
            # print('loss before back ward', loss.data)
            loss.backward()
            total_loss += loss.data

            #if inde % 50 == 0 and repe % 5 == 0:
                #print('*'*10+'batch:', inde, 'repe:',repe,'loss:', loss.data, '*'*10)
            optimizer.step()

    return loss/len(train_data*repe)

def train(model, train_data, optimizer, scheduler, opt):

    start = time.time()
    for i in range(opt.epochs):
        scheduler.step()
        print("Epoch: ", i)
        all_loss = train_epoch(model, train_data, optimizer, opt.repe_n)
        print(
            "Training Epoch:{epoch: 2d} done, loss: {loss: 5.5f}, elapse: {elaspe: 3.3f}".format
            (epoch=i, loss=all_loss, elapse=(time.time()-start)/60)
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_data', default='./train.pkl')
    # parser.add_argument('sim_test', default='./sim_test/')

    parser.add_argument('-v_emb',type=int, default=256)
    parser.add_argument('-v_hid', type=int, default=256)

    parser.add_argument('-mem_siz', type=int, default=3000)
    parser.add_argument('-mem_foc', type=int, default=0)

    parser.add_argument('-epochs', type=int, default=15)
    parser.add_argument('-repe_n', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=6)

    parser.add_argument('-lr_rate', type=float, default=0.05)
    parser.add_argument('-drop_rate', type=float, default=0.1)

    parser.add_argument('-save_model', default=None)

    opt = parser.parse_args()
    use_cuda = False#torch.cuda.is_available()
    with open(opt.train_data, 'rb') as at:
        all_sent_dict = pickle.load(at)
    data_param = {'batch_size':6,
                  'use_cuda':use_cuda,
                  'test':False}
    train_data = Dataloader(all_sent_dict, data_param)

    #vocab size should get from dataloader
    model_param = {'nt_words':train_data.nt_words,
                   'nc_words':train_data.nc_words,
                   'v_emb':opt.v_emb,
                   'mem_siz':train_data.nt_words,
                   'mem_foc':opt.mem_foc,
                   'drop_rate':opt.drop_rate,
                   'mem_up_r':0.25,
                   'use_cuda':use_cuda}
    model = Protonet(model_param)
    #if use_cuda: model.cuda()
    model.init_mem(train_data)

    optimizer = optim.Adam(params = model.parameters(), lr=opt.lr_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    train(model, train_data, optimizer, scheduler, opt)

if __name__ == '__main__':
    main()









