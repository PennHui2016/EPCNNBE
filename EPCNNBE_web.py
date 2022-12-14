import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold
import random
import os
import copy
from scipy import stats
import csv
import sys
import time
from Bio.Seq import Seq
from find_spacer import ACBE_spacers

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch()

def encode_seq(seqs, st, en):
    N = len(seqs)
    L = int(en - st)
    encoded_seqs = np.zeros((N, 4, L))
    for n in range(0, N):
        seq = seqs[n][st:en]
        seq = seq.upper()

        encoded_seq = np.zeros((4, L))
        for i in range(0, len(seq)):
            if seq[i] == 'A':
                encoded_seq[0, i] = 1
            elif seq[i] == 'C':
                encoded_seq[1, i] = 1
            elif seq[i] == 'G':
                encoded_seq[2, i] = 1
            elif seq[i] == 'T':
                encoded_seq[3, i] = 1
        encoded_seqs[n, :, :] = encoded_seq
    return encoded_seqs

def onehot_encoding(seqs, st, en):
    encoded_seq_target = encode_seq(seqs[:, 0], st, en)
    encoded_seq_outcome = encode_seq(seqs[:, 1], st, en)
    target_oh = np.array(encoded_seq_target)
    target_oh = target_oh.reshape(target_oh.shape[0], 1, target_oh.shape[1], target_oh.shape[2])
    outcome_oh = np.array(encoded_seq_outcome)
    outcome_oh = outcome_oh.reshape(outcome_oh.shape[0], 1, outcome_oh.shape[1], outcome_oh.shape[2])
    return target_oh, outcome_oh

class CnnRegression(nn.Module):
    def __init__(self, in_dim=1, out_dim=10, k1=1, k2=3, p=0, ns=1120, s=1, pd=0, di=1):
        super(CnnRegression, self).__init__()
        self.conv = nn.Sequential(
            # input : (N, Cin, hi, wi) = (bs, 1, 4, l)
            # output: (N, Cout, ho, wo) = (bs, 10, 4, l-k+1)
            # ho = (hi+2*padding-dilation[0]*(kernel_size[0]ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢1)ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢1)/stride[0] + 1 = 4-(k1-1)-1+1 = 5-k1
            # wo = (wi+2*padding-dilation[1]*(kernel_size[1]ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢1)ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢1)/stride[1] + 1 = 30-(k2-1)-1+1 = 31-k2
            nn.Conv2d(in_dim, out_dim, (k1, k2), stride=s, padding=pd),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
            nn.Dropout(p),
        )  # 5*5*16
        # ÃƒÂ¥Ã‚Â®Ã…Â¡ÃƒÂ¤Ã‚Â¹Ã¢â‚¬Â°ÃƒÂ¥Ã¢â‚¬Â¦Ã‚Â¨ÃƒÂ¨Ã‚Â¿Ã…Â¾ÃƒÂ¦Ã…Â½Ã‚Â¥ÃƒÂ¥Ã‚Â±Ã¢â‚¬Å¡
        self.fc = nn.Sequential(
            nn.Linear(ns * 2, 4096),  # ns = out_dim * (5-k1) * (31-k2)
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(4096, 1024),  # ns = out_dim * (5-k1) * (31-k2)
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p))
        self.pred = nn.Linear(128, 1)
        self.sg = nn.Sigmoid()
        self.sf = nn.Softmax(dim=1)

    def forward(self, x1, x2):  # , x3):
        out1 = self.conv(x1)
        out2 = self.conv(x2)

        out1 = out1.view(out1.size(0), out1.size(1) * out1.size(2) * out1.size(3))  # 400 = 3*1*438
        out2 = out2.view(out2.size(0), out2.size(1) * out2.size(2) * out2.size(3))
        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        out = self.sf(out) * out
        out = self.pred(out)
        out = self.sg(out)
        return out

## case study
def request_out(dataset, preds):
    preds = np.array(preds).squeeze().T
    pre_test_ens = np.mean(preds, axis=1)
    por_pred = np.array(pre_test_ens).reshape(len(pre_test_ens), 1)
    por_pred[np.where((por_pred[:, 0] < 0))[0], :] = 0
    por_pred[np.where((por_pred[:, 0] > 1))[0], :] = 1

    seqs = dataset['seqs']
    targets = seqs[:, 0]
    outcomes = seqs[:, 1]
    uni_tar = np.unique(targets)
    effs = np.zeros((len(uni_tar), 1))
    eff_preds = np.zeros((len(targets), 1))
    out_pors = np.zeros((len(targets), 1))
    out_fres = np.zeros((len(targets), 1))

    pss = np.zeros((len(targets), 1)) # picking score

    for i in range(0, len(uni_tar)):
        ind = np.where((targets == uni_tar[i]))[0]
        i_por_p = por_pred[ind, 0] / sum(por_pred[ind, 0])
        i_out = outcomes[ind]

        ind1 = np.where((i_out != uni_tar[i]))[0]  # edited
        ind2 = np.where((i_out == uni_tar[i]))[0]  # unedited

        if len(ind1) == 0:
            effs[i, 0] = 0
        else:
            effs[i, 0] = sum(i_por_p[ind1])

        eff_preds[ind, 0] = effs[i, 0]

        i_por_p1 = i_por_p / (sum(i_por_p[ind1]) + 1E-14)

        i_por_p1[ind2] = -1

        out_pors[ind, 0] = i_por_p1

        fre_p = effs[i, 0] * i_por_p1

        fre_p[ind2] = i_por_p[ind2]

        out_fres[ind, 0] = fre_p

        i_ps = np.zeros((len(ind), 1))

        for j in range(len(ind)):
            if j != ind2:
                ps = 2 * i_por_p[j] * (1 - (sum(i_por_p[ind1])-i_por_p[j]))/(i_por_p[j] + (1 - (sum(i_por_p[ind1])-i_por_p[j])))
                i_ps[j] = ps

        pss[ind, 0] = i_ps[:, 0]

    request_res = np.column_stack((seqs, eff_preds, out_pors, out_fres, pss))

    return request_res

## final tool
def ensemble_test_request(be, seq_ty, ens_ty, test, weight_fold):
    device = torch.device('cpu')
    para = pd.read_csv(weight_fold + 'para_' + be + '_' + str(ens_ty) + '_' + str(seq_ty) + '.csv', sep='\t', header=0).fillna(0).values
    pres = []
    for i in range(0, 10):
        o = int(para[i, 7])
        k1 = int(para[i, 4])
        k2 = int(para[i, 5])
        p = int(para[i, 6])
        ns = int(para[i, 8])

        model = []
        model = CnnRegression(out_dim=o, k1=k1, k2=k2, p=p, ns=ns)
        # weight_path = weight_fold + be + '_model_top' + str(i) + '_' + str(para[i, 1]) + '_' +\
        #               str(para[i, 2]) + '_' +\
        #               str(para[i, 3]) + '_' +\
        #               str(para[i, 4]) + '_' +\
        #               str(para[i, 5]) + '_' +\
        #               str(float(para[i, 6])) + '_' +\
        #               str(para[i, 7]) + '_' +\
        #               str(para[i, 8]) + '_' +\
        #               str(ens_ty) + '_' + str(seq_ty)
        weight_path = weight_fold + be + '_model_top' + str(i) + '_' + str(ens_ty) + '_' + str(seq_ty)
        #weight_path = save_fold + 'best/' + be + '_model_top' + str(i) + '_' + str(ens_ty) + '_' + str(seq_ty)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            te_s1 = torch.from_numpy(test['tar_oh'].astype(np.float32)).float()
            te_s2 = torch.from_numpy(test['oc_oh'].astype(np.float32)).float()

            pred_s = model(te_s1, te_s2)
            pre_score = pred_s.detach().to(torch.device('cpu')).numpy()
        Y_pred = np.array(pre_score).reshape(len(pre_score), 1)
        pres.append(Y_pred)
    test_res = request_out(test, pres)
    #print(test_res)
    return test_res

# case study
def user_request(filepath, filetype, be, wt, seq_ty, ens_ty, save_path, weight_fold):
    in_seqs, out_seqs, spacer_ex_all, cut_fea_all = ACBE_spacers(filepath, filetype, be, wt)
    seqs_all = np.column_stack((np.array([in_seqs]).T, np.array([out_seqs]).T))
    if (wt == 0) & (seq_ty == 0):
        target_oh, outcome_oh = onehot_encoding(seqs_all, 0, 20)
    elif (wt == 1) & (seq_ty == 0):
        target_oh, outcome_oh = onehot_encoding(seqs_all, 4, 24)
    elif (wt == 1) & (seq_ty == 1):
        target_oh, outcome_oh = onehot_encoding(seqs_all, 0, 30)

    test = {'tar_oh': target_oh, 'oc_oh': outcome_oh, 'seqs': seqs_all}
    res = ensemble_test_request(be, seq_ty, ens_ty, test, weight_fold)

    #save_path = save_fold
    with open(save_path, 'w', newline='') as ws:
        writer = csv.writer(ws)
        writer.writerow(['extend target', 'outcome', 'edit efficiency', 'outcome proportion', 'outcome frequecy', 'picking score'])
        writer.writerows(res)

    res_df = {'extend_target':res[:, 0], 'outcome':res[:, 1],
              'edit efficiency':res[:, 2], 'outcome proportion':res[:, 3],
                 'outcome frequecy':res[:, 4], 'picking score':res[:, 5]}
    return res_df

def IO(be='ABE', pth='./test.fa', exp_out=''):
    seed_torch()
    ## accepte user input sequences and save as a '.fa' file
    filepath = pth
    ## the path for network weights
    # weight_fold = 'F:/HUI/base_editing/best/'
    weight_fold = './be_weights/'
    ## constant, no need to change
    filetype = 'self_def'
    ## path for save prediction results
    save_path = filepath.replace('.fa', '_res.csv')
    ## whether use a extended sequnece as input, 0 not extend (20nt), 1 extended (30nt)
    wt = 1  # no need to change
    ## input sequence length 0:20nt, 1:30nt
    seq_ty = 1  # no need to change
    ## constant, no need to change
    ens_ty = 0
    res_df = user_request(filepath, filetype, be, wt, seq_ty, ens_ty, save_path, weight_fold)
    exp_df = []
    save_exp = ''
    if exp_out != '':
        expect_idx = np.where((res[:, 1] == exp_out))[0]
        fiter_out = res[expect_idx, :]

        save_exp = save_path.replace('.csv', 'expect_out.csv')
        with open(save_exp, 'w', newline='') as ws:
            writer = csv.writer(ws)
            writer.writerow(['extend target', 'outcome', 'edit efficiency', 'outcome proportion', 'outcome frequecy',
                             'picking score'])
            writer.writerows(fiter_out)
        exp_df = {'extend_target':fiter_out[:, 0], 'outcome':fiter_out[:, 1],
              'edit efficiency':fiter_out[:, 2], 'outcome proportion':fiter_out[:, 3],
                 'outcome frequecy':fiter_out[:, 4], 'picking score':fiter_out[:, 5]}
    return res_df, save_path, exp_df, save_exp

if __name__ == '__main__':
    # three inputs:
    # Be -- BE type, string
    # fa_path -- .fa file path, string
    # exp_out -- expect_outcome_sequence, string
    Be = sys.argv[1]
    fa_path = sys.argv[2]
    exp_out = sys.argv[3]
    # Be = 'ABE'
    # fa_path = './test.fa'
    # exp_out = ''
    # four outputs
    # res_df: all results in dataframe
    # save_path: path for download res_df in .csv
    # exp_df: filtered results with exp_out in dataframe
    # save_exp: path for download exp_df in .csv
    res_df, save_path, exp_df, save_exp = IO(be=Be, pth=fa_path, exp_out=exp_out)
