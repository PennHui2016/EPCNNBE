import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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

root = './tests/'
weight_fold = './be_weights/'
#weight_fold = 'F:/HUI/base_editing/best/'

def get_all_outcome_single(seq, bet, wt):
    in_nu = bet[0:1]
    tar_nu = bet[1:2]
    in_seq0 = copy.deepcopy(seq)
    if wt == 0: # raw=20
        ed_win = in_seq0[2:10]
    elif wt == 1: #raw = 30
        ed_win = in_seq0[6: 14]
    import re
    from itertools import combinations
    addr = [substr.start() for substr in re.finditer(in_nu, ed_win)]
    all_comb = []
    for i in range(len(addr)):
        all_comb.extend(list(combinations(addr, i + 1)))

    all_outcome = [seq]
    for i in range(len(all_comb)):
        ori_seq = copy.deepcopy(ed_win)
        list1 = list(ori_seq)
        list2 = list(in_seq0)
        ind = all_comb[i]
        for j in ind:
            list1[int(j)] = tar_nu

        if wt == 0: # raw=20
            list2[2:10] = list1
        elif wt == 1:
            list2[6:14] = list1
        in_seq0 = ''.join(list2)
        all_outcome.append(in_seq0)

    all_input = [seq for i in range(len(all_outcome))]
    return all_input, all_outcome

def get_all_outcome_compare(seqs, bet, wt, real):
    input_seqs = []
    ind_non_edited = []
    common_ind = []
    outcomes = []
    uni_seqs = np.unique(seqs[:, 0])
    for seq in uni_seqs:
        in_seqs, out_seqs = get_all_outcome_single(seq, bet, wt)
        outcomes.extend(out_seqs)
        input_seqs.extend(in_seqs)

    op = np.zeros((len(input_seqs), 1))
    seqs_all = np.column_stack((np.array([input_seqs]).T, np.array([outcomes]).T))
    for i in range(len(seqs[:, 0])):
        seq = seqs[i, 1]
        ind = np.where((seqs_all[:, 1] == seq))[0]
        if len(ind) == 1:
            if seq == seqs[i, 0]:
                if op[ind, 0] != 0:  # duplicate
                    continue
                ind_non_edited.append(ind)

            op[ind, 0] = real[i]
            common_ind.append(ind)
    common_ind = np.array(common_ind).T
    ind_non_edited = np.array(ind_non_edited).T
    return seqs_all, op, common_ind, ind_non_edited

def onehot_encoding(seqs, st, en):
    encoded_seq_target = encode_seq(seqs[:, 0], st, en)
    encoded_seq_outcome = encode_seq(seqs[:, 1], st, en)
    target_oh = np.array(encoded_seq_target)
    target_oh = target_oh.reshape(target_oh.shape[0], 1, target_oh.shape[1], target_oh.shape[2])
    outcome_oh = np.array(encoded_seq_outcome)
    outcome_oh = outcome_oh.reshape(outcome_oh.shape[0], 1, outcome_oh.shape[1], outcome_oh.shape[2])
    return target_oh, outcome_oh

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


def encode_binu(seqs1, seqs2, st, en):
    dinu = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    N = len(seqs1)
    L = en - st
    encoded_seqs = np.zeros((N, 16, L))
    for n in range(0, N):
        seq1 = seqs1[n][st:en]
        seq1 = seq1.upper()
        seq2 = seqs2[n][st:en]
        seq2 = seq2.upper()

        encoded_seq = np.zeros((16, L))
        for i in range(0, len(seq1)):
            ind = dinu.index(seq1[i] + seq2[i])
            encoded_seq[ind, i] = 1
        encoded_seqs[n, :, :] = encoded_seq
    return encoded_seqs


def onehot_encoding_dinu(seqs, st, en):
    tar_seqs = seqs[:, 0]
    out_seqs = seqs[:, 1]
    encode_seq = encode_binu(tar_seqs, out_seqs, st, en)
    encode_seq = encode_seq.reshape(encode_seq.shape[0], 1, encode_seq.shape[1], encode_seq.shape[2])
    return encode_seq

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

## no_kim dataset cross_dataset compare with pss with local
def get_eff_out(expect_seqs_all, real_seqs, expect_real_all, real, pred_all, seqs_por, real_por):
    pred_all[np.where((pred_all[:, 0] < 0))[0], :] = 0
    pred_all[np.where((pred_all[:, 0] > 1))[0], :] = 1
    fre_real = real
    por_real = real_por
    uni_tar = np.unique(real_seqs[:, 0])

    effs = np.zeros((len(uni_tar), 2))
    out_pors = np.zeros((len(pred_all), 2))
    out_fres = np.zeros((len(pred_all), 2))
    #eff_pred = np.zeros((len(uni_tar_real), 1))

    for i in range(0, len(uni_tar)):
        ind = np.where((expect_seqs_all[:, 0] == uni_tar[i]))[0]
        i_por_r = expect_real_all[ind, 0] #/ sum(expect_real_all[ind, 0])
        i_por_p = pred_all[ind, 0] / sum(pred_all[ind, 0] + 1E-14)
        i_out = expect_seqs_all[:, 1][ind]

        ind1 = np.where((i_out != uni_tar[i]))[0]  # edited
        ind2 = np.where((i_out == uni_tar[i]))[0]  # unedited

        if len(ind1) == 0:
            effs[i, 1] = 0 # pred
            effs[i, 0] = 0 # real
        else:
            effs[i, 1] = (1-i_por_p[ind2])
            effs[i, 0] = (1-i_por_r[ind2])

        i_por_r1 = i_por_r / (sum(i_por_r[ind1]) + 1E-14)
        i_por_p1 = i_por_p / (sum(i_por_p[ind1]) + 1E-14)

        i_por_r1[ind2] = -1
        i_por_p1[ind2] = -1

        out_pors[ind, 0] = i_por_r1
        out_pors[ind, 1] = i_por_p1

        fre_r = effs[i, 0] * i_por_r
        fre_r[ind2] = i_por_r[ind2]

        fre_p = effs[i, 1] * i_por_p
        fre_p[ind2] = i_por_p[ind2]

        out_fres[ind, 0] = fre_r
        out_fres[ind, 1] = fre_p

    eff_out = np.zeros((len(uni_tar), 2))
    fre_out = np.zeros((len(fre_real), 2))
    out_pors = np.zeros((len(fre_real), 2))

    pss = np.zeros((len(fre_real), 2))

    pr_loc_fre = []  # with no edit
    spr_loc_fre = []

    pr_loc_op = []  # only edited
    spr_loc_op = []

    fres = np.zeros((0, 2))
    ops = np.zeros((0, 2))

    mse_loc = []

    for i in range(0, len(uni_tar)):
        tar = uni_tar[i]
        ind_real = np.where((real_seqs[:, 0] == tar))[0]
        i_out = real_seqs[ind_real, 1]
        fre_r = fre_real[ind_real, :] # fre_real

        ind1 = np.where((i_out != tar))[0]  # edited
        ind2 = np.where((i_out == tar))[0]  # unedited
        ## non kim data, eff = 1-unedited_fre(our #pro), edited_fre =  eff * por;
        ## kim data, eff = sum(edited_fre), pro = #pro/sum(#pro), edited_fre = eff * por
        if len(ind2) == 1:
            eff_out[i, 0] = (1 - fre_r[ind2[0], 0])
        else:
            eff_out[i, 0] = 1

        #no_edit_seq = i_out[ind2]
        pre_fre = np.zeros((len(i_out), 1))
        for j in range(len(i_out)):
            idx = np.where((expect_seqs_all[:, 1] == i_out[j]))[0]
            if len(idx)==1:
                pre_fre[j, 0] = pred_all[idx, 0]
        pre_fre = pre_fre/(sum(pre_fre) + 1E-14)

        if len(ind2) == 1:
            eff_out[i, 1] = (1 - pre_fre[ind2[0], 0])
        else:
            eff_out[i, 1] = 1

        mse_loc.append(abs(eff_out[i, 1] - eff_out[i, 0]))

        fre_cb = np.column_stack((fre_r, pre_fre))
        fres = np.row_stack((fres, fre_cb))

        op_i_r = fre_r[ind1] / (sum(fre_r[ind1])+ 1E-14)
        op_i_p = pre_fre[ind1] / (sum(pre_fre[ind1]) + 1E-14)

        op_cb = np.column_stack((op_i_r, op_i_p))
        ops = np.row_stack((ops, op_cb))

        if len(ind_real) == 1:
            spr_loc_fre.append(1)
            pr_loc_fre.append(1)
        else:
            spr_loc_fre.append(stats.spearmanr(fre_r, pre_fre)[0])
            pr_loc_fre.append(stats.pearsonr(fre_r[:, 0], pre_fre[:, 0])[0])

        if len(ind1) <= 1:
            spr_loc_op.append(1)
            pr_loc_op.append(1)
        else:
            spr_loc_op.append(stats.spearmanr(op_i_r, op_i_p)[0])
            pr_loc_op.append(stats.pearsonr(op_i_r[:, 0], op_i_p[:, 0])[0])

        i_por_p1 = pre_fre/(sum(pre_fre[ind1]) + 1E-14)
        i_por_p1[ind2] = -1  # i_por_p[ind2]

        i_por_r1 = fre_r/(sum(fre_r[ind1]) + 1E-14)
        i_por_r1[ind2] = -1

        out_pors[ind_real, 0] = i_por_r1[:, 0]
        out_pors[ind_real, 1] = i_por_p1[:, 0]

        fre_p = eff_out[i, 1] * i_por_p1
        fre_p[ind2] = pre_fre[ind2]

        fre_out[ind_real, 0] = fre_r[:, 0]
        fre_out[ind_real, 1] = fre_p[:, 0]

        i_ps = np.zeros((len(ind_real), 2))

        for j in range(len(ind_real)):
            if j != ind2:
                ps = 2 * pre_fre[j] * (1 - (sum(pre_fre[ind1]) - pre_fre[j])) / (
                        pre_fre[j] + (1 - (sum(pre_fre[ind1]) - pre_fre[j])))
                ps0 = 2 * fre_r[j] * (1 - (sum(fre_r[ind1]) - fre_r[j])) / (
                        fre_r[j] + (1 - (sum(fre_r[ind1]) - fre_r[j])))
                i_ps[j, 0] = ps
                i_ps[j, 1] = ps0

        pss[ind_real, :] = i_ps

    spr_pss = stats.spearmanr(pss[:, 0], pss[:, 1])[0]
    pr_pss = stats.pearsonr(pss[:, 0], pss[:, 1])[0]

    spr_fre = stats.spearmanr(fres[:, 0], fres[:, 1])[0]
    pr_fre = stats.pearsonr(fres[:, 0], fres[:, 1])[0]

    spr_op = stats.spearmanr(ops[:, 0], ops[:, 1])[0]
    pr_op = stats.pearsonr(ops[:, 0], ops[:, 1])[0]

    res = [spr_fre, pr_fre, spr_op, pr_op, spr_pss, pr_pss]
    loc_res = np.column_stack((uni_tar, mse_loc, pr_loc_fre, spr_loc_fre, pr_loc_op, spr_loc_op))
    return res, loc_res

def non_dumplicate(dataset):
    dubseq = dataset['seqs'][:, 0] + dataset['seqs'][:, 1]
    uni_seq, inds = np.unique(dubseq, return_index=True)
    dataset_new = {'seqs': dataset['seqs'][inds, :], 'tar_oh': dataset['tar_oh'][inds, :],
                   'oc_oh': dataset['oc_oh'][inds, :], 'encoded': dataset['encoded'][inds, :],
                   'eff': dataset['eff'][inds, :], 'op': dataset['op'][inds, :], 'por': dataset['por'][inds, :],
                   'of': dataset['of'][inds]}
    return dataset_new


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

# split training data into 10 subsets for training 10 subregressors
def K_fold_split(dataset, K):
    seed_torch()
    logo = KFold(n_splits=K, shuffle=True)
    uni_seqs, indices = np.unique(dataset['seqs'][:, 0], return_index=True)
    inds_trains = []
    inds_tests = []
    for train_index, test_index in logo.split(uni_seqs):
        train_uni_seq = uni_seqs[train_index]
        test_uni_seq = uni_seqs[test_index]
        inds_train = []
        inds_test = []
        [inds_train.extend(np.where((dataset['seqs'][:, 0] == train_uni_seq[i]))[0]) for i in range(len(train_index))]
        [inds_test.extend(np.where((dataset['seqs'][:, 0] == test_uni_seq[i]))[0]) for i in range(len(test_index))]

        inds_train = np.array(inds_train, dtype='int')
        inds_test = np.array(inds_test, dtype='int')
        inds_trains.append(inds_train)
        inds_tests.append(inds_test)
    return inds_trains, inds_tests

def ensemble_res_seq(expect_seqs_all, real_seqs, expect_real_all, real, preds, seqs_por, real_por):
    preds = np.array(preds).squeeze().T
    pre_test_ens = np.mean(preds, axis=1)

    Y_pred = np.array(pre_test_ens).reshape(len(pre_test_ens), 1)

    res, loc_res = get_eff_out(expect_seqs_all, real_seqs, expect_real_all, real, Y_pred, seqs_por, real_por)

    out = res
    return out, loc_res

def ensemble_test_seq(be, seq_ty, ens_ty, in_seq, out_seq, real_all, seqs_all, real_seqs, real, seqs_por, real_por):
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
        weight_path = weight_fold + be + '_model_top' + str(i) + '_' + str(ens_ty) + '_' + str(seq_ty)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            te_s1 = torch.from_numpy(in_seq.astype(np.float32)).float()
            te_s2 = torch.from_numpy(out_seq.astype(np.float32)).float()

            pred_s = model(te_s1, te_s2)
            pre_score = pred_s.detach().to(torch.device('cpu')).numpy()
        Y_pred = np.array(pre_score).reshape(len(pre_score), 1)
        pres.append(Y_pred)
    #test_res = ensemble_res_seq(real, pres, 'test', ens_ty)
    test_res, loc_res = ensemble_res_seq(seqs_all, real_seqs, real_all, real, pres, seqs_por, real_por)

    return test_res, loc_res

def get_res_out_cross(seqs_fre, seqs_por, real_fre, pred_fre, real_por, pred_por):
    uni_tar = np.unique(seqs_fre[:, 0])
    effs = np.zeros((len(uni_tar), 2))
    out_pors = np.zeros((0, 4))

    pss = np.zeros((len(seqs_fre), 2))

    pr_loc_fre = []  # with no edit
    spr_loc_fre = []

    pr_loc_op = []  # only edited
    spr_loc_op = []

    mse_loc = []

    for i in range(0, len(uni_tar)):
        tar = uni_tar[i]
        ind = np.where((seqs_fre[:, 0] == tar))[0]
        i_r_fre = real_fre[ind, :]
        i_p_fre = pred_fre[ind, :]
        i_out = seqs_fre[ind, 1]

        ind1 = np.where((i_out != tar))[0]
        ind3 = np.where((i_out == tar))[0]
        ind2 = np.where((seqs_por[:, 0] == tar))[0]

        effs[i, 0] = sum(i_r_fre[ind1, :])
        effs[i, 1] = sum(i_p_fre[ind1, :])

        por_r = i_r_fre[ind1, 0] / (sum(i_r_fre[ind1, 0]) + 1E-14)
        por_p = i_p_fre[ind1, 0] / (sum(i_p_fre[ind1, 0]) + 1E-14)
        por_k_r = real_por[ind2, 0]
        por_k_p = pred_por[ind2, 0]

        por = np.column_stack((por_r, por_p, por_k_r, por_k_p))
        out_pors = np.row_stack((out_pors, por))

        mse_loc.append(abs(effs[i, 1] - effs[i, 0]))

        if len(ind) == 1:
            spr_loc_fre.append(1)
            pr_loc_fre.append(1)
        else:
            spr_loc_fre.append(stats.spearmanr(i_r_fre, i_p_fre)[0])
            pr_loc_fre.append(stats.pearsonr(i_r_fre[:, 0], i_p_fre[:, 0])[0])

        if len(ind1) <= 1:
            spr_loc_op.append(1)
            pr_loc_op.append(1)
        else:
            spr_loc_op.append(stats.spearmanr(por_k_r, por_k_p)[0])
            pr_loc_op.append(stats.pearsonr(por_k_r, por_k_p)[0])

        i_ps = np.zeros((len(ind), 2))

        for j in range(len(ind)):
            if j != ind3:
                ps = 2 * i_p_fre[j] * (1 - (sum(i_p_fre[ind1]) - i_p_fre[j])) / (
                        i_p_fre[j] + (1 - (sum(i_p_fre[ind1]) - i_p_fre[j])))
                ps0 = 2 * i_r_fre[j] * (1 - (sum(i_r_fre[ind1]) - i_r_fre[j])) / (
                        i_r_fre[j] + (1 - (sum(i_r_fre[ind1]) - i_r_fre[j])))
                i_ps[j, 0] = ps
                i_ps[j, 1] = ps0

        pss[ind, :] = i_ps

    spr_fre = stats.spearmanr(real_fre[:, 0], pred_fre[:, 0])[0]
    pr_fre = stats.pearsonr(real_fre[:, 0], pred_fre[:, 0])[0]

    spr_op = stats.spearmanr(real_por[:, 0], pred_por[:, 0])[0]
    pr_op = stats.pearsonr(real_por[:, 0], pred_por[:, 0])[0]

    spr_pss = stats.spearmanr(pss[:, 0], pss[:, 1])[0]
    pr_pss = stats.pearsonr(pss[:, 0], pss[:, 1])[0]

    loc_res = np.column_stack((uni_tar, mse_loc, pr_loc_fre, spr_loc_fre, pr_loc_op, spr_loc_op))

    res = [spr_fre, pr_fre, spr_op, pr_op, spr_pss, pr_pss]
    return res, loc_res

def res_others(be, method, da_name, real_index):
    if be == 'abe':
        BE = 'ABE'
    elif be == 'cbe':
        BE = 'CBE'

    pre_index = real_index + 1
    sheet_fre = method + '_' + da_name + '_' + 'freq_' + BE
    sheet_por = method + '_' + da_name + '_' + 'prop_' + BE
    datasets_fre = pd.read_excel(root + 'compare_datasets.xlsx', header=0, index_col=None,
                             sheet_name=sheet_fre).values
    datasets_por = pd.read_excel(root + 'compare_datasets.xlsx', header=0, index_col=None,
                                 sheet_name=sheet_por).values
    real_fre = np.array([datasets_fre[:, real_index]]).T
    pred_fre = np.array([datasets_fre[:, pre_index]]).T
    real_por = np.array([datasets_por[:, real_index]]).T
    pred_por = np.array([datasets_por[:, pre_index]]).T
    in_seq0 = datasets_fre[:, 1]
    out_seq0 = datasets_fre[:, 2]
    seqs_fre = np.column_stack((in_seq0, out_seq0))

    in_seq1 = datasets_por[:, 1]
    out_seq1 = datasets_por[:, 2]
    seqs_por = np.column_stack((in_seq1, out_seq1))

    res, loc_res = get_res_out_cross(seqs_fre, seqs_por, real_fre, pred_fre, real_por, pred_por)
    out = [be, method, da_name]
    out.extend(res)
    return out, loc_res

def other_res_all():
    fl = 0
    for be in ['abe', 'cbe']:
        method1 = 'BE-DICT'
        for da_name in ['own', 'arbab', 'song']:
            res, loc = res_others(be, method1, da_name, 3)
            save_file_name = './res/corr_cross_dataset.csv'

            with open(save_file_name, 'a+', newline='') as ws:
                writer = csv.writer(ws)
                if fl == 0:
                    writer.writerow(['be', 'method', 'dataset_name', 'spr_fre', 'pr_fre', 'spr_op', 'pr_op', 'spr_pss', 'pr_pss'])
                    fl += 1
                writer.writerow(res)

            save_file_name = './res/loc_res_cross_dataset_' + be + '_' + method1 + '_' + da_name + '.csv'
            with open(save_file_name, 'w', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerow(['target sequence', 'Effgap', 'pr_loc_fre', 'spr_loc_fre', 'pr_loc_op', 'spr_loc_op'])
                writer.writerows(loc)

        method2 = 'BE-HIVE'
        for da_name in ['marquart', 'own', 'song']:
            res, loc = res_others(be, method2, da_name, 3)
            save_file_name = './res/corr_cross_dataset.csv'
            with open(save_file_name, 'a+', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerow(res)

            save_file_name = './res/loc_res_cross_dataset_' + be + '_' + method2 + '_' + da_name + '.csv'
            with open(save_file_name, 'w', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerow(['target sequence', 'Effgap', 'pr_loc_fre', 'spr_loc_fre', 'pr_loc_op', 'spr_loc_op'])
                writer.writerows(loc)

        method3 = 'DEEPBE'
        for da_name in ['marquart', 'arbab', 'own']:
            if da_name == 'marquart':
                real_index = 3
            else:
                real_index = 4
            res, loc = res_others(be, method3, da_name, real_index)
            #save_path = save_fold
            #save_file_name = save_path + 'compare_res_others.csv'
            save_file_name = './res/corr_cross_dataset.csv'
            with open(save_file_name, 'a+', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerow(res)

            save_file_name = './res/loc_res_cross_dataset_' + be + '_' + method3 + '_' + da_name + '.csv'
            with open(save_file_name, 'w', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerow(['target sequence', 'Effgap', 'pr_loc_fre', 'spr_loc_fre', 'pr_loc_op', 'spr_loc_op'])
                writer.writerows(loc)

def compare_test(be, seq_ty, ens_ty):
    # BE-DICT_own_prop_ABE
    fl = 0
    method = 'DEEPBE'
    if (be == 'abe') | (be == 'abe1'):
        BE = 'ABE'
        bet = 'AG'
    elif be == 'cbe':
        BE = 'CBE'
        bet = 'CT'

    if seq_ty==1:
        mets = ['arbab', 'own']
    else:
        mets = ['arbab', 'marquart', 'own']

    for da_name in mets: # ['arbab', 'marquart', 'own']
    #for da_name in ['own']:
        if da_name == 'marquart':
            wt = 0
            real_index = 3
        else:
            wt = 1
            real_index = 4
        sheet_fre = method + '_' + da_name + '_' + 'freq_' + BE
        sheet_por = method + '_' + da_name + '_' + 'prop_' + BE
        datasets_fre = pd.read_excel(root + 'compare_datasets.xlsx', header=0,
                                     index_col=None,
                                     sheet_name=sheet_fre).values
        datasets_por = pd.read_excel(root + 'compare_datasets.xlsx', header=0,
                                     index_col=None,
                                     sheet_name=sheet_por).values
        in_seq0 = datasets_fre[:, 1]
        out_seq0 = datasets_fre[:, 2]
        seqs_fre = np.column_stack((in_seq0, out_seq0))

        in_seq1 = datasets_por[:, 1]
        out_seq1 = datasets_por[:, 2]
        seqs_por = np.column_stack((in_seq1, out_seq1))

        real_fre = np.array([datasets_fre[:, real_index]]).T
        real_por = np.array([datasets_por[:, real_index]]).T

        seqs_all, op, common_ind, ind_non_edited = get_all_outcome_compare(seqs_fre, bet, wt, real_fre)
        if (wt == 0) & (seq_ty == 0):
            target_oh, outcome_oh = onehot_encoding(seqs_all, 0, 20)
        elif (wt == 1) & (seq_ty == 0):
            target_oh, outcome_oh = onehot_encoding(seqs_all, 4, 24)
        elif (wt == 1) & (seq_ty == 1):
            target_oh, outcome_oh = onehot_encoding(seqs_all, 0, 30)

        res_our, loc = ensemble_test_seq(be, seq_ty, ens_ty, target_oh, outcome_oh, op, seqs_all, seqs_fre, real_fre, seqs_por, real_por)
        out = [be, 'EPCNNBE', da_name, seq_ty, ens_ty]
        out.extend(res_our)

        save_file_name = './res/corr_cross_dataset.csv'
        with open(save_file_name, 'a+', newline='') as ws:
            writer = csv.writer(ws)
            if fl==0:
                writer.writerow(
                    ['be', 'method', 'dataset_name',  'seq_ty', 'ens_ty', 'spr_fre', 'pr_fre', 'spr_op', 'pr_op', 'spr_pss', 'pr_pss'])
                fl += 1

            writer.writerow(out)

        save_file_name = './res/loc_res_cross_dataset' + '_' + be + '_EPCNNBE_' + da_name + '_' + str(seq_ty) + '_' + str(ens_ty) + '.csv'
        with open(save_file_name, 'w', newline='') as ws:
            writer = csv.writer(ws)
            writer.writerow(['target sequence', 'Effgap', 'pr_loc_fre', 'spr_loc_fre', 'pr_loc_op', 'spr_loc_op'])
            writer.writerows(loc)

if __name__ == '__main__':
    ###### reproduce cross dataset comparison test
    other_res_all()
    # seq len = 20
    compare_test('abe', 0, 0)
    compare_test('cbe', 0, 0)

    # seq len = 30
    compare_test('abe', 1, 0)
    compare_test('cbe', 1, 0)





