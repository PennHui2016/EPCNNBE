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

def pack_data(seqs, eff, op, por, of, st, en):
    target_oh, outcome_oh = onehot_encoding(seqs, st, en)
    encoded = onehot_encoding_dinu(seqs, st, en)
    eff = eff.reshape(eff.shape[0], 1)
    op = op.reshape(op.shape[0], 1)
    por = por.reshape(por.shape[0], 1)
    data = {'seqs': seqs, 'tar_oh': target_oh, 'oc_oh': outcome_oh, 'encoded': encoded, 'eff': eff, 'op': op,
            'por': por, 'of': of}
    return data


def load_dataset(type='abe', st=4, en=24):
    test_data = pd.read_csv(root + 'test_' + type + '.csv', header=0)
    test_seqs = np.column_stack((test_data['target'], test_data['outcome']))  # sequences of target and outcome
    test_eff = np.array(test_data['eff'], dtype='float')  # measured editing efficiency
    test_op = np.array(test_data['op'], dtype='float')  # measured editing relative outcome proportion
    test_por = np.array(test_data['por'], dtype='float')  # outcome exact proportion for each outcome
    test_of = np.array(test_data['of'], dtype='float')

    h1_data = pd.read_csv(root + type + '_endo_h1.csv', header=0)
    h1_seqs = np.column_stack((h1_data['target'], h1_data['outcome']))  # sequences of target and outcome
    h1_eff = np.array(h1_data['eff'], dtype='float')  # measured editing efficiency
    h1_op = np.array(h1_data['op'], dtype='float')  # measured editing relative outcome proportion
    h1_por = np.array(h1_data['por'], dtype='float')
    h1_of = np.array(h1_data['of'], dtype='float')

    hc_data = pd.read_csv(root + type + '_endo_hc.csv', header=0)
    hc_seqs = np.column_stack((hc_data['target'], hc_data['outcome']))  # sequences of target and outcome
    hc_eff = np.array(hc_data['eff'], dtype='float')  # measured editing efficiency
    hc_op = np.array(hc_data['op'], dtype='float')  # measured editing relative outcome proportion
    hc_por = np.array(hc_data['por'], dtype='float')
    hc_of = np.array(hc_data['of'], dtype='float')

    test = pack_data(test_seqs, test_eff, test_op, test_por, test_of, st, en)
    h1 = pack_data(h1_seqs, h1_eff, h1_op, h1_por, h1_of, st, en)
    hc = pack_data(hc_seqs, hc_eff, hc_op, hc_por, hc_of, st, en)
    return test, h1, hc

def get_all_outcome(in_seq, bet):
    in_nu = bet[0:1]
    tar_nu = bet[1:2]
    in_seq0 = copy.deepcopy(in_seq)
    ed_win = in_seq0[6:14]
    import re
    from itertools import combinations
    addr = [substr.start() for substr in re.finditer(in_nu, ed_win)]
    all_comb = []
    for i in range(len(addr)):
        all_comb.extend(list(combinations(addr, i + 1)))

    all_outcome = [in_seq]
    for i in range(len(all_comb)):
        ori_seq = copy.deepcopy(ed_win)
        list1 = list(ori_seq)
        list2 = list(in_seq0)
        ind = all_comb[i]
        for j in ind:
            list1[int(j)] = tar_nu

        list2[6:14] = list1
        in_seq0 = ''.join(list2)
        all_outcome.append(in_seq0)

    all_input = [in_seq for i in range(len(all_outcome))]

    return all_input, all_outcome

def get_all_outcome_common(dataset, bet, st, en):
    input_seqs = []
    outcomes = []
    common_ind = []
    ind_non_edited = []
    uni_seqs = np.unique(dataset['seqs'][:, 0])
    for seq in uni_seqs:
        in_seqs, out_seqs = get_all_outcome(seq, bet)
        outcomes.extend(out_seqs)
        input_seqs.extend(in_seqs)

    eff = np.zeros((len(input_seqs), 1))
    por = np.zeros((len(input_seqs), 1))
    op = np.zeros((len(input_seqs), 1))
    of = np.zeros((len(input_seqs), 1))
    seqs_all = np.column_stack((np.array([input_seqs]).T, np.array([outcomes]).T))
    for i in range(len(dataset['seqs'][:, 0])):
        seq = dataset['seqs'][i, 1]
        if seq == dataset['seqs'][i, 0]:
            ind_seq = np.where((seqs_all[:, 0] == seq))[0]
            eff[ind_seq, 0] = dataset['eff'][i, 0]
        ind = np.where((seqs_all[:, 1] == seq))[0]
        if len(ind) == 1:
            if seq == dataset['seqs'][i, 0]:
                if por[ind, 0] != 0:  # duplicate
                    continue
                ind_non_edited.append(ind)
            # eff[ind, 0] = dataset['eff'][i, 0]
            por[ind, 0] = dataset['por'][i, 0]
            op[ind, 0] = dataset['op'][i, 0]
            of[ind, 0] = dataset['of'][i]
            common_ind.append(ind)

    common_ind = np.array(common_ind).T
    ind_non_edited = np.array(ind_non_edited).T
    encoded = onehot_encoding_dinu(seqs_all, st, en)
    target_oh, outcome_oh = onehot_encoding(seqs_all, st, en)
    dataset_all = {'seqs': seqs_all, 'tar_oh': target_oh, 'oc_oh': outcome_oh, 'encoded': encoded, 'eff': eff,
                   'por': por, 'op': op, 'of': of,
                   'outcome_ind': common_ind[0], 'nonedited_ind': ind_non_edited[0]}

    return dataset_all

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


def get_eff_out(dataset, por_pred, be, name):
    ## all_performance
    # fre
    por_pred[np.where((por_pred[:, 0] < 0))[0], :] = 0
    por_pred[np.where((por_pred[:, 0] > 1))[0], :] = 1
    por_real = dataset['por'] # outcome sequence proportion
    eff_real = dataset['eff']

    seqs = dataset['seqs']
    targets = seqs[:, 0]
    outcomes = seqs[:, 1]
    uni_tar = np.unique(targets)
    effs = np.zeros((len(uni_tar), 3))
    eff_preds = np.zeros((len(targets), 1))
    out_pors = np.zeros((len(targets), 3))
    out_fres = np.zeros((len(targets), 3))

    for i in range(0, len(uni_tar)):
        ind = np.where((targets == uni_tar[i]))[0]
        i_por_r = por_real[ind, 0]
        i_por_p = por_pred[ind, 0] / sum(por_pred[ind, 0])
        i_out = outcomes[ind]

        ind1 = np.where((i_out != uni_tar[i]))[0]  # edited
        ind2 = np.where((i_out == uni_tar[i]))[0]  # unedited

        if len(ind1) == 0:
            effs[i, 1] = 0
            effs[i, 0] = 0
        else:
            effs[i, 1] = sum(i_por_p[ind1]) * 100
            effs[i, 0] = sum(i_por_r[ind1]) * 100

        effs[i, 2] = eff_real[ind[0]] * 100

        eff_preds[ind, 0] = effs[i, 1]

        i_por_r1 = i_por_r / (sum(i_por_r[ind1]) + 1E-14)
        i_por_p1 = i_por_p / (sum(i_por_p[ind1]) + 1E-14)

        i_por_r1[ind2] = i_por_r[ind2]
        i_por_p1[ind2] = i_por_p[ind2]

        out_pors[ind, 0] = i_por_r1
        out_pors[ind, 1] = i_por_p1
        out_pors[ind, 2] = i_por_r1

        fre_r0 = sum(i_por_r[ind1]) * i_por_r1
        fre_p = sum(i_por_p[ind1]) * i_por_p1
        fre_r1 = eff_real[ind[0]] * i_por_r1

        fre_r0[ind2] = i_por_r1[ind2] * 100
        fre_p[ind2] = i_por_p1[ind2] * 100
        fre_r1[ind2] = i_por_r1[ind2] * 100

        out_fres[ind, 0] = fre_r0
        out_fres[ind, 1] = fre_p
        out_fres[ind, 2] = fre_r1

    ########### compare DeepBE result
    por_real_kim = dataset['por'][dataset['outcome_ind'], :]
    eff_real_kim = dataset['eff'][dataset['outcome_ind'], :]

    por_pred_kim = por_pred[dataset['outcome_ind'], :]
    eff_pred_kim = eff_preds[dataset['outcome_ind'], :]

    seqs_kim = dataset['seqs'][dataset['outcome_ind'], :]

    targets_kim = seqs_kim[:, 0]
    outcomes_kim = seqs_kim[:, 1]
    uni_tar_k = np.unique(targets_kim)

    # read files containing sequences that can be predicted by DeepBE webserver
    comp_kim_web = pd.read_csv(root + 'compared_' + be + '_' + be + '_' + name + '.csv', header=None, sep=',').values
    uni_seq_kim_web = np.unique(comp_kim_web[:, 1])

    out_pors_k = np.zeros((0, 3))
    out_fres_k = np.zeros((0, 3))
    out_dbs_score_k = np.zeros((0, 3))

    pr_loc_fre = []  # with no edit
    spr_loc_fre = []

    pr_loc_op = []  # only edited
    spr_loc_op = []

    fres = np.zeros((0, 2))
    ops = np.zeros((0, 2))
    effs1 = np.zeros((0, 2))
    pss = np.zeros((0, 2))
    seq_out = np.zeros((0, 2))
    mse_loc = []
    uni_tar_k = uni_seq_kim_web
    for i in range(0, len(uni_tar_k)):
        ind = np.where((targets_kim == uni_tar_k[i]))[0]
        i_por_r = por_real_kim[ind, 0] # outcome sequence proportion
        i_por_p = por_pred_kim[ind, 0] /sum(por_pred_kim[ind, 0]) # outcome sequence proportion
        i_out = outcomes_kim[ind]
        seq_out = np.row_stack((seq_out, seqs_kim[ind,:]))

        ind1 = np.where((i_out != uni_tar_k[i]))[0]  # edited
        ind2 = np.where((i_out == uni_tar_k[i]))[0]  # unedited

        eff_i_r = eff_real_kim[ind[0]]
        if len(ind1) == 0:
            eff_i_p = 0
            eff_i_r = 0
        else:
            eff_i_p = sum(i_por_p[ind1]) * 100
            eff_i_r = sum(i_por_r[ind1]) * 100

        effs1 = np.row_stack((effs1, np.column_stack((eff_i_r, eff_i_p))))

        mse_loc.append(abs(eff_i_r - eff_i_p)) # efficiency gap

        op_i_r = i_por_r[ind1] / sum(i_por_r[ind1] + 1E-14) # outcome proportion for edited only
        op_i_p = i_por_p[ind1] / sum(i_por_p[ind1] + 1E-14) # outcome proportion for edited only

        op_cb = np.column_stack((op_i_r, op_i_p))
        ops = np.row_stack((ops, op_cb))

        fre_i_r = eff_i_r * i_por_r # outcome frequence including non edited
        fre_i_p = eff_i_p * i_por_p # outcome frequence including non edited
        fre_i_r[ind2[0]] = i_por_r[ind2[0]]*100
        fre_i_p[ind2[0]] = i_por_p[ind2[0]]*100

        fre_cb = np.column_stack((fre_i_r, fre_i_p))
        fres = np.row_stack((fres, fre_cb))

        # local frequency
        if len(ind) == 1:
            spr_loc_fre.append(1)
            pr_loc_fre.append(1)
        else:
            spr_loc_fre.append(stats.spearmanr(fre_i_r, fre_i_p)[0])
            pr_loc_fre.append(stats.pearsonr(fre_i_r, fre_i_p)[0])

        # local outcome proportion
        if len(ind1) <= 1:
            spr_loc_op.append(1)
            pr_loc_op.append(1)
        else:
            spr_loc_op.append(stats.spearmanr(op_i_r, op_i_p)[0])
            pr_loc_op.append(stats.pearsonr(op_i_r, op_i_p)[0])

        i_por_r1 = i_por_r / (sum(i_por_r[ind1]) + 1E-14)
        i_por_p1 = i_por_p / (sum(i_por_p[ind1]) + 1E-14)

        i_por_r1[ind2] = i_por_r[ind2]
        i_por_p1[ind2] = i_por_p[ind2]

        out_pors_i = np.column_stack((i_por_r1, i_por_p1, i_por_r1))
        out_pors_k = np.row_stack((out_pors_k, out_pors_i))

        fre_r0 = eff_i_r * i_por_r1
        fre_p = eff_i_p * i_por_p1
        fre_r1 = eff_i_r * i_por_r1

        dbs_r0 = eff_i_r * i_por_r1
        dbs_p = eff_i_p * i_por_p1
        dbs_r1 = eff_i_r * i_por_r1

        dbs_r0[ind2] = -1
        dbs_p[ind2] = -1
        dbs_r1[ind2] = -1

        fre_r0[ind2] = i_por_r1[ind2] * 100
        fre_p[ind2] = i_por_p1[ind2] * 100
        fre_r1[ind2] = i_por_r1[ind2] * 100

        out_fres_i = np.column_stack((fre_r0, fre_p, fre_r1))
        out_fres_k = np.row_stack((out_fres_k, out_fres_i))

        out_dbs_score_i = np.column_stack((dbs_r0, dbs_p, dbs_r1))
        out_dbs_score_k = np.row_stack((out_dbs_score_k, out_dbs_score_i))

        # picking score
        i_ps = np.zeros((len(ind), 2))

        for j in range(len(ind)):
            if j != ind2:
                ps = 2 * i_por_p[j] * (1 - (sum(i_por_p[ind1]) - i_por_p[j])) / (
                        i_por_p[j] + (1 - (sum(i_por_p[ind1]) - i_por_p[j])))
                ps0 = 2 * i_por_r[j] * (1 - (sum(i_por_r[ind1]) - i_por_r[j])) / (
                        i_por_r[j] + (1 - (sum(i_por_r[ind1]) - i_por_r[j])))
                i_ps[j, 0] = ps
                i_ps[j, 1] = ps0

        pss = np.row_stack((pss, i_ps))

    # correlations of dbs scores (defined in kim et al.'s paper, dbs = eff * proportion)
    spr_dbs = stats.spearmanr(out_dbs_score_k[np.where((out_dbs_score_k[:, 0] != -1))[0], 1],
                              out_dbs_score_k[np.where((out_dbs_score_k[:, 0] != -1))[0], 2])[0]
    pr_dbs = stats.pearsonr(out_dbs_score_k[np.where((out_dbs_score_k[:, 0] != -1))[0], 1],
                            out_dbs_score_k[np.where((out_dbs_score_k[:, 0] != -1))[0], 2])[0]

    # final correlations of picking scores
    spr_pss = stats.spearmanr(pss[:, 0], pss[:, 1])[0]
    pr_pss = stats.pearsonr(pss[:, 0], pss[:, 1])[0]

    # final correlations of editing efficiencies
    spr_eff1 = stats.spearmanr(effs1[:, 0], effs1[:, 1])[0]
    pr_eff1 = stats.pearsonr(effs1[:, 0], effs1[:, 1])[0]

    # final correlations of outcome frequencies (include non-editing outcomes)
    spr_fre = stats.spearmanr(fres[:, 0], fres[:, 1])[0]
    pr_fre = stats.pearsonr(fres[:, 0], fres[:, 1])[0]

    # final correlations of outcome proportons (without non-edited outcomes)
    spr_op = stats.spearmanr(ops[:, 0], ops[:, 1])[0]
    pr_op = stats.pearsonr(ops[:, 0], ops[:, 1])[0]

    corrs = [spr_eff1, pr_eff1, spr_fre, pr_fre, spr_op, pr_op, spr_pss, pr_pss, spr_dbs, pr_dbs]

    loc_res = np.column_stack((uni_tar_k, mse_loc, pr_loc_fre, spr_loc_fre, pr_loc_op, spr_loc_op))

    return corrs, loc_res, np.column_stack((seq_out,pss)), np.column_stack((seq_out,out_pors_k)), np.column_stack((seq_out,out_fres_k)), np.column_stack((seq_out,out_dbs_score_k)), np.column_stack((np.array(uni_tar_k),effs1)), np.column_stack((seq_out,fres)), ops

##ensemble get_eff_out used
def ensemble_res(data, preds, name, ms_ty, be):
    preds = np.array(preds).squeeze().T
    pre_test_ens = np.mean(preds, axis=1)

    Y_pred = np.array(pre_test_ens).reshape(len(pre_test_ens), 1)

    res, loc_res, pss, out_pors_k, out_fres_k, out_dbs_score_k, effs1, fres, ops = get_eff_out(data, Y_pred, be, name)

    return res, loc_res, pss, out_pors_k, out_fres_k, out_dbs_score_k, effs1, fres, ops

### final tool work with cpu only
def ensemble_test(be, seq_ty, ens_ty, test, name):
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
            te_s1 = torch.from_numpy(test['tar_oh'].astype(np.float32)).float()
            te_s2 = torch.from_numpy(test['oc_oh'].astype(np.float32)).float()

            pred_s = model(te_s1, te_s2)
            pre_score = pred_s.detach().to(torch.device('cpu')).numpy()
        Y_pred = np.array(pre_score).reshape(len(pre_score), 1)
        pres.append(Y_pred)
    test_res, loc_res, pss, out_pors_k, out_fres_k, out_dbs_score_k, effs1, fres, ops = ensemble_res(test, pres, name, ens_ty, be)
    out = [name, be, seq_ty, ens_ty]
    out.extend(test_res)
    save_file_name = 'res/gloal_corr.csv'
    flag = 0
    header = ['dataset_name', 'be', 'seq_ty', 'ens_ty', 'spr_eff', 'pr_eff', 'spr_fre', 'pr_fre', 'spr_op', 'pr_op', 'spr_pss', 'pr_pss', 'spr_dbs', 'pr_dbs']
    with open(save_file_name, 'a+', newline='') as ws:
        writer = csv.writer(ws)
        if flag == 0:
            writer.writerow(header)
            flag += 1
        writer.writerow(out)

    save_file_name = 'res/local_corr_' + name + be + '_' + str(ens_ty) + '_' + str(seq_ty) +'.csv'
    with open(save_file_name, 'w', newline='') as ws:
        writer = csv.writer(ws)
        writer.writerow(['target sequence', 'Effgap', 'pr_loc_fre', 'spr_loc_fre', 'pr_loc_op', 'spr_loc_op'])
        writer.writerows(loc_res)

    save_file_name = 'res/pss_' + name + be + '_' + str(
        ens_ty) + '_' + str(seq_ty) + '.csv'
    with open(save_file_name, 'w', newline='') as ws:
        writer = csv.writer(ws)
        writer.writerow(['target sequence', 'outcome sequence', 'predicted picking score', 'true picking score'])
        writer.writerows(pss)

    save_file_name = 'res/effs_' + name + be + '_' + str(
        ens_ty) + '_' + str(seq_ty) + '.csv'
    with open(save_file_name, 'w', newline='') as ws:
        writer = csv.writer(ws)
        writer.writerow(['target sequence', 'true efficiency', 'predicted efficiency'])
        writer.writerows(effs1)

    save_file_name = 'res/fres_' + name + be + '_' + str(
        ens_ty) + '_' + str(seq_ty) + '.csv'
    with open(save_file_name, 'w', newline='') as ws:
        writer = csv.writer(ws)
        writer.writerow(['target sequence', 'outcome sequence', 'true frequency', 'predicted frequency'])
        writer.writerows(fres)

    save_file_name = 'res/ops_' + name + be + '_' + str(
        ens_ty) + '_' + str(seq_ty) + '.csv'
    with open(save_file_name, 'w', newline='') as ws:
        writer = csv.writer(ws)
        writer.writerow(['target sequence', 'outcome sequence', 'true outcome proportion', 'predicted outcome proportion'])
        writer.writerows(ops)
    return test_res

if __name__ == '__main__':
    seed_torch()
    #####################
    ##
    ##  reproduce idenpendent tests
    ##
    ############
    ### proformance comparison with DeepBE
    #######
    for be in ['abe', 'cbe']:
        for ms_ty in [0]:
            for sl in [0, 1]:
                if sl == 0:
                    st = 4
                    en = 24
                elif sl == 1:
                    st = 0
                    en = 30

                if be == 'abe':
                    bet = 'AG'
                elif be == 'cbe':
                    bet = 'CT'

                test, h1, hc = load_dataset(be, st=st, en=en)
                test_out = get_all_outcome_common(test, bet, st, en)
                h1_out = get_all_outcome_common(h1, bet, st, en)
                hc_out = get_all_outcome_common(hc, bet, st, en)

                ensemble_test(be, sl, ms_ty, test_out, 'test')
                ensemble_test(be, sl, ms_ty, h1_out, 'h1')
                ensemble_test(be, sl, ms_ty, hc_out, 'hc')





