import numpy as np
from Bio.Seq import Seq
import math
import copy
import csv
import sys
import time

contents = ['GRC', 'exon', 'intron', 'utr3', 'utr5', 'cds', 'cdna', 'peptide']


def obtain_fasta(filepath):
    fasta = {}
    with open(filepath) as file_one:
        for line in file_one:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta:
                    fasta[active_sequence_name] = []
                continue
            sequence = line
            fasta[active_sequence_name].append(sequence)

    return fasta

def find_all(a_string, sub):
    result = []
    k = 0
    while k < len(a_string):
        k = a_string.find(sub, k)
        if k == -1:
            return result
        else:
            result.append(k)
            k += 1  # change to k += len(sub) to not search overlapping results
    return result

def extract_spacers_selfdef(fasta):
    fasta_content = {}
    keys = list(fasta.keys())
    key = keys[0]
    seqs = fasta[key]
    seq = ''
    for j in range(0, len(seqs)):
        seq = seq + seqs[j]
    fasta_content[key]=seq

    genoSeq = str.upper(fasta_content[key])
    pos_sen = find_all(genoSeq, 'GG')
    pos_ant = find_all(genoSeq, 'CC')
    spacer = []
    spacer_ex = []
    cut_Feas = np.zeros((1, 10))
    for i in range(0, len(pos_sen)):
        start_p = pos_sen[i] - 21
        end_p = pos_sen[i] - 2
        start_p_ex = pos_sen[i] - 25
        end_p_ex = pos_sen[i] + 4
        cut_Fea = np.zeros((1, 10))
        cut_Fea[0][6] = start_p
        cut_Fea[0][7] = end_p
        cut_Fea[0][8] = start_p_ex
        cut_Fea[0][9] = end_p_ex
        cut_Feas = np.row_stack((cut_Feas, cut_Fea))

        if start_p_ex >= 0 and end_p_ex < len(genoSeq):
            spacer.append(str(genoSeq[start_p:end_p + 1]))
            spacer_ex.append(str(genoSeq[start_p_ex:end_p_ex + 1]))

    spacer_anti = []
    spacer_anti_ex = []
    for i in range(0, len(pos_ant)):
        start_p = pos_ant[i] + 3
        end_p = pos_ant[i] + 22
        start_p_ex = pos_ant[i] - 3
        end_p_ex = pos_ant[i] + 26
        cut_Fea = np.zeros((1, 10))
        cut_Fea[0][6] = start_p
        cut_Fea[0][7] = end_p
        cut_Fea[0][8] = start_p_ex
        cut_Fea[0][9] = end_p_ex
        cut_Feas = np.row_stack((cut_Feas, cut_Fea))

        if start_p_ex >= 0 and end_p_ex < len(genoSeq):
            s_an = Seq(genoSeq[start_p:end_p + 1]).reverse_complement()
            s_an_e = Seq(genoSeq[start_p_ex:end_p_ex + 1]).reverse_complement()
            spacer_anti.append(str(s_an._data.decode('utf-8')))
            spacer_anti_ex.append(str(s_an_e._data.decode('utf-8')))

    spacer_all = spacer + spacer_anti
    spacer_ex_all = spacer_ex + spacer_anti_ex
    return spacer_all, spacer_ex_all, cut_Feas


def extract_spacers_geno(fasta):
    fasta_content = {}
    keys = list(fasta.keys())
    key = keys[0]
    seqs = fasta[key]
    seq = ''
    for j in range(0, len(seqs)):
        seq = seq + seqs[j]
    fasta_content[key]=seq

    genoSeq = str.upper(fasta_content[key])
    pos_sen = find_all(genoSeq, 'GG')
    pos_ant = find_all(genoSeq, 'CC')
    spacer = []
    spacer_ex = []
    cut_Feas = np.zeros((1, 10))
    for i in range(0, len(pos_sen)):
        start_p = pos_sen[i] - 21
        end_p = pos_sen[i] - 2
        start_p_ex = pos_sen[i] - 25
        end_p_ex = pos_sen[i] + 4
        cut_Fea = np.zeros((1, 10))
        cut_Fea[0][6] = start_p
        cut_Fea[0][7] = end_p
        cut_Fea[0][8] = start_p_ex
        cut_Fea[0][9] = end_p_ex

        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            #cut_Fea = np.zeros((1, 6))
            spacer.append(genoSeq[start_p:end_p + 1])
            spacer_ex.append(genoSeq[start_p_ex:end_p_ex + 1])
            cut_geno = pos_sen[i] - 4
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            cut_Feas = np.row_stack((cut_Feas, cut_Fea))
    #cut_fea_sen = cut_fea[1:len(cut_fea[:, 0]), :]

    spacer_anti = []
    spacer_anti_ex = []
    #cut_fea = np.zeros((1, 6))
    for i in range(0, len(pos_ant)):
        start_p = pos_ant[i] + 3
        end_p = pos_ant[i] + 22
        start_p_ex = pos_ant[i] - 3
        end_p_ex = pos_ant[i] + 26
        cut_Fea = np.zeros((1, 10))
        cut_Fea[0][6] = start_p
        cut_Fea[0][7] = end_p
        cut_Fea[0][8] = start_p_ex
        cut_Fea[0][9] = end_p_ex

        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            #cut_Fea = np.zeros((1, 6))
            s_an = Seq(genoSeq[start_p:end_p + 1]).reverse_complement()
            s_an_e = Seq(genoSeq[start_p_ex:end_p_ex + 1]).reverse_complement()
            spacer_anti.append(s_an._data)
            spacer_anti_ex.append(s_an_e._data)
            cut_geno = pos_ant[i] + 6
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            cut_Feas = np.row_stack((cut_Feas, cut_Fea))
    #cut_fea_anti = cut_fea[1:len(cut_fea[:, 0]), :]

    spacer_all = spacer + spacer_anti
    spacer_ex_all = spacer_ex + spacer_anti_ex
    #cut_fea_all = np.vstack((cut_fea_sen, cut_fea_anti))

    return spacer_all, spacer_ex_all, cut_Feas


def extract_spacers_ensembl(fasta):
    fasta_content = {}
    keys = list(fasta.keys())
    for i in range(0, len(keys)):
        key = keys[i]
        for t in range(0, len(contents)):
            if key.find(contents[t]) > 0:
                newkey = contents[t]
                if newkey not in fasta_content:
                    fasta_content[newkey] = []
        seqs = fasta[key]
        seq = ''
        for j in range(0, len(seqs)):
            seq = seq + seqs[j]
        fasta_content[newkey].append(seq)

    genoSeq = str.upper(fasta_content['GRC'][0])
    exon_pos = []
    intron_pos = []
    if 'exon' in fasta_content:
        exons = fasta_content['exon']
        exon_pos = np.zeros((len(exons), 2))
        for i in range(0, len(exons)):
            exon = str.upper(exons[i])
            start = int(genoSeq.find(exon) + 1)  # seqence start from 1 not 0
            end = (start + len(exon))
            exon_pos[i][0] = start
            exon_pos[i][1] = end - 1
    if len(exon_pos) > 0:
        Exon_pos = sorted(exon_pos, key=lambda row: row[0])
        exon_pos = np.zeros((len(Exon_pos), 2))
        for i in range(0, len(Exon_pos)):
            exon_pos[i][0] = Exon_pos[i][0]
            exon_pos[i][1] = Exon_pos[i][1]

    if 'intron' in fasta_content:
        introns = fasta_content['intron']
        intron_pos = np.zeros((len(introns), 2))
        for i in range(0, len(introns)):
            intron = str.upper(introns[i])
            start = int(genoSeq.find(intron) + 1)  # seqence start from 1 not 0
            end = int(start + len(intron))
            intron_pos[i][0] = start
            intron_pos[i][1] = end - 1
    if len(intron_pos) > 0:
        Intron_pos = sorted(intron_pos, key=lambda row: row[0])
        intron_pos = np.zeros((len(Intron_pos), 2))
        for i in range(0, len(Intron_pos)):
            intron_pos[i][0] = Intron_pos[i][0]
            intron_pos[i][1] = Intron_pos[i][1]

    pos_sen = find_all(genoSeq, 'GG')
    pos_ant = find_all(genoSeq, 'CC')

    spacer = []
    spacer_ex = []
    #cut_fea = np.zeros((1, 6))
    cut_Feas = np.zeros((1, 10))
    for i in range(0, len(pos_sen)):
        start_p = pos_sen[i] - 21
        end_p = pos_sen[i] - 2
        start_p_ex = pos_sen[i] - 25
        end_p_ex = pos_sen[i] + 4
        cut_Fea = np.zeros((1, 10))
        cut_Fea[0][6] = start_p
        cut_Fea[0][7] = end_p
        cut_Fea[0][8] = start_p_ex
        cut_Fea[0][9] = end_p_ex
        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            #cut_Fea = np.zeros((1, 6))
            spacer.append(genoSeq[start_p:end_p + 1])
            spacer_ex.append(genoSeq[start_p_ex:end_p_ex + 1])
            cut_geno = pos_sen[i] - 4
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            flag = 0
            exon_num = 0
            if len(exon_pos) > 0:
                for j in range(0, len(exon_pos[:, 0])):
                    if cut_geno >= exon_pos[j, 0] and cut_geno <= exon_pos[j, 1]:  # cut at exon

                        flag = 1
                        exon_num = j
            if flag == 1:
                if exon_num == 0:
                    cut_trans = cut_geno
                else:
                    cut_trans = cut_geno
                    for j in range(0, exon_num):
                        cut_trans = cut_trans - (intron_pos[j, 1] - intron_pos[j, 0] + 1)
                cut_trans_per = 100 * ((cut_trans - 1) / len(fasta_content['cdna'][0]))
                cut_Fea[0][2] = cut_trans
                cut_Fea[0][3] = cut_trans_per
                len_utr5 = len(fasta_content['utr5'][0])
                len_protein = len(fasta_content['peptide'][0])
                cut_pro = math.ceil(((cut_trans - len_utr5 - 1) / 3))
                cut_pro_per = 100 * cut_pro / len_protein

                if cut_pro_per > 100 or cut_pro_per < 0:
                    cut_pro = 0
                    cut_pro_per = 0

                cut_Fea[0][4] = cut_pro
                cut_Fea[0][5] = cut_pro_per

            cut_Feas = np.row_stack((cut_Feas, cut_Fea))

    #cut_fea_sen = cut_fea[1:len(cut_fea[:, 0]), :]

    spacer_anti = []
    spacer_anti_ex = []
    #cut_fea = np.zeros((1, 6))
    for i in range(0, len(pos_ant)):
        start_p = pos_ant[i] + 3
        end_p = pos_ant[i] + 22
        start_p_ex = pos_ant[i] - 3
        end_p_ex = pos_ant[i] + 26
        cut_Fea = np.zeros((1, 10))
        cut_Fea[0][6] = start_p
        cut_Fea[0][7] = end_p
        cut_Fea[0][8] = start_p_ex
        cut_Fea[0][9] = end_p_ex
        if start_p_ex > 0 and end_p_ex < len(genoSeq):
            #cut_Fea = np.zeros((1, 6))
            s_an = Seq(genoSeq[start_p:end_p + 1]).reverse_complement()
            s_an_e = Seq(genoSeq[start_p_ex:end_p_ex + 1]).reverse_complement()
            spacer_anti.append(s_an._data)
            spacer_anti_ex.append(s_an_e._data)
            cut_geno = pos_ant[i] + 6
            cut_geno_per = 100 * (cut_geno - 1) / len(genoSeq)
            cut_Fea[0][0] = cut_geno
            cut_Fea[0][1] = cut_geno_per
            flag = 0
            exon_num = 0
            if len(exon_pos) > 0:
                for j in range(0, len(exon_pos[:, 0])):
                    if cut_geno >= exon_pos[j, 0] and cut_geno <= exon_pos[j, 1]:  # cut at exon
                        flag = 1
                        exon_num = j

            if flag == 1:
                if exon_num == 0:
                    cut_trans = cut_geno
                else:
                    cut_trans = cut_geno
                    for j in range(0, exon_num):
                        cut_trans = cut_trans - (intron_pos[j, 1] - intron_pos[j, 0] + 1)
                cut_trans_per = 100 * ((cut_trans - 1) / len(fasta_content['cdna'][0]))
                cut_Fea[0][2] = cut_trans
                cut_Fea[0][3] = cut_trans_per
                len_utr5 = len(fasta_content['utr5'][0])
                len_protein = len(fasta_content['peptide'][0])
                cut_pro = math.ceil(((cut_trans - len_utr5 - 1) / 3))
                cut_pro_per = 100 * cut_pro / len_protein

                if cut_pro_per > 100 or cut_pro_per < 0:
                    cut_pro = 0
                    cut_pro_per = 0

                cut_Fea[0][4] = cut_pro
                cut_Fea[0][5] = cut_pro_per

            cut_Feas = np.row_stack((cut_Feas, cut_Fea))

    cut_Feas = cut_Feas[1:len(cut_Feas[:, 0]), :]

    spacer_all = spacer + spacer_anti
    spacer_ex_all = spacer_ex + spacer_anti_ex
    #cut_fea_all = np.vstack((cut_fea_sen, cut_fea_anti))

    return spacer_all, spacer_ex_all, cut_Feas


def read_fasta(filepath, filetype):

    fasta = obtain_fasta(filepath)
    if filetype == 'annotated':
        spacer_all, spacer_ex_all, cut_fea_all = extract_spacers_ensembl(fasta)
    elif filetype == 'genome':
        spacer_all, spacer_ex_all, cut_fea_all = extract_spacers_geno(fasta)
    else:
        spacer_all, spacer_ex_all, cut_fea_all = extract_spacers_selfdef(fasta)
        #cut_fea_all = []

    return spacer_all, spacer_ex_all, cut_fea_all

def get_all_outcome_single(seq, bet, wt):
    all_input = []
    all_outcome = []

    in_nu = bet[0:1]
    tar_nu = bet[1:2]
    in_seq0 = copy.deepcopy(seq)
    if wt == 0: # raw=20
        ed_win = in_seq0[2:10]
    elif wt == 1: #raw = 30
        ed_win = in_seq0[6: 14]

    num = ed_win.find(in_nu)
    if num < 0:
        return all_input, all_outcome
    else:
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
            # ori_seq[ind] = tar_nu
            if wt == 0: # raw=20
                list2[2:10] = list1
            elif wt == 1:
                list2[6:14] = list1
            in_seq0 = ''.join(list2)
            all_outcome.append(in_seq0)

        all_input = [seq for i in range(len(all_outcome))]
        # all_outcome = np.array([all_outcome]).T
    return all_input, all_outcome

# def get_all_outcome(in_seq, bet):
#     all_input = []
#     all_outcome = []
#     in_nu = bet[0:1]
#     tar_nu = bet[1:2]
#     in_seq0 = copy.deepcopy(in_seq)
#     ed_win = in_seq0[6:14]
#     num = ed_win.find(in_nu)
#     if num < 0:
#         return
#     else:
#         import re
#         from itertools import combinations
#         addr = [substr.start() for substr in re.finditer(in_nu, ed_win)]
#         all_comb = []
#         for i in range(len(addr)):
#             all_comb.extend(list(combinations(addr, i + 1)))
#
#         all_outcome = [in_seq]
#         for i in range(len(all_comb)):
#             ori_seq = copy.deepcopy(ed_win)
#             list1 = list(ori_seq)
#             list2 = list(in_seq0)
#             ind = all_comb[i]
#             for j in ind:
#                 list1[int(j)] = tar_nu
#             # ori_seq[ind] = tar_nu
#
#             list2[6:14] = list1
#             in_seq0 = ''.join(list2)
#             all_outcome.append(in_seq0)
#
#         all_input = [in_seq for i in range(len(all_outcome))]
#         # all_outcome = np.array([all_outcome]).T
#     return all_input, all_outcome

def ACBE_spacers(filepath, filetype, be, wt):
    spacer_all, spacer_ex_all, cut_fea_all = read_fasta(filepath, filetype)
    #spacer_ex_all = np.array(spacer_ex_all, dtype='char')
    if (be == 'abe') | (be == 'ABE'):
        bet = 'AG'
    elif (be == 'cbe') | (be == 'CBE'):
        bet = 'CT'

    in_seqs = []
    out_seqs = []
    if len(spacer_ex_all) > 0:
        for i in range(len(spacer_ex_all)):

            in_seq = str(spacer_ex_all[i])
            all_input, all_outcome = get_all_outcome_single(in_seq, bet, wt)
            in_seqs.extend(all_input)
            out_seqs.extend(all_outcome)

    return in_seqs, out_seqs, spacer_ex_all, cut_fea_all


