import os
import json
#from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
from scipy import stats
import pandas as pd 
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import os.path
from Bio import SeqIO
import string
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from propy.AAComposition import CalculateAADipeptideComposition
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import *
from xgboost import XGBClassifier
from FC import FC_model
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier



def isEnglish(s):
    #return s.translate(None, string.punctuation).isalnum()
    return s.translate(str.maketrans('','',string.punctuation)).isalnum()


def load_seq(path):
    seq_set = set()
    fasta_sequences = SeqIO.parse(open(path),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        #if len(sequence) < 5:
        #    continue
        if len(sequence) > 50:
            continue
        flag = False
        for j in sequence:
            if not isEnglish(j):
                flag = True
                break
        if flag:
            continue
        seq_set.add(sequence.upper())
    return seq_set

pos_anti_set = load_seq('/home/fangping/fangpingwan/anti-inflammation/data/benchmarking-positive.txt')
neg_anti_set = load_seq('/home/fangping/fangpingwan/anti-inflammation/data/benchmarking-negative.txt')

print (len(pos_anti_set))
print (len(neg_anti_set))

ind_pos_anti_set = load_seq('/home/fangping/fangpingwan/anti-inflammation/data/Ind-positive.txt')
ind_neg_anti_set = load_seq('/home/fangping/fangpingwan/anti-inflammation/data/Ind-negative.txt')

print (len(ind_pos_anti_set))
print (len(ind_neg_anti_set))


num_p = 30 # 30 properties
max_len = 52 # maximun peptide length

word2idx, idx2word = make_vocab()

emb, AAindex_dict = AAindex('/home/fangping/fangpingwan/anti-inflammation/aaindex1.csv', word2idx)



from modlamp.descriptors import PeptideDescriptor
from modlamp.descriptors import GlobalDescriptor

aa_set = []
aa_set.append('A')
aa_set.append('R')
aa_set.append('N')
aa_set.append('D')
aa_set.append('C')
aa_set.append('Q')
aa_set.append('E')
aa_set.append('G')
aa_set.append('H')
aa_set.append('I')
aa_set.append('L')
aa_set.append('K')
aa_set.append('M')
aa_set.append('F')
aa_set.append('P')
aa_set.append('S')
aa_set.append('T')
aa_set.append('W')
aa_set.append('Y')
aa_set.append('V')

twomer_counter = 0
twomer_dict = {}
for i in aa_set:
    for j in aa_set:
        twomer_dict[i+j] = twomer_counter
        twomer_counter += 1

onemer_counter = 0
onemer_dict = {}
for i in aa_set:
    onemer_dict[i] = onemer_counter
    onemer_counter += 1

def gap_kmer(a_seq, g): 
    x = np.zeros(len(twomer_dict))
    for i in range(len(a_seq)):
        if i+g+1 >= len(a_seq):
            continue
        else:
            x[twomer_dict[a_seq[i]+a_seq[i+g+1]]] += 1
    return x

def protein_feature(a_seq, gap_num):
    
    aa_dim = len(AAindex_dict['R'])
    D = PeptideDescriptor(a_seq)
    D.count_ngrams([2, 1])
    x_2 = np.zeros(len(twomer_dict))
    x_1 = np.zeros(len(onemer_dict))
    x_aa = np.zeros(aa_dim)
    for key, value in D.descriptor.items():
        if len(key) == 2:
            x_2[twomer_dict[key]] = value
        if len(key) == 1:
            x_1[onemer_dict[key]] = value
        
    x_2 = x_2 #/ float(np.sum(x_2))
    x_1 = x_1 #/ float(np.sum(x_1))
        
    for aa in a_seq:
        if aa not in AAindex_dict:
            continue
        else:
            x_aa += AAindex_dict[aa]
            x_aa = x_aa #/ float(len(a_seq))

    x = x_1.reshape(-1).tolist() +  x_2.reshape(-1).tolist() +  x_aa.reshape(-1).tolist()

    for i in np.arange(gap_num):
        gapped_feat = gap_kmer(a_seq, g=i+1)
        x = x + gapped_feat.reshape(-1).tolist()


    #gap2 = gap_kmer(a_seq, g=2)#PSSM(a_seq, RECMindex_dict, 5)
    #gap1 = gap_kmer(a_seq, g=1)
    #gap3 = gap_kmer(a_seq, g=3)
    #gap4 = gap_kmer(a_seq, g=4)
    #gap5 = gap_kmer(a_seq, g=5)
    #gap6 = gap_kmer(a_seq, g=6)
    #gap7 = gap_kmer(a_seq, g=7)


    #x = x_1.reshape(-1).tolist() +  x_2.reshape(-1).tolist() +  x_aa.reshape(-1).tolist() + \
    #    gap1.reshape(-1).tolist() + gap2.reshape(-1).tolist() + gap3.reshape(-1).tolist() + gap4.reshape(-1).tolist() + gap5.reshape(-1).tolist() + gap6.reshape(-1).tolist() + gap7.reshape(-1).tolist()
    
    #x = xpssm.reshape(-1).tolist()
    #return np.array(x)

    desc =  GlobalDescriptor(a_seq)
    
    desc.length()
    x.append(desc.descriptor[0][0])
    desc.calculate_charge()
    x.append(desc.descriptor[0][0])
    desc.isoelectric_point()
    x.append(desc.descriptor[0][0])
    desc.aromaticity()
    x.append(desc.descriptor[0][0])
    desc.aliphatic_index()
    x.append(desc.descriptor[0][0])
    desc.hydrophobic_ratio()
    x.append(desc.descriptor[0][0])
    return np.array(x)


def count_feats(seqs, gap_num):
    X = []
    for a_seq in seqs:
        X.append(protein_feature(a_seq, gap_num))
    return np.array(X)









def train_model(X_train, Y_train, X_test, Y_test, batch_size=128, num_epoch=1000):

    model = FC_model(np.shape(X_train)[1], 32)

    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4, amsgrad=False) 
    #optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-6) 


    min_loss = 1000000000

    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    pos_idx = []
    neg_idx = []
    for i in range(len(Y_train)):
        if Y_train[i, 0] == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)
    print ('pos, neg', len(pos_idx), len(neg_idx))
    pos_idx_sample = np.random.choice(pos_idx, size=len(neg_idx), replace=True)


    for epoch in range(num_epoch):  
        #np.random.seed(100)
        model.train()

        X_train_sample = np.vstack([X_train[pos_idx_sample], X_train[neg_idx]])
        Y_train_smaple = np.vstack([np.ones((len(pos_idx_sample), 1)), np.zeros((len(neg_idx), 1))])

        train_len = len(X_train_sample)
        shuffle_index = np.arange(train_len)
        np.random.shuffle(shuffle_index)
        np.random.shuffle(shuffle_index)
        np.random.shuffle(shuffle_index)
        np.random.shuffle(shuffle_index)
        np.random.shuffle(shuffle_index)

        total_loss = []

        for i in range(int(train_len/batch_size)):
            latent = X_train_sample[shuffle_index[i*batch_size:(i+1)*batch_size]]
            latent = torch.FloatTensor(latent).cuda()
            Y_test_cpu = Y_train_smaple[shuffle_index[i*batch_size:(i+1)*batch_size]]
            Y_test_gpu = torch.FloatTensor(Y_test_cpu).cuda()

            optimizer.zero_grad()
            Y_pred = model(latent)

            loss = F.binary_cross_entropy_with_logits(Y_pred, Y_test_gpu)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()

        if (epoch+1) % 10 == 0:
            Y_pred_train = test_model(model, X_train)
            Y_pred_test = test_model(model, X_test)

            train_auc = roc_auc_score(Y_train, Y_pred_train)
            train_aupr = average_precision_score(Y_train, Y_pred_train)
            Y_pred_train[np.where(Y_pred_train>0.5)] = 1
            Y_pred_train[np.where(Y_pred_train<=0.5)] = 0
            train_acc = accuracy_score(Y_train, Y_pred_train)

            test_auc = roc_auc_score(Y_test, Y_pred_test)
            test_aupr = average_precision_score(Y_test, Y_pred_test)
            Y_pred_test[np.where(Y_pred_test>0.5)] = 1
            Y_pred_test[np.where(Y_pred_test<=0.5)] = 0
            test_acc = accuracy_score(Y_test, Y_pred_test)

            print ('Epoch', epoch+1)
            print ('train and test aucs', train_auc, test_auc)
            print ('train and test auprs', train_aupr, test_aupr)
            print ('train and test acc', train_acc, test_acc)

    return test_auc, test_aupr, test_acc, model

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel

def CV(X, Y, n_estimators=4096, max_depth=32, rep_num=1, fold_num=5):
    auc_list = []
    aupr_list = []
    acc_list = []
    for a_rep in range(rep_num):

        kf = StratifiedKFold(n_splits=fold_num, shuffle=True) #random_state=seed_list[a_rep])
        fold_counter = 0
        for train_index, test_index in kf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            fold_counter += 1

            #regr = ElasticNet(alpha=0.005)
            #regr.fit(X_train, Y_train)
            #Y_pred = regr.predict(X_test)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            #regr_pre = ExtraTreesClassifier(n_estimators=4096, max_depth=32, n_jobs=16)#RandomForestClassifier(n_estimators=2048, max_depth=None, n_jobs=16)
            #sfs = SelectFromModel(regr_pre)
            #sfs.fit(X_train, Y_train.reshape(-1))
            #X_train = sfs.transform(X_train)
            #X_test = sfs.transform(X_test)

            #clf = ExtraTreeClassifier()
            #regr = AdaBoostClassifier(n_estimators=1000)#, base_estimator = clf)
            #regr = RandomForestClassifier(n_estimators=4096, max_depth=32, n_jobs=16)
            regr = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)#RandomForestClassifier(n_estimators=2048, max_depth=None, n_jobs=16)#XGBClassifier(n_estimators=512, max_depth=4, n_jobs=16)

            regr.fit(X_train, Y_train.reshape(-1))
            #regr2.fit(X_train, Y_train.reshape(-1))

            Y_pred = regr.predict_proba(X_test)[:, 1]# + 0.5*regr2.predict_proba(X_test)[:, 1]


            auc = roc_auc_score(Y_test, Y_pred)
            aupr = average_precision_score(Y_test, Y_pred)
            Y_pred[np.where(Y_pred>0.5)] = 1
            Y_pred[np.where(Y_pred<=0.5)] = 0
            acc = accuracy_score(Y_test, Y_pred)

            auc_list.append(auc)
            aupr_list.append(aupr)
            acc_list.append(acc)

        #print ('rep num', a_rep)

    #print ('auc:', np.mean(auc_list))
    #print ('aupr:', np.mean(aupr_list))
    #print ('acc:', np.mean(acc_list))
    return np.mean(auc_list), np.mean(aupr_list), np.mean(acc_list)


vocab_size = len(word2idx)
"""
performance_dict = {}
for gap_num in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
    for n_estimators in [1024, 2048, 4098, 8192, 8192*2]:
        for max_depth in [32, 64, 128]:

            pos_anti_onehot = count_feats(pos_anti_set, gap_num)
            neg_anti_onehot = count_feats(neg_anti_set, gap_num)


            #print (pos_anti_onehot, neg_anti_onehot)
            #print (np.shape(pos_anti_onehot), np.shape(neg_anti_onehot))
            X = np.vstack([pos_anti_onehot, neg_anti_onehot])
            Y = np.vstack([np.ones((len(pos_anti_onehot), 1)), np.zeros((len(neg_anti_onehot), 1))])


            ind_pos_anti_onehot = count_feats(ind_pos_anti_set, gap_num)
            ind_neg_anti_onehot= count_feats(ind_neg_anti_set, gap_num)


            X_ind = np.vstack([ind_pos_anti_onehot, ind_neg_anti_onehot])
            Y_ind = np.vstack([np.ones((len(ind_pos_anti_onehot), 1)), np.zeros((len(ind_neg_anti_onehot), 1))])




            auc, aupr, acc = CV(X, Y, n_estimators, max_depth, rep_num=5, fold_num=5)
            print ('gap num:', gap_num, 'n_estimators:', n_estimators, 'max_depth:', max_depth)
            print ('cv auc:', auc)
            print ('cv aupr:', aupr)
            print ('cv acc:', acc)
            print ('=========')
            performance_dict[str(gap_num)+'&'+str(n_estimators)+'&'+str(max_depth)] = [auc, aupr, acc]

import pickle
f = open('tune_dict', 'wb')
pickle.dump(performance_dict, f)
f.close()




import pickle
f = open('tune_dict', 'rb')
performance_dict = pickle.load(f)
f.close()

#for key, value in performance_dict.items():
#    print (key, value)

max_auc = 0
best_gap = 0
best_n_estimators = 0
best_max_depth = 0
for key, value in performance_dict.items():
    if max_auc < value[1]:
        max_auc = value[1]
        best_gap, best_n_estimators, best_max_depth = key.split('&')
best_gap = int(best_gap)
best_n_estimators =int(best_n_estimators)
best_max_depth = int(best_max_depth)

print ('best', best_gap, best_n_estimators, best_max_depth)

pos_anti_onehot = count_feats(pos_anti_set, best_gap)
neg_anti_onehot = count_feats(neg_anti_set, best_gap)
X = np.vstack([pos_anti_onehot, neg_anti_onehot])
Y = np.vstack([np.ones((len(pos_anti_onehot), 1)), np.zeros((len(neg_anti_onehot), 1))])

ind_pos_anti_onehot = count_feats(ind_pos_anti_set, best_gap)
ind_neg_anti_onehot= count_feats(ind_neg_anti_set, best_gap)
X_ind = np.vstack([ind_pos_anti_onehot, ind_neg_anti_onehot])
Y_ind = np.vstack([np.ones((len(ind_pos_anti_onehot), 1)), np.zeros((len(ind_neg_anti_onehot), 1))])



print (np.shape(X))
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_ind = scaler.transform(X_ind)


#regr_pre = ExtraTreesClassifier(n_estimators=4096, max_depth=32, n_jobs=16)
#sfs = SelectFromModel(regr_pre)
#sfs.fit(X, Y.reshape(-1))
#X = sfs.transform(X)
#X_ind = sfs.transform(X_ind)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=3)

#sfs = SequentialFeatureSelector(knn, n_features_to_select=100)
#sfs = SelectKBest(score_func=f_classif, k=800)
from sklearn.decomposition import KernelPCA, PCA

#sfs = PCA(n_components=128)
#sfs.fit(X, Y.reshape(-1))
#X = sfs.transform(X)
#X_ind = sfs.transform(X_ind)


#clf = ExtraTreeClassifier()
#regr = AdaBoostClassifier(n_estimators=1000)#, base_estimator = clf)
#regr = RandomForestClassifier(n_estimators=4096, max_depth=32, n_jobs=16)
regr = ExtraTreesClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, n_jobs=-1)#RandomForestClassifier(n_estimators=2048, max_depth=None, n_jobs=16) #XGBClassifier(n_estimators=512, max_depth=4, n_jobs=16)
regr.fit(X, Y.reshape(-1))
#regr2.fit(X, Y.reshape(-1))

Y_pred = regr.predict_proba(X_ind)[:, 1] #+ 0.5*regr2.predict_proba(X_ind)[:, 1] 

auc = roc_auc_score(Y_ind, Y_pred)
aupr = average_precision_score(Y_ind, Y_pred)
Y_pred[np.where(Y_pred>0.5)] = 1
Y_pred[np.where(Y_pred<=0.5)] = 0
acc = accuracy_score(Y_ind, Y_pred)
mcc = matthews_corrcoef(Y_ind, Y_pred)


print ('ind auc:', auc)
print ('ind aupr:', aupr)
print ('ind acc', acc)
print ('ind mcc', mcc)

"""

import pickle
f = open('tune_dict', 'rb')
performance_dict = pickle.load(f)
f.close()

#for key, value in performance_dict.items():
#    print (key, value)

performance_list = []
for key, value in performance_dict.items():
    best_gap, best_n_estimators, best_max_depth = key.split('&')
    performance_list.append([int(best_gap), int(best_n_estimators), int(best_max_depth), value[0], value[2]])


performance_list.sort(key=lambda x: x[3])

ensemble_counter = 0
ensemble_num = 10
for i in performance_list[::-1]:
    best_gap, best_n_estimators, best_max_depth = i[0], i[1], i[2]
    print ('performance:', i[3], i[4])

    pos_anti_onehot = count_feats(pos_anti_set, best_gap)
    neg_anti_onehot = count_feats(neg_anti_set, best_gap)
    X = np.vstack([pos_anti_onehot, neg_anti_onehot])
    Y = np.vstack([np.ones((len(pos_anti_onehot), 1)), np.zeros((len(neg_anti_onehot), 1))])

    ind_pos_anti_onehot = count_feats(ind_pos_anti_set, best_gap)
    ind_neg_anti_onehot= count_feats(ind_neg_anti_set, best_gap)
    X_ind = np.vstack([ind_pos_anti_onehot, ind_neg_anti_onehot])
    Y_ind = np.vstack([np.ones((len(ind_pos_anti_onehot), 1)), np.zeros((len(ind_neg_anti_onehot), 1))])


    print (np.shape(X))
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_ind = scaler.transform(X_ind)

    regr = ExtraTreesClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, n_jobs=-1)#RandomForestClassifier(n_estimators=2048, max_depth=None, n_jobs=16) #XGBClassifier(n_estimators=512, max_depth=4, n_jobs=16)
    regr.fit(X, Y.reshape(-1))

    if ensemble_counter == 0:
        Y_pred = regr.predict_proba(X_ind)[:, 1]
    else:
        Y_pred += regr.predict_proba(X_ind)[:, 1]
    ensemble_counter += 1
    if ensemble_counter == ensemble_num:
        break

Y_pred = Y_pred / float(ensemble_num)


np.save('ind_pred_pos', Y_pred[:len(ind_pos_anti_onehot)])
np.save('ind_pred_neg', Y_pred[len(ind_pos_anti_onehot):])

auc = roc_auc_score(Y_ind, Y_pred)
aupr = average_precision_score(Y_ind, Y_pred)
Y_pred[np.where(Y_pred>0.5)] = 1
Y_pred[np.where(Y_pred<=0.5)] = 0
acc = accuracy_score(Y_ind, Y_pred)
mcc = matthews_corrcoef(Y_ind, Y_pred)


print ('ind auc:', auc)
print ('ind aupr:', aupr)
print ('ind acc', acc)
print ('ind mcc', mcc)



scaler_list = []
model_list = []
gap_list = []
ensemble_counter = 0
ensemble_num = 10
for i in performance_list[::-1]:
    best_gap, best_n_estimators, best_max_depth = i[0], i[1], i[2]


    pos_anti_onehot = count_feats(pos_anti_set, best_gap)
    neg_anti_onehot = count_feats(neg_anti_set, best_gap)
    X = np.vstack([pos_anti_onehot, neg_anti_onehot])
    Y = np.vstack([np.ones((len(pos_anti_onehot), 1)), np.zeros((len(neg_anti_onehot), 1))])

    ind_pos_anti_onehot = count_feats(ind_pos_anti_set, best_gap)
    ind_neg_anti_onehot= count_feats(ind_neg_anti_set, best_gap)
    X_ind = np.vstack([ind_pos_anti_onehot, ind_neg_anti_onehot])
    Y_ind = np.vstack([np.ones((len(ind_pos_anti_onehot), 1)), np.zeros((len(ind_neg_anti_onehot), 1))])


    X = np.vstack([X, X_ind])
    Y = np.vstack([Y, Y_ind])

    print (np.shape(X))
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    regr = ExtraTreesClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, n_jobs=-1)#RandomForestClassifier(n_estimators=2048, max_depth=None, n_jobs=16) #XGBClassifier(n_estimators=512, max_depth=4, n_jobs=16)
    regr.fit(X, Y.reshape(-1))


    scaler_list.append(scaler)
    model_list.append(regr)
    gap_list.append(best_gap)
    ensemble_counter += 1
    if ensemble_counter == ensemble_num:
        break

f = open('Anti_inflammatory_peptide_prediction_models2.pkl', 'wb')
pickle.dump([gap_list, scaler_list, model_list], f)
f.close()
