import argparse
import math
import os
import sys
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np
import lpips
from datasets.fundus_idrid_grading import get_train_dataset_fundus, get_test_dataset_fundus
from nets.model import Generator, Discriminator, TransEnc, CNNEnc

if __name__ == "__main__":
    
    #load model
    device = "cuda:7"
    transenc = TransEnc(256, 256).to(device)
    cnnsenc = CNNEnc(256, 256).to(device)
    ckpt_path = "/data/pycode/TransGAN/ckpts/090000.pt"
    transenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:7'})["trans"], strict=False)
    transenc.eval()
    cnnsenc.load_state_dict(torch.load(ckpt_path, map_location={'cuda:0': 'cuda:7'})["cnns"], strict=False)
    cnnsenc.eval()

    #load dataset
    train_loader = torch.utils.data.DataLoader(get_train_dataset_fundus(), batch_size=16, shuffle=True,num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(get_test_dataset_fundus(), batch_size=16, shuffle=False, num_workers=0, pin_memory=True)


    #retrieval evaluation
    print('********************Build feature for trainset!********************')
    tr_label = torch.FloatTensor()
    tr_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(train_loader):
            tr_label = torch.cat((tr_label, label), 0)

            #latent_in_trans, latent_in_cnn = transenc(image.to(device)),  cnnsenc(image.to(device))
            #var_feat = torch.cat((latent_in_trans, latent_in_cnn), dim=-1)
            var_feat = cnnsenc(image.to(device))

            tr_feat = torch.cat((tr_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Extract feature for testset!********************')
    te_label = torch.FloatTensor()
    te_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            te_label = torch.cat((te_label, label), 0)

            #latent_in_trans, latent_in_cnn = transenc(image.to(device)),  cnnsenc(image.to(device))
            #var_feat = torch.cat((latent_in_trans, latent_in_cnn), dim=-1)
            var_feat = cnnsenc(image.to(device))

            te_feat = torch.cat((te_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.numpy(), tr_feat.numpy())
    te_label = te_label.numpy()
    tr_label = tr_label.numpy()

    for topk in [5, 10, 20]:
        mHRs = {0: [], 1: [], 2: [], 3: [], 4: []}  # Hit Ratio
        mHRs_avg = []
        mAPs = {0: [], 1: [], 2: [], 3: [], 4: []}  # mean average precision
        mAPs_avg = []
        # NDCG: lack of ground truth ranking labels
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i, :].tolist()), key=lambda x: x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = np.where(te_label[i, :] == 1)[0][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = np.where(tr_label[j, :] == 1)[0][0]
                if tr_idx == te_idx:  # hit
                    num_pos = num_pos + 1
                    mAP.append(num_pos / rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs[te_idx].append(np.mean(mAP))
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs[te_idx].append(0)
                mAPs_avg.append(0)
            mHRs[te_idx].append(num_pos / rank_pos)
            mHRs_avg.append(num_pos / rank_pos)
            sys.stdout.write('\r test set process: = {}'.format(i + 1))
            sys.stdout.flush()

        CLASS_NAMES = ['Normal', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
        # Hit ratio
        for i in range(len(CLASS_NAMES)):
            print('Fundus mHR of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mHRs[i])))
        print("Fundus Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        # average precision
        for i in range(len(CLASS_NAMES)):
            print('Fundus mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAPs[i])))
        print("Fundus Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))

    #python proj_fundus.py