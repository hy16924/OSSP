"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('./pretrained/3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
len_crop = 128

# Directory containing mel-spectrograms
rootDir = '/data/hypark/VC/vcc/AUTO-VC2/autovc/mine/LibriSpeech/preprocess_dir/train_reconcat/train_reconcat'
# targetDir = './train'
# dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % rootDir)
# print(subdirList)
name = ['SF1', 'SM2', 'SF2', 'TF1', 'TF2', 'TM2', 'SF3', 'SM1','TM1' ,'TM3']
speakers = []
list_ = []
#for name in subdirList:
#    if "p3" in name:
#        list_.append(name)
print(name)
for speaker in name:# sorted(list_):
    if speaker == '.ipynb_checkpoints':
        continue
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    # print(os.path.join(dirName, speaker))
    _, _, fileList = next(os.walk(os.path.join(rootDir,speaker)))
    # print(fileList)
    # print(len(fileList))
    # make speaker embedding
    
    # assert len(fileList) >= num_uttrs
    # idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(len(fileList)):
        tmp = np.load(os.path.join(rootDir, speaker, fileList[i]))
        # candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        # while tmp.shape[0] < len_crop:
            # idx_alt = np.random.choice(candidates)
        # tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
        # candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        if tmp.shape[0] <= len_crop:
            print(i)
            continue
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'SF1andSM2.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

