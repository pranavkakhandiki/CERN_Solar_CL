import numpy as np
import subprocess
import tqdm
import pandas as pd

import os
import os.path as osp

import glob

import h5py
import uproot

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader

import awkward as ak
import random

#singularity shell --bind /afs/cern.ch/user/p/pkakhand/public/CL/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif

#singularity shell --bind /eos/project/c/contrast/public/solar/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif

#source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos9-gcc11-opt/setup.sh

class CCV1(Dataset):
    r'''
        input: layer clusters

    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8):
        super(CCV1, self).__init__(root, transform)
        
        self.fill_data(max_events)

    def fill_data(self,max_events):
        counter = 0
        for fi,path in enumerate(self.raw_paths):

            for array in uproot.iterate("%s:ticlNtuplizer/simtrackstersCP"%path, ["stsCP_vertices_x", "stsCP_vertices_y", "stsCP_vertices_z", "stsCP_vertices_energy", "stsCP_vertices_multiplicity"], step_size=500):
                tmp_stsCP_vertices_x = array['stsCP_vertices_x']
                tmp_stsCP_vertices_y = array['stsCP_vertices_y']
                tmp_stsCP_vertices_z = array['stsCP_vertices_z']
                tmp_stsCP_vertices_energy = array['stsCP_vertices_energy']

                tmp_stsCP_vertices_multiplicity = array['stsCP_vertices_multiplicity']


                #Filtering all the points where energy percent < 50%
                energyPercent = 1/tmp_stsCP_vertices_multiplicity
                skim_mask_energyPercent = energyPercent > 0.5
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask_energyPercent]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask_energyPercent]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask_energyPercent]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask_energyPercent]

                
                
                skim_mask = []
                for e in tmp_stsCP_vertices_x:
                    if len(e) >= 2:
                        skim_mask.append(True)
                    else:
                        skim_mask.append(False)
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask]
                
                

                if counter == 0:
                    self.stsCP_vertices_x = tmp_stsCP_vertices_x
                    self.stsCP_vertices_y = tmp_stsCP_vertices_y
                    self.stsCP_vertices_z = tmp_stsCP_vertices_z
                    self.stsCP_vertices_energy = tmp_stsCP_vertices_energy
                else:
                    self.stsCP_vertices_x = ak.concatenate((self.stsCP_vertices_x,tmp_stsCP_vertices_x))
                    self.stsCP_vertices_y = ak.concatenate((self.stsCP_vertices_y,tmp_stsCP_vertices_y))
                    self.stsCP_vertices_z = ak.concatenate((self.stsCP_vertices_z,tmp_stsCP_vertices_z))
                    self.stsCP_vertices_energy = ak.concatenate((self.stsCP_vertices_energy,tmp_stsCP_vertices_energy))
                print(len(self.stsCP_vertices_x))
                counter += 1
                if len(self.stsCP_vertices_x) > max_events:
                    break
            if len(self.stsCP_vertices_x) > max_events:
                break


    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return len(self.stsCP_vertices_x)

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.root')))
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        edge_index = torch.empty((2,0), dtype=torch.long)
 
        lc_x = self.stsCP_vertices_x[idx]
        flat_lc_x = np.expand_dims(np.array(ak.flatten(lc_x)),axis=1)
        lc_y = self.stsCP_vertices_y[idx]
        flat_lc_y = np.expand_dims(np.array(ak.flatten(lc_y)),axis=1)
        lc_z = self.stsCP_vertices_z[idx]
        flat_lc_z = np.expand_dims(np.array(ak.flatten(lc_z)),axis=1)
        lc_e = self.stsCP_vertices_energy[idx]
        flat_lc_e = np.expand_dims(np.array(ak.flatten(lc_e)),axis=1)                                                                                   


        flat_lc_feats = np.concatenate((flat_lc_x,flat_lc_y,flat_lc_z,flat_lc_e),axis=-1)


        # Loop over number of calo particles to build positive and negative edges
        pos_edges = []
        neg_edges = []
        offset = 0
        idlc = 0

        for cp in range(len(lc_x)):
            n_lc_cp = len(lc_x[cp])

            
            # First the positive edge
            for lc in range(n_lc_cp):
                random_num_pos = int(random.uniform(offset, offset+n_lc_cp))
                random_num_neg = int(random.uniform(0, flat_lc_x.shape[0]))
                while random_num_neg >= offset and random_num_neg < offset+n_lc_cp:
                    random_num_neg = int(random.uniform(0, flat_lc_x.shape[0]))

                #while (idlc == random_num_pos):
                #    random_num_pos = int(random.uniform(offset, offset+n_lc_cp))
                pos_edges.append([idlc,random_num_pos])
                neg_edges.append([idlc,random_num_neg])
                idlc += 1
            offset += n_lc_cp

        x = torch.from_numpy(flat_lc_feats).float()
        x_lc = x
        x_pos_edge = torch.from_numpy(np.array(pos_edges))
        x_neg_edge = torch.from_numpy(np.array(neg_edges))       


        y = torch.from_numpy(np.array([0 for u in range(len(flat_lc_feats))])).float()
        return Data(x=x, edge_index=edge_index, y=y,
                        x_lc=x_lc, x_pe=x_pos_edge, x_ne=x_neg_edge)



##
##
## Training part
##
##

BATCHSIZE = 1

#FOR PHOTONS
#ipath = '/eos/project/c/contrast/public/solar/data/20230220_multi_photons/train' # therein, files should be in a subfolder raw/*root
#vpath = '/eos/project/c/contrast/public/solar/data/20230220_multi_photons/test' # therein, files should be in a subfolder raw/*root
#opath = '/eos/project/c/contrast/public/solar/models/20230220_multi_photons/' # therein, files should be in a subfolder raw/*root

#FOR PIONS
ipath = '/eos/project/c/contrast/public/solar/data/20230214_two_pions/train' # therein, files should be in a subfolder raw/*root
vpath = '/eos/project/c/contrast/public/solar/data/20230214_two_pions/test' # therein, files should be in a subfolder raw/*root
opath = '/eos/project/c/contrast/public/solar/models/20230214_two_pions/' # therein, files should be in a subfolder raw/*root

#FOR MULTI-PHOTONS
#ipath = '/eos/project/c/contrast/public/solar/data/20230419_multi_photons/train' # therein, files should be in a subfolder raw/*root
#vpath = '/eos/project/c/contrast/public/solar/data/20230419_multi_photons/test' # therein, files should be in a subfolder raw/*root
#opath = '/eos/project/c/contrast/public/solar/models/20230419_multi_photons/' # therein, files should be in a subfolder raw/*root


print('Loading train dataset at', ipath)
data_train = CCV1(ipath,max_events=6000)

print('Loading test dataset at', vpath)
data_test = CCV1(vpath,max_events=4000)

train_loader = DataLoader(data_train, batch_size=BATCHSIZE,shuffle=False,
                          follow_batch=['x_lc'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=False,
                         follow_batch=['x_lc'])

#exit(1)

import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter


def contrastive_loss( start_all, end_all, temperature=0.1 ):
    xdevice = start_all.get_device()
    z_start = F.normalize( start_all, dim=1 )
    z_end = F.normalize( end_all, dim=1 )
    positives = F.cosine_similarity(z_start[:int(len(z_start)/2)],z_end[:int(len(z_end)/2)],dim=1)
    negatives = F.cosine_similarity(z_start[int(len(z_start)/2):],z_end[int(len(z_end)/2):],dim=1)
    nominator = torch.exp( positives / temperature )
    denominator = torch.exp( negatives / temperature )
    loss = torch.exp(-torch.log( torch.sum( nominator ) / torch.sum( denominator )))


    #print("Loss:",loss)

    return loss



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 32
        
        self.lc_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )
        
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )
        
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=24
        )
        
        self.output = nn.Sequential(
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 8)
#            nn.ELU(),
#            nn.Linear(16, 8)
        )
        
    def forward(self,
                x_lc,
                batch_lc):


        #print(x_lc)
        x_lc_enc = self.lc_encode(x_lc)
        
        # create a representation of LCs to LCs
        feats1 = self.conv1(x=(x_lc_enc, x_lc_enc), batch=(batch_lc, batch_lc))
        feats2 = self.conv2(x=(feats1, feats1), batch=(batch_lc, batch_lc))
        # similarly a representation LCs to Trackster
        feats3 = self.conv3(x=(feats2, feats2), batch=(batch_lc, batch_lc))

        batch = batch_lc
        out = self.output(feats3)

        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

cc = Net().to(device)

#cc.load_state_dict(torch.load(opath+"epoch-32.pt")['model'])
optimizer = torch.optim.Adam(cc.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(opath+"epoch-32.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#scheduler.load_state_dict(torch.load(opath+"epoch-32.pt")['lr'])


def train():
    cc.train()
    counter = 0

    total_loss = 0
    for data in tqdm.tqdm(train_loader):
        counter += 1

        data = data.to(device)
        optimizer.zero_grad()
        out = cc(data.x_lc,
                    data.x_lc_batch)
 
        start_pos = out[0][data.x_pe[:,0]]
        end_pos = out[0][data.x_pe[:,1]]
        start_neg = out[0][data.x_ne[:,0]]
        end_neg = out[0][data.x_ne[:,1]]

        start_all = torch.cat((start_pos,start_neg),0)
        end_all = torch.cat((end_pos,end_neg),0)

        loss = contrastive_loss(start_all,end_all,0.1)
        
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        #if counter > 1:
        #    break
        
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    cc.eval()
    total_loss = 0
    counter = 0
    for data in tqdm.tqdm(test_loader):
        counter += 1
        #print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        with torch.no_grad():
            out = cc(data.x_lc,
                       data.x_lc_batch)



            start_pos = out[0][data.x_pe[:,0]]
            end_pos = out[0][data.x_pe[:,1]]
            start_neg = out[0][data.x_ne[:,0]]
            end_neg = out[0][data.x_ne[:,1]]

            start_all = torch.cat((start_pos,start_neg),0)
            end_all = torch.cat((end_pos,end_neg),0)

            loss = contrastive_loss(start_all,end_all,0.1)

            total_loss += loss.item()
    return total_loss / len(test_loader.dataset)

best_val_loss = 1e9

all_train_loss = []
all_val_loss = []

loss_dict = {'train_loss': [], 'val_loss': []}

for epoch in range(1, 2):
    print(f'Training Epoch {epoch} on {len(train_loader.dataset)} jets')
    loss = train()
    scheduler.step()

    #exit(1)
    
    print(f'Validating Epoch {epoch} on {len(test_loader.dataset)} jets')
    loss_val = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    all_train_loss.append(loss)
    all_val_loss.append(loss_val)
    loss_dict['train_loss'].append(loss)
    loss_dict['val_loss'].append(loss_val)
    df = pd.DataFrame.from_dict(loss_dict)
    

    if not os.path.exists(opath):
        subprocess.call("mkdir -p %s"%opath,shell=True)

    df.to_csv("%s/"%opath+"/loss.csv")
    
    state_dicts = {'model':cc.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}

    torch.save(state_dicts, os.path.join(opath, f'epoch-{epoch}.pt'))

    if loss_val < best_val_loss:
        best_val_loss = loss_val

        torch.save(state_dicts, os.path.join(opath, 'best-epoch.pt'.format(epoch)))


print(all_train_loss)
print(all_val_loss)

