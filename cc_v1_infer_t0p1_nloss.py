import sklearn
import numpy as np
import random
from random import randrange
import subprocess
import tqdm
import pandas as pd
import uproot
import awkward as ak
import glob
import os
import os.path as osp


#from deepjet_geometric.datasets import CCV1
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader

import os
import argparse

##SINGULARITY SHELL
#singularity shell --bind /eos/project/c/contrast/public/solar/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif


##PION COMMAND
#python3 cc_v1_infer_t0p1_nloss.py --ipath /eos/project/c/contrast/public/solar/data/20230214_two_pions/val --mpath /eos/project/c/contrast/public/solar/models/20230214_two_pions/ --opath /eos/project/c/contrast/public/solar/output/20230214_two_pions/


#PHOTON COMMAND
#python3 cc_v1_infer_t0p1_nloss.py --ipath /eos/project/c/contrast/public/solar/data/20230220_multi_photons/val --mpath /eos/project/c/contrast/public/solar/models/20230220_multi_photons/ --opath /eos/project/c/contrast/public/solar/output/20230220_multi_photons/

#MULTI PHOTON COMMAND
#python3 cc_v1_infer_t0p1_nloss.py --ipath /eos/project/c/contrast/public/solar/data/20230419_multi_photons/val --mpath /eos/project/c/contrast/public/solar/models/20230419_multi_photons/ --opath /eos/project/c/contrast/public/solar/output/20230419_multi_photons/


#externally define this class
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
                
                #Change to '==2' to only get the two photon data
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
        #print("Number of LCs", len(flat_lc_x))
        #print(flat_lc_x.shape)

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
        
        #define x_counts = torch.([indices1, indices2])
        #print(np.array(lc_x[0]).shape)
        #print(np.array(lc_x).shape)
        #Send Lcx array
        x_counts = lc_x
        #just separated 
        y = torch.from_numpy(np.array([0 for u in range(len(flat_lc_feats))])).float()
        return Data(x=x, edge_index=edge_index, y=y,
                        x_lc=x_lc, x_pe=x_pos_edge, x_ne=x_neg_edge, x_counts = x_counts)



BATCHSIZE = 1

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--mpath', action='store', type=str, help='Path to model files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
args = parser.parse_args()


print('Loading test dataset at', args.ipath)
data_test = CCV1(args.ipath,max_events=100)

test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=False,
                         follow_batch=['x_lc'])

#exit(1)

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
#from torch_geometric.graphgym.models.pooling import global_add_pool
from torch.nn import Sequential, Linear
from torch_geometric.nn import DataParallel
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter

model_dir = args.mpath


def find_similarities( x_out, temperature ):
    xdevice = x_out.get_device()
    z_out = F.normalize( x_out, dim=1 )
    #print(z_out) 
    #positives = F.cosine_similarity(z_start[:int(len(z_start)/2)],z_end[:int(len(z_end)/2)],dim=1)
    #for i in range(len(z_out)):
    #    print(i)
    #    print(torch.bmm(z_out[i].view(len(z_out),4),z_out.view(4,len(z_out))))
    #print(torch.inner(z_out, z_out))
    # 
    #sims = F.cosine_similarity(z_out,z_out,dim=1)
    #sims = torch.exp( sims / temperature )
    #print(sims)

    sims = torch.tensordot(z_out, z_out, dims=([1],[1]))
    #print(sims)
    #print(nn.Softmax(sims))
    
    ids = []
    scores = []
    for lc in range(len(z_out)):
        #print(sims[lc])
        #print("Shape")
        #print(np.max(sims[lc].detach().cpu().numpy().shape))
        start = -2 
        idx = np.argsort(sims[lc].detach().cpu().numpy(), axis=0)[start]
        #print(idx)
        while (z_out[idx].detach().cpu().numpy() == z_out[lc].detach().cpu().numpy()).all():
            start -= 1
            idx = np.argsort(sims[lc].detach().cpu().numpy(), axis=0)[start]
        while idx < lc and ids[idx] == lc:
            start -= 1
            idx = np.argsort(sims[lc].detach().cpu().numpy(), axis=0)[start]
        ids.append(idx)
        scores.append(sims[lc][idx].detach().cpu().numpy())
        

    #try:
    #    print(z_out[405])
    #    print(z_out[623])
    #except:
    #    pass
    #denominator = torch.exp( negatives / temperature )
    #loss = -torch.log( torch.sum(nominator)  / torch.sum(denominator) )

        
    #print(ids)
    return ids,scores



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

cc.load_state_dict(torch.load(model_dir+"best-epoch.pt")['model'])
optimizer = torch.optim.Adam(cc.parameters(), lr=0.001)
optimizer.load_state_dict(torch.load(model_dir+"best-epoch.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
scheduler.load_state_dict(torch.load(model_dir+"best-epoch.pt")['lr'])



@torch.no_grad()
def test():
    cc.eval()
    total_loss = 0
    counter = 0

    out1 = []
    out2 = []
    out3 = []
    out4 = []
    out5 = []
    out6 = []
    out7 = []
    out8 = []

    lc_x = []
    lc_y = []
    lc_z = []
    lc_e = []
    soft_labels = []
    soft_scores = [] 
    gt_index = []

    for data in tqdm.tqdm(test_loader):
        counter += 1
        #print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        tmp_lc_feats = []
        with torch.no_grad():
            out = cc(data.x_lc,
                       data.x_lc_batch)
            
            counts = data.x_counts

            #print("len of first pion: ", len(counts[0][0]))
            #print("len of second pion: ", len(counts[0][1]))

            idx = find_similarities(out[0],0.1)
            tmp_lc_feats = data.x_lc.detach().cpu().numpy()
            tmp_lc_x = tmp_lc_feats[:,0]
            tmp_lc_y = tmp_lc_feats[:,1]
            tmp_lc_z = tmp_lc_feats[:,2]
            tmp_lc_e = tmp_lc_feats[:,3]
            tmp_soft_label = idx[0]
            tmp_soft_score = idx[1]

            
            temp = (F.normalize( out[0], dim=1 )).cpu()
            tmp_output_rep = np.array(temp)
            
            out1.append(tmp_output_rep[:, 0])
            out2.append(tmp_output_rep[:, 1])
            out3.append(tmp_output_rep[:, 2])
            out4.append(tmp_output_rep[:, 3])
            out5.append(tmp_output_rep[:, 4])
            out6.append(tmp_output_rep[:, 5])
            out7.append(tmp_output_rep[:, 6])
            out8.append(tmp_output_rep[:, 7])
            

            lc_x.append(tmp_lc_x)
            lc_y.append(tmp_lc_y)
            lc_z.append(tmp_lc_z)
            lc_e.append(tmp_lc_e)
            soft_labels.append(tmp_soft_label)
            soft_scores.append(tmp_soft_score)
            
            gt_index.append(len(counts[0][0]))

            #print(tmp_lc_x)
            


            #total_loss += loss.item()

        if counter > 1000:
            break

    final_out1 = ak.Array(out1)
    final_out2 = ak.Array(out2)
    final_out3 = ak.Array(out3)
    final_out4 = ak.Array(out4)
    final_out5 = ak.Array(out5)
    final_out6 = ak.Array(out6)
    final_out7 = ak.Array(out7)
    final_out8 = ak.Array(out8)


    final_lc_x = ak.Array(lc_x)
    final_lc_y = ak.Array(lc_y)
    final_lc_z = ak.Array(lc_z)
    final_lc_e = ak.Array(lc_e)

    final_gt_index = ak.Array(gt_index)

    #print("ONE: ", output_rep.shape)
    #print("TWO: ",lc_x.shape)
    #print("ONE: ", type(final_lc_x))
    #print("TWO: ",type(final_output_rep))

    ofile = uproot.recreate('%s/outfile.root'%args.opath)
    ofile['events'] = { 'lc_x' : final_lc_x, 
                        'lc_y' : final_lc_y,
                        'lc_z' : final_lc_z,
                        'lc_e' : final_lc_e,
                        'soft_labels' : soft_labels,
                        'soft_scores' : soft_scores,
                        'out1': final_out1,
                        'out2': final_out2,
                        'out3': final_out3,
                        'out4': final_out4,
                        'out5': final_out5,
                        'out6': final_out6,
                        'out7': final_out7,
                        'out8': final_out8,
                        'gt_index': final_gt_index
                      }
    

    #ofile['events'].extend({'lc_x':final_lc_x})

    return total_loss / len(test_loader.dataset)

best_val_loss = 1e9

all_train_loss = []
all_val_loss = []

loss_dict = {'train_loss': [], 'val_loss': []}

for epoch in range(1, 2):
    
    print(f'Validating Epoch {epoch} on {len(test_loader.dataset)} jets')
    loss_val = test()

    
    #if not os.path.exists(args.opath):
    #    subprocess.call("mkdir -p %s"%args.opath,shell=True)

    #df.to_csv("%s/"%args.opath+"/loss.csv")
    
    #state_dicts = {'model':cc.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}

    #torch.save(state_dicts, os.path.join(args.opath, f'epoch-{epoch}.pt'))

    #if loss_val < best_val_loss:
    #    best_val_loss = loss_val

    #    torch.save(state_dicts, os.path.join(args.opath, 'best-epoch.pt'.format(epoch)))


