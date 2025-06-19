import os
import pickle
import json
from collections import OrderedDict
import random
import glob

import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from Bio.PDB import *
import deepchem
# import pickle as pe
from rdkit.Chem.rdchem import *
random.seed(42)

pk = deepchem.dock.ConvexHullPocketFinder()
#
import torch
import rdkit
from transformers import BertModel, BertTokenizer
import os

import numpy as np
from time import time
import gzip
import warnings
import pickle
warnings.filterwarnings("ignore")
from Bio.PDB import *
import Bio
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqUtils
from config import DefaultConfig
configs = DefaultConfig()


downloadFolderPath = './inputs/ProtBert_model/'

modelFolderPath = downloadFolderPath

modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')

configFilePath = os.path.join(modelFolderPath, 'config.json')

vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')



tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
model = BertModel.from_pretrained(modelFolderPath)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

# seq = ['MSREEVESLIQEVLEVYPEKARKDRNKHLAVNDPAVTQSKKCIISNKKSQPGLMTIRGCAYAGSKGVVWGPIKDMIHISHGPVGCGQYSRAGRRNYYIGTTGVNAFVTMNFTSDFQEKDIVFGGDKKLAKLIDEVETLFPLNKGISVQSECPIGLIGDDIESVSKVKGAELSKTIVPVRCEGFRGVSQSLGHHIANDAVRDWVLGKRDEDTTFASTPYDVAIIGDYNIGGDAWSSRILLEEMGLRCVAQWSGDGSISEIELTPKVKLNLVHCYRSMNYISRHMEEKYGIPWMEYNFFGPTKTIESLRAIAAKFDESIQKKCEEVIAKYKPEWEAVVAKYRPRLEGKRVMLYIGGLRPRHVIGAYEDLGMEVVGTGYEFAHNDDYDRTMKEMGDSTLLYDDVTGYEFEEFVKRIKPDLIGSGIKEKFIFQKMGIPFREMHSWDYSGPYHGFDGFAIFARDMDMTLNNPCWKKLQAPWEASQQVDKIKASYPLFLDQDYKDM',
#         'HLQSTPQNLVSNAPIAETAGIAEPPDDDLQARLNTLKKQ']
# smi_ch_ind = {"0": 'A',"1": 'B', "2": 'C', "3": 'D', "4": 'E', "5": 'F', "6": 'G',
#                "7": 'H', "8": 'I', "9": 'J', "10": 'K', "11": 'L', "12": 'M',
#                "13": 'N', "14": 'O', "15": 'P', "16": 'Q', "17": 'R', "18": 'S',
#                "19": 'T', "20": 'U', "21": 'V', "22": 'W', "23": 'X', "24": 'Y', "25": 'Z'}

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (10, 6, 5, 6, 1) --> total 28


def get_atom_feature(m):
    H = []
    for i in range(len(m)):
        H.append(atom_feature(m[i][0]))
    H = np.array(H)

    return H
#
#
def process_protein(pdb_file):
    m = Chem.MolFromPDBFile(pdb_file)
    am = GetAdjacencyMatrix(m)
    pockets = pk.find_pockets(pdb_file)
    n2 = m.GetNumAtoms()
    c2 = m.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)

        ami = am[np.array(idxs)[:, None], np.array(idxs)]
        H = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_matrix(ami)
        graph = dgl.DGLGraph(g)
        graph.ndata['h'] = torch.Tensor(H)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)
        binding_parts.append(binding_parts_atoms)

    constructed_graphs = dgl.batch(constructed_graphs)

    # amj = am[np.array(not_in_binding)[:, None], np.array(not_in_binding)]
    # not_binding_atoms = []
    # for item in not_in_binding:
    #     not_binding_atoms.append((m.GetAtoms()[item], d2[item]))
    # H = get_atom_feature(not_binding_atoms)
    # g = nx.convert_matrix.from_numpy_matrix(amj)
    # graph = dgl.DGLGraph(g)
    # graph.ndata['h'] = torch.Tensor(H)
    # graph = dgl.add_self_loop(graph)
    # constructed_graphs = dgl.batch([constructed_graphs, graph])
    return binding_parts, not_in_binding, constructed_graphs
from rdkit.Chem import AllChem


def get_pos_z(self, smile1, i):

    # print(smile1)
    m1 = rdkit.Chem.MolFromSmiles(smile1)

    if m1 is None:
        self.pass_list.append(i)
        self.pass_smiles.add(smile1)
        return None, None

    if m1.GetNumAtoms() == 1:
        self.pass_list.append(i)
        if m1.GetNumAtoms() == 1:
            self.pass_smiles.add(smile1)
        return None, None
    m1 = Chem.AddHs(m1)

    ignore_flag1 = 0
    ignore1 = False

    while AllChem.EmbedMolecule(m1) == -1:
        print('retry')
        ignore_flag1 = ignore_flag1 + 1
        if ignore_flag1 >= 10:
            ignore1 = True
            break
    if ignore1:
        self.pass_list.append(i)
        self.pass_smiles.add(smile1)
        return None, None
    AllChem.MMFFOptimizeMolecule(m1)
    m1 = Chem.RemoveHs(m1)
    m1_con = m1.GetConformer(id=0)

    pos1 = []
    for j in range(m1.GetNumAtoms()):
        pos1.append(list(m1_con.GetAtomPosition(j)))
    np_pos1 = np.array(pos1)
    ten_pos1 = torch.Tensor(np_pos1)

    z1 = []
    for atom in m1.GetAtoms():
        if self.atomType.__contains__(atom.GetSymbol()):
            z = self.atomType[atom.GetSymbol()]
        else:
            z = self.NOTINDICT
        z1.append(z)

    z1 = np.array(z1)
    z1 = torch.tensor(z1)
    return ten_pos1, z1


node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

zero = np.eye(2)[1]
one = np.eye(2)[0]
#________________________________________________________________________
# df = pd.read_csv("humanPDBnew.txt")
df = pd.read_csv("data/pdball.txt")
print(len(df['pdb_id'].unique()))

with open("data/Metz.txt", 'r') as fp:
# with open("humanesm.txt", 'r') as fp:
    train_raw = fp.read()

previous_pdb = ""
constructed_graphs = ""
raw_data = train_raw.split("\n")
random.shuffle(raw_data)


raw_data_train = raw_data[0: int(len(raw_data)*0.8)]
#raw_data_valid = raw_data[int(len(raw_data)*0.8): int(len(raw_data)*0.9)]
raw_data_test = raw_data[int(len(raw_data)*0.8): int(len(raw_data))]

del raw_data

del raw_data_train[332]  #ini   1
del raw_data_train[837]  #1
del raw_data_train[2776]  #1
del raw_data_train[3766]  #1

#del raw_data_valid[287]  #1


parser = PDBParser()
THIRD_ATOM = 'N'  # 'O'

def residue_distance(res1, res2):
    distance = []
    cnt = 0
    for atom1 in res1:
        for atom2 in res2:
            distance += [abs(atom1 - atom2)]
            cnt += 1
    distance = np.array(distance)
    dist_mean = distance.mean()
    dist_std = distance.std()
    if 'CA' in res1 and 'CA' in res2:
        dist_ca = abs(res1['CA'] - res2['CA'])
    else:
        dist_ca = dist_mean
    return dist_mean, dist_std, dist_ca

def residue_relative_angle(res1, res2):
    if 'CA' in res1 and THIRD_ATOM in res1 and 'C' in res1:
        v1 = res1['CA'].get_vector().get_array()
        v2 = res1[THIRD_ATOM].get_vector().get_array()
        v3 = res1['C'].get_vector().get_array()
        normal1 = np.cross((v2 - v1), (v3 - v1))
    else:
        k = list(res1)
        if len(k) == 1:
            normal1 = k[0].get_vector().get_array()
        else:
            raise
    normal1 = normal1 / np.linalg.norm(normal1)

    if 'CA' in res2 and THIRD_ATOM in res2 and 'C' in res2:
        v1 = res2['CA'].get_vector().get_array()
        v2 = res2[THIRD_ATOM].get_vector().get_array()
        v3 = res2['C'].get_vector().get_array()
        normal2 = np.cross((v2 - v1), (v3 - v1))
    else:
        k = list(res2)
        if len(k) == 1:
            normal2 = k[0].get_vector().get_array()
        else:
            raise
    normal2 = normal2 / np.linalg.norm(normal2)

    return np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))

def get_dist_and_angle_matrix(residues):
    size = len(residues)
    dist_mat = np.zeros([size, size, 3])
    angle_mat = np.zeros([size, size])
    for i in range(size):
        for j in range(i + 1, size):
            dist_mean, dist_std, dist_ca = residue_distance(residues[i], residues[j])
            angle = residue_relative_angle(residues[i], residues[j])

            dist_mat[i, j, 0] = dist_mean
            dist_mat[i, j, 1] = dist_std
            dist_mat[i, j, 2] = dist_ca

            dist_mat[j, i, 0] = dist_mean
            dist_mat[j, i, 1] = dist_std
            dist_mat[j, i, 2] = dist_ca

            angle_mat[i, j] = angle
            angle_mat[j, i] = angle

    return dist_mat, angle_mat
def add_edges_custom(G, neighborhood_indices, edge_features):
    # t1 = time()
    size = neighborhood_indices.shape[0]
    neighborhood_indices = neighborhood_indices.tolist()
    src = []
    dst = []
    temp_edge_features = []
    for center in range(size):
        src += neighborhood_indices[center]
        dst += [center] * (21 - 1)
        for nbr in neighborhood_indices[center]:
            temp_edge_features += [np.abs(edge_features[center, nbr])]
    if len(src) != len(dst):
        print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
        raise Exception
    G.add_edges(src, dst)
    G.edata['ex'] = torch.tensor(temp_edge_features)

edge_feat_mean = [31.83509173, 1.56021911] #calculated from trainset only
edge_feat_std = [16.79204272, 0.69076342]

def add_edges_custom2(G, neighborhood_indices, edge_features):
    # t1 = time()
    size = neighborhood_indices.shape[0]
    neighborhood_indices = neighborhood_indices.tolist()
    src = []
    dst = []
    temp_edge_features = []
    for center in range(size):
        src += neighborhood_indices[center]
        dst += [center] * (size - 1) #
        for nbr in neighborhood_indices[center]:
            temp_edge_features += [np.abs(edge_features[center, nbr])]
    if len(src) != len(dst):
        print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
        raise Exception
    G.add_edges(src, dst)
    G.edata['ex'] = torch.tensor(temp_edge_features)

edge_feat_mean = [31.83509173, 1.56021911] #calculated from trainset only
edge_feat_std = [16.79204272, 0.69076342]


atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18}
NOTINDICT = 19


from rdkit.Chem import AllChem
def get_pos_z(smile1):
    # print(smile1)
    m1 = Chem.MolFromSmiles(smile1)

    if m1 is None:
        # self.pass_list.append(i)
        set().add(smile1)
        return None, None
    #
    if m1.GetNumAtoms() == 1:
        # self.pass_list.append(i)
        if m1.GetNumAtoms() == 1:
            set().add(smile1)
        return None, None
    m1 = Chem.AddHs(m1)

    ignore_flag1 = 0
    ignore1 = False
    while AllChem.EmbedMolecule(m1) == -1:  
        print('retry')
        ignore_flag1 = ignore_flag1 + 1
        if ignore_flag1 >= 10:
            ignore1 = True
            break
    if ignore1:
        # self.pass_list.append(i)
        set().add(smile1)
        return None, None
    AllChem.MMFFOptimizeMolecule(m1)  
    m1 = Chem.RemoveHs(m1)
    m1_con = m1.GetConformer(id=0)

    pos1 = []
    for j in range(m1.GetNumAtoms()):
        pos1.append(list(m1_con.GetAtomPosition(j)))
    np_pos1 = np.array(pos1)
    ten_pos1 = torch.Tensor(np_pos1)

    z1 = []
    for atom in m1.GetAtoms():
        if atomType.__contains__(atom.GetSymbol()):
            z = atomType[atom.GetSymbol()]
        else:
            z = NOTINDICT
        z1.append(z)

    z1 = np.array(z1)
    z1 = torch.tensor(z1)
    return ten_pos1, z1


test_set = []
z = 1
for item in raw_data_test:
    print(z)
    z += 1
    try:
        # a = item.split(",")
        a = item.split()
        smile = a[0]
        sequence = a[1]

        # pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]

        # seqX = " ".join(str(k) for k in sequence)
        # ids = tokenizer.batch_encode_plus([seqX], add_special_tokens=True, pad_to_max_length=True)
        # input_ids = torch.tensor(ids['input_ids']).to(device)  # proemb
        # attention_mask = torch.tensor(ids['attention_mask']).to(device) 
        # # ids['attention_mask']  
        # with torch.no_grad():
        #     embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # embedding = embedding.cpu().numpy()
        # bertfeatures = []
        # for seq_num in range(len(embedding)):
        #     seq_len = (attention_mask[seq_num] == 1).sum()
        #     seq_emd = embedding[seq_num][1:seq_len - 1]
        #     bertfeatures.append(seq_emd)
       
        # bertfeatures = torch.Tensor(seq_emd)
        # # b = df2.loc[df2["uniprot_id"] == uniprot_id].index[0]
        # bertfeatures = bertfeatures[:500]
        # seq_len = bertfeatures.shape[0]
        #
        # if seq_len < 500:
        #     temp = np.zeros([500, bertfeatures.shape[1]])
        #     temp[:seq_len, :] = bertfeatures
        #     bertfeatures = temp
        #
        # bertfeatures = bertfeatures[np.newaxis, :, :]

        # smile = df1.loc[df1["cid"] == int(cid)]["smile"].item()
        # pdb_code = df2.loc[df2["uniprot_id"] == uniprot_id]["pdb_id"].item()[:-1]
        pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]
        protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1

        ppb = PPBuilder()

        protein_name = pdb_code
        # protein_name = '1n2c'
        chains = df.loc[df["sequence"] == sequence]["pdb_id"].item()[-1:]

        path = './pdbs/'+ protein_name + '.pdb'
        structure = parser.get_structure(protein_name, path)

    #提取各个链的氨基酸残基以及构建多肽序列
        model3d = structure[0]  # every structure object has only one model object in DeepPPISP's dataset
        pep_pdb = ''
        residues = []
        for chain_id in chains:
            for residue in model3d[chain_id].get_residues():
                residues += [residue]
            peptides = ppb.build_peptides(model3d[chain_id])
            pep_pdb += ''.join([str(pep.get_sequence()) for pep in peptides])

        pep_seq_from_res_list = ''
        i = 0
        total_res = 0
        temp = 0
        residues2 = []
        original_total_res = len(residues)
        while i < original_total_res:
            res = residues[i]
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                temp += 1
            else:
                pep_seq_from_res_list += protein_letters_3to1[res_name[0] + res_name[1:].lower()]
                residues2 += [residues[i]]
                total_res += 1
                if total_res == configs.max_sequence_length:
                    break
            i += 1

        dist_mat, angle_mat = get_dist_and_angle_matrix(residues2[:configs.max_sequence_length])
        # dist_mat.dtype = 'float32'
        # angle_mat.dtype = 'float32'
        graph_list = {}
        G = dgl.DGLGraph()
        G.add_nodes(DefaultConfig().max_sequence_length)
        neighborhood_indices = dist_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length, 0] \
                                   .argsort()[:, 1:21]
        if neighborhood_indices.max() > DefaultConfig().max_sequence_length - 1 or neighborhood_indices.min() < 0:
            print(neighborhood_indices.max(), neighborhood_indices.min())
            # raise
        edge_feat = np.array([
            dist_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length, 0],
            angle_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length]
        ])
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = (edge_feat - edge_feat_mean) / edge_feat_std  # standardize features

        add_edges_custom(G,
                         neighborhood_indices,
                         edge_feat
                         )
        graph_list = G

        constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")
        g = smiles_to_bigraph(smile, node_featurizer=node_featurizer)
        g = dgl.add_self_loop(g)
        #____drug3D____
        smile_pos_dict = {}
        smile_z_dict = {}
        pass_list = []
        pass_smiles = set()
        # word_dict = defaultdict(lambda: len(word_dict))
        if pass_smiles.__contains__(smile):
            pass_list.append(i)
            continue
        if smile_pos_dict.__contains__(smile):
            ten_pos1 = smile_pos_dict[smile]
            z1 = smile_z_dict[smile]
        else:
            ten_pos1, z1 = get_pos_z(smile)
            if ten_pos1 == None:
                continue
            else:
                smile_pos_dict[smile] = ten_pos1
                smile_z_dict[smile] = z1
        # ____drug3Ddist____
        np_pos1 = ten_pos1.numpy()
        dist_mat = np.zeros([len(z1), len(z1), 3])

        # __
        dist_mean = []
        dist_std = []
        dist_ptp = []
        for i in range(len(z1)):
            for j in range(i + 1, len(z1)):
                #         distance += [abs(np_pos1[i] - np_pos1[j])]
                dist_mean += [(abs(np_pos1[i] - np_pos1[j])).mean()]
                dist_std += [(abs(np_pos1[i] - np_pos1[j])).std()]
                dist_ptp += [np.ptp(abs(np_pos1[i] - np_pos1[j]))]
        k = 0
        for i in range(len(z1)):
            for j in range(i + 1, len(z1)):
                dist_mat[i, j, 0] = dist_mean[k]
                dist_mat[i, j, 1] = dist_std[k]
                dist_mat[i, j, 2] = dist_ptp[k]

                dist_mat[j, i, 0] = dist_mean[k]
                dist_mat[j, i, 1] = dist_std[k]
                dist_mat[j, i, 2] = dist_ptp[k]
                k += 1
                if k >= len(z1):
                    break
            else:
                continue
            break

        distmat_atom = dist_mat
        # distmat_atom = residue_atom(smile)
        # pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]
        graph_list1 = {}
        Gdist = dgl.DGLGraph()
        Gdist.add_nodes(100)
        neighborhood_indices = distmat_atom[:100, :100,
                               0] \
                                   .argsort()[:, 1:len(z1)]
        if neighborhood_indices.max() > 100 - 1 or neighborhood_indices.min() < 0:
            print(neighborhood_indices.max(), neighborhood_indices.min())
            # raise
        edge_feat = np.array([
            distmat_atom[:100, :100, 0],
            # angle_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length]
        ])
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = (edge_feat - edge_feat_mean) / edge_feat_std  # standardize features

        add_edges_custom2(Gdist,
                          neighborhood_indices,
                          edge_feat
                          )
        label = float(a[2])
        label = round(label, 2)
        test_set.append(((Gdist, g, G,constructed_graphs[2]), label))

        # val_set.append(((constructed_graphs[2], bertfeatures, G, sequence, g, smile), zero))

    except Exception as e:
        print(e)
        continue
with open(f'test.pkl', 'wb') as f:
    pickle.dump(test_set, f)


val_set = []
z = 1
for item in raw_data_valid:
    print(z)
    z += 1
    try:
        # a = item.split(",")
        a = item.split()
        smile = a[0]
        sequence = a[1]

        # pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]

        # seqX = " ".join(str(k) for k in sequence)
        # ids = tokenizer.batch_encode_plus([seqX], add_special_tokens=True, pad_to_max_length=True)
        # input_ids = torch.tensor(ids['input_ids']).to(device)  # proemb
        # attention_mask = torch.tensor(ids['attention_mask']).to(device) 
        # # ids['attention_mask']  
        # with torch.no_grad():
        #     embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # embedding = embedding.cpu().numpy()
        # bertfeatures = []
        # for seq_num in range(len(embedding)):
        #     seq_len = (attention_mask[seq_num] == 1).sum()
        #     seq_emd = embedding[seq_num][1:seq_len - 1]
        #     bertfeatures.append(seq_emd)
        
        # bertfeatures = torch.Tensor(seq_emd)
        # # b = df2.loc[df2["uniprot_id"] == uniprot_id].index[0]
        # bertfeatures = bertfeatures[:500]
        # seq_len = bertfeatures.shape[0]
        #
        # if seq_len < 500:
        #     temp = np.zeros([500, bertfeatures.shape[1]])
        #     temp[:seq_len, :] = bertfeatures
        #     bertfeatures = temp
        #
        # bertfeatures = bertfeatures[np.newaxis, :, :]

        # smile = df1.loc[df1["cid"] == int(cid)]["smile"].item()
        # pdb_code = df2.loc[df2["uniprot_id"] == uniprot_id]["pdb_id"].item()[:-1]
        pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]
        protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1

        ppb = PPBuilder()

        protein_name = pdb_code
        # protein_name = '1n2c'
        chains = df.loc[df["sequence"] == sequence]["pdb_id"].item()[-1:]

        path = './pdbs/'+ protein_name + '.pdb'
        structure = parser.get_structure(protein_name, path)

        model3d = structure[0]  # every structure object has only one model object in DeepPPISP's dataset
        pep_pdb = ''
        residues = []
        for chain_id in chains:
            for residue in model3d[chain_id].get_residues():
                residues += [residue]
            peptides = ppb.build_peptides(model3d[chain_id])
            pep_pdb += ''.join([str(pep.get_sequence()) for pep in peptides])

        pep_seq_from_res_list = ''
        i = 0
        total_res = 0
        temp = 0
        residues2 = []
        original_total_res = len(residues)
        while i < original_total_res:
            res = residues[i]
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                temp += 1
            else:
                pep_seq_from_res_list += protein_letters_3to1[res_name[0] + res_name[1:].lower()]
                residues2 += [residues[i]]
                total_res += 1
                if total_res == configs.max_sequence_length:
                    break
            i += 1

        dist_mat, angle_mat = get_dist_and_angle_matrix(residues2[:configs.max_sequence_length])
        # dist_mat.dtype = 'float32'
        # angle_mat.dtype = 'float32'
        graph_list = {}
        G = dgl.DGLGraph()
        G.add_nodes(DefaultConfig().max_sequence_length)
        neighborhood_indices = dist_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length, 0] \
                                   .argsort()[:, 1:21]
        if neighborhood_indices.max() > DefaultConfig().max_sequence_length - 1 or neighborhood_indices.min() < 0:
            print(neighborhood_indices.max(), neighborhood_indices.min())
            # raise
        edge_feat = np.array([
            dist_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length, 0],
            angle_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length]
        ])
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = (edge_feat - edge_feat_mean) / edge_feat_std  # standardize features

        add_edges_custom(G,
                         neighborhood_indices,
                         edge_feat
                         )
        graph_list = G

        constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")
        g = smiles_to_bigraph(smile, node_featurizer=node_featurizer)
        g = dgl.add_self_loop(g)
        #____drug3D____
        smile_pos_dict = {}
        smile_z_dict = {}
        pass_list = []
        pass_smiles = set()
        # word_dict = defaultdict(lambda: len(word_dict))
        if pass_smiles.__contains__(smile):
            pass_list.append(i)
            continue
        if smile_pos_dict.__contains__(smile):
            ten_pos1 = smile_pos_dict[smile]
            z1 = smile_z_dict[smile]
        else:
            ten_pos1, z1 = get_pos_z(smile)
            if ten_pos1 == None:
                continue
            else:
                smile_pos_dict[smile] = ten_pos1
                smile_z_dict[smile] = z1
        # ____drug3Ddist____
        np_pos1 = ten_pos1.numpy()
        dist_mat = np.zeros([len(z1), len(z1), 3])

        # __
        dist_mean = []
        dist_std = []
        dist_ptp = []
        for i in range(len(z1)):
            for j in range(i + 1, len(z1)):
                #         distance += [abs(np_pos1[i] - np_pos1[j])]
                dist_mean += [(abs(np_pos1[i] - np_pos1[j])).mean()]
                dist_std += [(abs(np_pos1[i] - np_pos1[j])).std()]
                dist_ptp += [np.ptp(abs(np_pos1[i] - np_pos1[j]))]
        k = 0
        for i in range(len(z1)):
            for j in range(i + 1, len(z1)):
                dist_mat[i, j, 0] = dist_mean[k]
                dist_mat[i, j, 1] = dist_std[k]
                dist_mat[i, j, 2] = dist_ptp[k]

                dist_mat[j, i, 0] = dist_mean[k]
                dist_mat[j, i, 1] = dist_std[k]
                dist_mat[j, i, 2] = dist_ptp[k]
                k += 1
                if k >= len(z1):
                    break
            else:
                continue
            break

        distmat_atom = dist_mat
        # distmat_atom = residue_atom(smile)
        # pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]
        graph_list1 = {}
        Gdist = dgl.DGLGraph()
        Gdist.add_nodes(100)
        neighborhood_indices = distmat_atom[:100, :100,
                               0] \
                                   .argsort()[:, 1:len(z1)]
        if neighborhood_indices.max() > 100 - 1 or neighborhood_indices.min() < 0:
            print(neighborhood_indices.max(), neighborhood_indices.min())
            # raise
        edge_feat = np.array([
            distmat_atom[:100, :100, 0],
            # angle_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length]
        ])
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = (edge_feat - edge_feat_mean) / edge_feat_std  # standardize features

        add_edges_custom2(Gdist,
                          neighborhood_indices,
                          edge_feat
                          )
        label = float(a[2])
        val_set.append(((Gdist, g, G,constructed_graphs[2]), label))

        # val_set.append(((constructed_graphs[2], bertfeatures, G, sequence, g, smile), zero))

    except Exception as e:
        print(e)
        continue
with open(f'valid.pkl', 'wb') as f:
    pickle.dump(val_set, f)


train_set = []
z = 1
for item in raw_data_train:
    print(z)
    z += 1
    try:
        # a = item.split(",")
        a = item.split()
        smile = a[0]
        sequence = a[1]

        # pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]

        # seqX = " ".join(str(k) for k in sequence)
        # ids = tokenizer.batch_encode_plus([seqX], add_special_tokens=True, pad_to_max_length=True)
        # input_ids = torch.tensor(ids['input_ids']).to(device)  # proemb
        # attention_mask = torch.tensor(ids['attention_mask']).to(device)  
        # # ids['attention_mask']  
        # with torch.no_grad():
        #     embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # embedding = embedding.cpu().numpy()
        # bertfeatures = []
        # for seq_num in range(len(embedding)):
        #     seq_len = (attention_mask[seq_num] == 1).sum()
        #     seq_emd = embedding[seq_num][1:seq_len - 1]
        #     bertfeatures.append(seq_emd)
        
        # bertfeatures = torch.Tensor(seq_emd)
        # # b = df2.loc[df2["uniprot_id"] == uniprot_id].index[0]
        # bertfeatures = bertfeatures[:500]
        # seq_len = bertfeatures.shape[0]
        #
        # if seq_len < 500:
        #     temp = np.zeros([500, bertfeatures.shape[1]])
        #     temp[:seq_len, :] = bertfeatures
        #     bertfeatures = temp
        #
        # bertfeatures = bertfeatures[np.newaxis, :, :]

        # smile = df1.loc[df1["cid"] == int(cid)]["smile"].item()
        # pdb_code = df2.loc[df2["uniprot_id"] == uniprot_id]["pdb_id"].item()[:-1]
        pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]
        protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1

        ppb = PPBuilder()

        protein_name = pdb_code
        # protein_name = '1n2c'
        chains = df.loc[df["sequence"] == sequence]["pdb_id"].item()[-1:]

        path = './pdbs/'+ protein_name + '.pdb'
        structure = parser.get_structure(protein_name, path)

        model3d = structure[0]  # every structure object has only one model object in DeepPPISP's dataset
        pep_pdb = ''
        residues = []
        for chain_id in chains:
            for residue in model3d[chain_id].get_residues():
                residues += [residue]
            peptides = ppb.build_peptides(model3d[chain_id])
            pep_pdb += ''.join([str(pep.get_sequence()) for pep in peptides])

        pep_seq_from_res_list = ''
        i = 0
        total_res = 0
        temp = 0
        residues2 = []
        original_total_res = len(residues)
        while i < original_total_res:
            res = residues[i]
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                temp += 1
            else:
                pep_seq_from_res_list += protein_letters_3to1[res_name[0] + res_name[1:].lower()]
                residues2 += [residues[i]]
                total_res += 1
                if total_res == configs.max_sequence_length:
                    break
            i += 1

        dist_mat, angle_mat = get_dist_and_angle_matrix(residues2[:configs.max_sequence_length])
        # dist_mat.dtype = 'float32'
        # angle_mat.dtype = 'float32'
        graph_list = {}
        G = dgl.DGLGraph()
        G.add_nodes(DefaultConfig().max_sequence_length)
        neighborhood_indices = dist_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length, 0] \
                                   .argsort()[:, 1:21]
        if neighborhood_indices.max() > DefaultConfig().max_sequence_length - 1 or neighborhood_indices.min() < 0:
            print(neighborhood_indices.max(), neighborhood_indices.min())
            # raise
        edge_feat = np.array([
            dist_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length, 0],
            angle_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length]
        ])
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = (edge_feat - edge_feat_mean) / edge_feat_std  # standardize features

        add_edges_custom(G,
                         neighborhood_indices,
                         edge_feat
                         )
        graph_list = G

        constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")
        g = smiles_to_bigraph(smile, node_featurizer=node_featurizer)
        g = dgl.add_self_loop(g)
        #____drug3D____
        smile_pos_dict = {}
        smile_z_dict = {}
        pass_list = []
        pass_smiles = set()
        # word_dict = defaultdict(lambda: len(word_dict))
        if pass_smiles.__contains__(smile):
            pass_list.append(i)
            continue
        if smile_pos_dict.__contains__(smile):
            ten_pos1 = smile_pos_dict[smile]
            z1 = smile_z_dict[smile]
        else:
            ten_pos1, z1 = get_pos_z(smile)
            if ten_pos1 == None:
                continue
            else:
                smile_pos_dict[smile] = ten_pos1
                smile_z_dict[smile] = z1
        # ____drug3Ddist____
        np_pos1 = ten_pos1.numpy()
        dist_mat = np.zeros([len(z1), len(z1), 3])

        # __
        dist_mean = []
        dist_std = []
        dist_ptp = []
        for i in range(len(z1)):
            for j in range(i + 1, len(z1)):
                #         distance += [abs(np_pos1[i] - np_pos1[j])]
                dist_mean += [(abs(np_pos1[i] - np_pos1[j])).mean()]
                dist_std += [(abs(np_pos1[i] - np_pos1[j])).std()]
                dist_ptp += [np.ptp(abs(np_pos1[i] - np_pos1[j]))]
        k = 0
        for i in range(len(z1)):
            for j in range(i + 1, len(z1)):
                dist_mat[i, j, 0] = dist_mean[k]
                dist_mat[i, j, 1] = dist_std[k]
                dist_mat[i, j, 2] = dist_ptp[k]

                dist_mat[j, i, 0] = dist_mean[k]
                dist_mat[j, i, 1] = dist_std[k]
                dist_mat[j, i, 2] = dist_ptp[k]
                k += 1
                if k >= len(z1):
                    break
            else:
                continue
            break

        distmat_atom = dist_mat
        # distmat_atom = residue_atom(smile)
        # pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()  # [:-1]
        graph_list1 = {}
        Gdist = dgl.DGLGraph()
        Gdist.add_nodes(100)
        neighborhood_indices = distmat_atom[:100, :100,
                               0] \
                                   .argsort()[:, 1:len(z1)]
        if neighborhood_indices.max() > 100 - 1 or neighborhood_indices.min() < 0:
            print(neighborhood_indices.max(), neighborhood_indices.min())
            # raise
        edge_feat = np.array([
            distmat_atom[:100, :100, 0],
            # angle_mat[:DefaultConfig().max_sequence_length, :DefaultConfig().max_sequence_length]
        ])
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = (edge_feat - edge_feat_mean) / edge_feat_std  # standardize features

        add_edges_custom2(Gdist,
                          neighborhood_indices,
                          edge_feat
                          )
        label = float(a[2])
        label = round(label, 2)
        train_set.append(((Gdist, g, G,constructed_graphs[2]), label))

        # val_set.append(((constructed_graphs[2], bertfeatures, G, sequence, g, smile), zero))

    except Exception as e:
        print(e)
        continue
with open(f'train.pkl', 'wb') as f:
    pickle.dump(train_set, f)