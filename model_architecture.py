import random
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, GINConv, APPNPConv, SAGEConv
# from SAGE_unsupervised import SAGE
# from dgl.nn import TWIRLSConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
# from torch_geometric.nn.conv import SAGEConv
import torch
import torch
# import esm
from torch.autograd import Variable
from rdkit import Chem
import numpy as np
from layers import MultiHeadAttention
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from dgl import unbatch
from resnet_encoder import ResnetEncoderModel, ResnetEncoderTiny, ResnetEncoderSuperTiny
# from resnet18q import resnet18
from module import Conv1dLSTM
from graph_layers import HGANLayer
from collections import defaultdict
from rdkit import Chem
from models import edge_gat_layer as egret
from schnet_model import SchNet

# import edge_gat_layer as egret
config_dict = egret.config_dict
config_dict['feat_drop'] = 0.2
config_dict['edge_feat_drop'] = 0.1
config_dict['attn_drop'] = 0.2

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class DTIHGAT(nn.Module):

    def __init__(self):
        super(DTIHGAT, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(HGANLayer(1, 31, 31, 2, dropout=0.2))

        self.ligand_graph_conv = nn.ModuleList()

        self.ligand_graph_conv.append(HGANLayer(1, 74, 70, 2, dropout=0.2))
        self.ligand_graph_conv.append(HGANLayer(1, 70, 65, 2, dropout=0.2))
        self.ligand_graph_conv.append(HGANLayer(1, 65, 60, 2, dropout=0.2))
        self.ligand_graph_conv.append(HGANLayer(1, 60, 55, 2, dropout=0.2))
        self.ligand_graph_conv.append(HGANLayer(1, 55, 31, 2, dropout=0.2))
        # self.ligand_graph_conv.append(TAGConv(50, 31, 2))
        # self.ligand_graph_conv.append(TAGConv(45, 40, 2))
        # self.ligand_graph_conv.append(TAGConv(40, 31, 2))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))

        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)

        self.dropout = nn.Dropout(p=0.2) #0.2?
        # self.resnet = ResnetEncoderModel(1)
        # self.resnet = ResnetEncoderModel(0)
        # self.resnet = ResnetEncoderSuperTiny(1)
        # self.resnet = resnet18()
        #self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=self.dropout)
        self.bilstm = nn.LSTM(input_size=31, hidden_size=31, num_layers=1, bidirectional=True, dropout=0.5)
        # self.fc_in = nn.Linear(8680, 4340) #1922
        # self.fc_in = nn.Linear(48, 4340)  # 1922
        # self.fc_in = nn.Linear(80, 4340)  # supertiny
        # self.fc_in = nn.Linear(320, 4340)  # tiny
        self.fc_in = nn.Linear(512, 4340)  # resnet18
        self.fc_out = nn.Linear(4340, 1)
        # self.attention = MultiHeadAttention(62, 62, 2)
        # self.attention = MultiHeadAttention(16, 16, 2)
        # self.attention = MultiHeadAttention(64, 64, 2)
        self.attention = MultiHeadAttention(512, 512, 2)  # resnet18

    #    self.W_s1 = nn.Linear(60, 45) #62
    #    self.W_s2 = nn.Linear(45, 30)

    # def attention_net(self, lstm_output):
    #    attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
    #    attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
    #    attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

    #    return attn_weight_matrix

    def forward(self, g):
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']

        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))

        pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[0], feature_protein).view(-1, 31)
        ligand_rep = pool_ligand(g[1], feature_smile).view(-1, 31)
        # sequence = []
        # for item in protein_rep:
        #    sequence.append(item.view(1, 31))
        #    sequence.append(ligand_rep)

        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, 31)
        # mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).cuda()
        mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).to(device)
        mask[0, sequence.size()[1]:140, :] = 0
        mask[0, :, sequence.size()[1]:140] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0, sequence.size()[1] - 1, sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        # sequence = sequence.permute(1, 0, 2)
        sequence = sequence.view(1, 1, 140, 31)
        # # h_0 = Variable(torch.zeros(2, 1, 31).cuda())
        # # c_0 = Variable(torch.zeros(2, 1, 31).cuda())
        # h_0 = Variable(torch.zeros(2, 1, 31).to(device))
        # c_0 = Variable(torch.zeros(2, 1, 31).to(device))
        #
        # output, _ = self.bilstm(sequence, (h_0, c_0))
        output = self.resnet(sequence)
        # output = output.permute(1, 0,  2)
        # output = output.view(1,16,3)#tput=(1,16,3,1)ResnetEncoderModel(1)
        # output = output.view(1, 16, 5)#，tput=(1,16,5,1)ResnetEncoderSuperTiny(1)
        # output = output.view(1, 64, 5)  # ，tput=(1,16,5,1)ResnetEncoderSuperTiny(1)
        output = output.view(1, 512, 1)  # ，tput=(1,16,5,1)Resnet18
        output = output.permute(0, 2, 1)
        # out = self.attention(output, mask=mask)
        out = self.attention(output)
        # attn_weight_matrix = self.attention_net(output)
        # out = torch.bmm(attn_weight_matrix, output)
        out = F.relu(self.fc_in(out.view(-1, out.size()[1] * out.size()[2])))

        out = torch.sigmoid(self.fc_out(out))
        return out


class DTITAG(nn.Module):
    def __init__(self):
        super(DTITAG, self).__init__()

        # 蛋白质图卷积层
        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(SAGEConv(31, 31, aggregator_type='mean'))  # 添加aggregator_type参数

        # 配体图卷积层
        self.ligand_graph_conv = nn.ModuleList()
        self.ligand_graph_conv.append(SAGEConv(74, 70, aggregator_type='mean'))
        self.ligand_graph_conv.append(SAGEConv(70, 65, aggregator_type='mean'))
        self.ligand_graph_conv.append(SAGEConv(65, 60, aggregator_type='mean'))
        self.ligand_graph_conv.append(SAGEConv(60, 55, aggregator_type='mean'))
        self.ligand_graph_conv.append(SAGEConv(55, 31, aggregator_type='mean'))

        # sequence图卷积层
        self.sequence_graph_conv = nn.ModuleList()
        self.sequence_graph_conv.append(SAGEConv(31, 24, aggregator_type='mean'))
        self.sequence_graph_conv.append(SAGEConv(24, 12, aggregator_type='mean'))
        self.sequence_graph_conv.append(SAGEConv(12, 1, aggregator_type='mean'))


        # 池化层
        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)
        # 线性变换层
        self.TtoT = nn.Linear(31, 32)

        self.dropout = 0.2
        self.weight_decay = 0.0002  #1e-5
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # ResNet编码器
        self.resnet = ResnetEncoderTiny(1)

        # 全连接层
        self.fc_in = nn.Linear(448, 4340)  # tiny

        # 输出层
        self.fc_out = nn.Linear(4340, 1)

        # 多头注意力机制
        self.attention = MultiHeadAttention(512, 512, 2)  # resnet18

        self.model1 = SchNet(energy_and_force=False, cutoff=10.0, num_layers=6,
                             hidden_channels=128, num_filters=128,
                             num_gaussians=50,
                             out_channels=31)  # 5->31   #2024.1.10:之前是31，改成了32
        self.atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18}
        self.NOTINDICT = 19
        self.fc1 = nn.Linear(5, 31)

        self.fc_d3d = nn.Sequential(
            nn.Linear(31, 31 * 2, bias=True),
            nn.PReLU(),
            nn.Linear(31 * 2, 31, bias=True)
        )  # 之前是32，改成31
        self.b = nn.Linear(1000, 31)
        self.bb = nn.Linear(100, 31)
        self.bert = nn.Linear(500, 31)
        # 对输入数据进行一维卷积操作。采用输入通道数为1024，输出通道数为32的卷积核，采用LeakyReLU作为激活函数，接着使用批归一化和dropout进行正则化
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=32,
                      kernel_size=7, stride=1,
                      padding=7 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm1d(num_features=32,
                           eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(.2)  # it was .2
        )
        # 多头EGRET层
        self.gat_layer = egret.MultiHeadEGRETLayer(
            in_dim=32,
            out_dim=32,
            edge_dim=2,
            num_heads=1,
            use_bias=False,
            merge='cat',
            config_dict=config_dict)

    def forward(self, g):

        dist = g[0]  # .ndata['h']
        feature_smile = g[1].ndata['h']
        G = g[2]
        feature_protein = g[3].ndata['h']

        # 应用图注意力层
        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))
        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[3], feature_protein))

        pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[3], feature_protein).view(-1, 31)
        ligand_rep = pool_ligand(g[1], feature_smile).view(-1, 31)

        protein_rep3=self.TtoT(protein_rep)
        ligand_rep3 = self.TtoT(ligand_rep)


        '''
        mask = torch.eye(500, dtype=torch.uint8).view(1, 500, 500).to(device)
        mask[0, protein_rep3.size()[1]:500, :] = 0
        mask[0, :, protein_rep3.size()[1]:500] = 0
        mask[0, :, protein_rep3.size()[1] - 1] = 1
        mask[0, protein_rep3.size()[1] - 1, :] = 1
        mask[0, protein_rep3.size()[1] - 1, protein_rep3.size()[1] - 1] = 0
        '''
        protein_rep3 = F.pad(input=protein_rep3, pad=(0, 0, 0, 100 - protein_rep3.size()[0]), mode='constant', value=0)
        # sequence = sequence.permute(1, 0, 2)
        protein_rep3 = protein_rep3.view(1, 100, 32)

        shapes = protein_rep3.data.shape
        '''
        features2, head_attn_scores = self.gat_layer(G, protein_rep3.view([shapes[0] * 500, 32]))
        features2 = features2.view([shapes[0], 500, 32])
        # print('features2.shape, features.shape:', features2.shape, features.shape)
        # z.repeat(1,2).view(shapes[0],2,shapes[1])
        protein_rep3 = protein_rep3.view(1, 32, 500)
        features2 = features2.view(1, 32, 500)
        featuresE = torch.cat((features2, protein_rep3), 2)
        # features = features.view([shapes[0], 500, 32*2])[t.nonzero(label_idx_onehot.view([shapes[0], 500])==1, as_tuple=True)]
        featuresE = featuresE.view(32, 1000)  # 对一个，view(1, 1000, 32)
        protein_rep_3D = self.b(featuresE)  # 32,31
        '''
        '''
        mask = torch.eye(100, dtype=torch.uint8).view(1, 100, 100).to(device)
        mask[0, ligand_rep3.size()[1]:100, :] = 0
        mask[0, :, ligand_rep3.size()[1]:100] = 0
        mask[0, :, ligand_rep3.size()[1] - 1] = 1
        mask[0, ligand_rep3.size()[1] - 1, :] = 1
        mask[0, ligand_rep3.size()[1] - 1, ligand_rep3.size()[1] - 1] = 0
        '''
        ligand_rep3 = F.pad(input=ligand_rep3, pad=(0, 0, 0, 131 - ligand_rep3.size()[1]), mode='constant', value=0)
        # sequence = sequence.permute(1, 0, 2)
        ligand_rep3 = ligand_rep3.view(1, 100, 32)

        '''
        features3, head_attn_scores = self.gat_layer(dist, ligand_rep3.view([shapes[0] * 100, 32]))
        # features3 = features3.view([shapes[0], 100, 32])

        features3 = features3.view(32, 100)
        ligand_rep_3D = self.bb(features3)

        # 应用 L1 正则化
        protein_rep_3D = protein_rep_3D + self.weight_decay * torch.sign(protein_rep_3D)
        ligand_rep_3D = ligand_rep_3D + self.weight_decay * torch.sign(ligand_rep_3D)

        sequence = torch.cat((ligand_rep_3D,  protein_rep_3D), dim=0).view(1, -1, 31)
        '''
        sequence = torch.cat((ligand_rep3,protein_rep3),dim=1)#.view(1,-1,31)

        '''
        # mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).cuda()
        mask = torch.eye(224, dtype=torch.uint8).view(1, 224, 224).to(device)
        mask[0, sequence.size()[1]:224, :] = 0
        mask[0, :, sequence.size()[1]:224] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0, sequence.size()[1] - 1, sequence.size()[1] - 1] = 0
        '''
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 224 - sequence.size()[1]), mode='constant', value=0)
        # sequence = sequence.permute(1, 0, 2)
        sequence = sequence.view(1, 1, 224, 32)
        # # c_0 = Variable(torch.zeros(2, 1, 31).cuda())
        # h_0 = Variable(torch.zeros(2, 1, 31).to(device))
        # c_0 = Variable(torch.zeros(2, 1, 31).to(device))
        #
        # output, _ = self.bilstm(sequence, (h_0, c_0))
        output = self.resnet(sequence)

        # output = output.permute(1, 0,  2)
        # output = output.view(1,16,3)#tput=(1,16,3,1)ResnetEncoderModel(1)
        # output = output.view(1, 16, 5)#，tput=(1,16,5,1)ResnetEncoderSuperTiny(1)
        # output = output.view(1, 64, 5)  # ，tput=(1,16,5,1)ResnetEncoderSuperTiny(1)
        output = output.view(1, 448, 1)  # ，tput=(1,16,5,1)Resnet18
        output = output.permute(0, 2, 1)
        #output = self.dropout_layer(output)
        out = output

        # 通过全连接层和激活函数处理最终输出
        out = F.relu(self.fc_in(out.view(-1, out.size()[1] * out.size()[2])))
        out = self.fc_out(out)

        # 计算 L2 正则化损失
        #l2_loss = 0
        #for param in self.parameters():
            #l2_loss += param.norm(2) ** 2
        #l2_loss *= self.weight_decay

        # 返回输出和 L2 正则化损失
        return out


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    # atom_dict = defaultdict(lambda: len(atom_dict))
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_dict = defaultdict(lambda: len(bond_dict))
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    # edge_dict = defaultdict(lambda: len(edge_dict))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Dense_Block(nn.Module):
    # def __init__(self, in_channels,n_filter,k, dropRate=0.0):
    def __init__(self, in_channels, out_channels, n_filter, k, dropRate=0.0):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(num_channels = in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=k)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=n_filter, kernel_size=k)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=n_filter, kernel_size=k)

    def forward(self, x):
        bn = self.bn(x)
        # conv1 = self.relu(self.conv1(bn))  houmian de dou gai le
        out = self.relu(self.conv1(bn))
        conv1 = F.dropout(out, p=self.droprate, training=self.training)
        out = self.relu(self.conv2(conv1))
        conv2 = F.dropout(out, p=self.droprate, training=self.training)
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        out = self.relu(self.conv3(c2_dense))
        conv3 = F.dropout(out, p=self.droprate, training=self.training)
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        out = SELayer(c3_dense)
        return out


class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out


class DTISAGE(nn.Module):

    def __init__(self):
        super(DTISAGE, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(SAGEConv(31, 31, 'mean'))

        self.ligand_graph_conv = nn.ModuleList()

        self.ligand_graph_conv.append(SAGEConv(74, 70, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(70, 65, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(65, 60, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(60, 55, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(55, 31, 'mean'))
        # self.ligand_graph_conv.append(TAGConv(50, 31, 2))
        # self.ligand_graph_conv.append(TAGConv(45, 40, 2))
        # self.ligand_graph_conv.append(TAGConv(40, 31, 2))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))
        # self.ligand_graph_conv.append(TAGConv(31, 31, 2, activation=F.relu))

        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)
        # self.dim = 64
        # self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        # self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        # self.embedding_xd = nn.Embedding(79, self.dim,padding_idx=0)#embed dim
        self.embed_smile = nn.Embedding(100, 100)
        # self.create_atoms = create_atoms()
        self.embed_fingerprint = nn.Embedding(len(fingerprint_dict), 100)  # (n_fingerprint, dim)
        # len(fingerprint_dict)
        self.W_gnn = nn.ModuleList([nn.Linear(100, 100)
                                    for _ in range(3)])
        self.W_rnn = Conv1dLSTM(in_channels=100,  # Corresponds to input size
                                out_channels=100,  # Corresponds to hidden size
                                kernel_size=3, num_layers=1, bidirectional=True,
                                dropout=0.4,
                                batch_first=True)
        self.W_rnn2 = Conv1dLSTM(in_channels=1000,  # Corresponds to input size
                                 out_channels=31,  # Corresponds to hidden size
                                 kernel_size=3, num_layers=1, bidirectional=True,
                                 dropout=0.4,
                                 batch_first=True)
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(25 + 1, 31)
        # self.denseblock1 = Dense_Block(in_channels=1000, out_channels=32, n_filter=32, k=6, dropRate=0.2)
        # self.denseblock2 = Dense_Block(in_channels=500, out_channels=32, n_filter=32, k=9, dropRate=0.2)
        # self.denseblock3 = Dense_Block(in_channels=500, out_channels=32, n_filter=32, k=12, dropRate=0.2)
        # self.denseblock1 = self.add_dense_block(dense_block, 64)
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels=196, out_channels=32)
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels=196, out_channels=32)
        self.fc_xt1 = nn.Linear(32 * 121, 128)

        # self.pro
        self.d = nn.Linear(100, 31)
        self.b = nn.Linear(1000, 31)

        self.bert = nn.Linear(500, 31)
        # combined layers
        self.fc1 = nn.Linear(1024, 768)
        self.fc2 = nn.Linear(768, 512)
        self.out = nn.Linear(512, 128)

        self.dropout = 0.2
        # self.resnet = ResnetEncoderModel(1)
        self.resnet = ResnetEncoderTiny(1)
        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=self.dropout)

        # self.fc_in = nn.Linear(8680, 4340) #1922
        self.fc_in = nn.Linear(320, 4340)  # 1922
        self.fc_out = nn.Linear(4340, 1)
        # self.attention = MultiHeadAttention(62, 62, 2)
        self.attention = MultiHeadAttention(64, 64, 2)
        #    self.W_s1 = nn.Linear(60, 45) #62
        #    self.W_s2 = nn.Linear(45, 30)

        # def attention_net(self, lstm_output):
        #    attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        #    attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        #    attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        #    return attn_weight_matrix
        #     self.conv_encoder = nn.Linear(2,500)
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=32,
                      kernel_size=7, stride=1,
                      padding=7 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm1d(num_features=32,
                           eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(.5)  # it was .2
        )
        self.gat_layer = egret.MultiHeadEGRETLayer(
            in_dim=32,
            out_dim=32,
            edge_dim=2,
            num_heads=1,
            use_bias=False,
            merge='cat',
            config_dict=config_dict)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def add_dense_block(self, block, in_channels):
        layer = []
        layer.append(block(in_channels))
        D_seq = nn.Sequential(*layer)
        return D_seq

    def _make_transition_layer(self, layer, in_channels, out_channels):
        modules1 = []
        modules1.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules1)
        # return modules1

    def rnn(self, xs):
        xs = torch.unsqueeze(xs, 0)
        xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def rnn2(self, xs):
        xs = torch.unsqueeze(xs, 0)
        xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn2(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def forward(self, g):

        feature_protein = g[0].ndata['h']
        feature_smile = g[4].ndata['h']

        # proteinemb = g[2]
        proteinseq = g[3]
        drugsmile = g[5]

        probert = g[1]
        pro3d = g[2]
        # adjacency = g[4]   x  no use...
        # adjacency = torch.tensor(adjacency)

        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[4], feature_smile))

        pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        pool_protein = GlobalAttentionPooling(self.pooling_protein)
        protein_rep = pool_protein(g[0], feature_protein).view(-1, 31)
        protein_rep = torch.Tensor(protein_rep)
        ligand_rep = pool_ligand(g[4], feature_smile).view(-1, 31)
        ligand_rep = torch.Tensor(ligand_rep)
        # sequence = []
        # for item in protein_rep:
        #    sequence.append(item.view(1, 31))
        #    sequence.append(ligand_rep)
        """protein vector with esm-1b."""

        # Load ESM-1b model
        # model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        # batch_converter = alphabet.get_batch_converter()

        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        # data = [
        #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        #     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        #     ("protein3",  "K A <mask> I S Q"),
        # # ]
        # seq1 = proteinseq
        # data = [("protein", seq1)]
        # data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")]
        # batch_labels, batch_strs, batch_tokens = batch_converter(data)
        #
        # with torch.no_grad():
        #     results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        # results = model(data, repr_layers=[33], return_contacts=True)
        # token_representations = results["representations"][33]
        #
        # sequence_representations = []
        # for i, (_, seq) in enumerate(data):
        #     sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))
        #
        # protein_rep_esm = self.esm(sequence_representations)

        """protein input feed-forward:"""
        # target = proteinseq
        # protein_max = 1000
        # target = torch.from_numpy(label_sequence(proteinseq, CHARPROTSET, protein_max))
        # embedded_xt = self.embedding_xt(proteinemb)

        # protein_rep_ConvLSTM = self.rnn2(embedded_xt)

        # embedded_xt = embedded_xt.view(1,1,1000,128)
        # denseout = self.denseblock1(embedded_xt)
        # denseout = self.transitionLayer1(denseout)
        # denseout = self.denseblock2(denseout)
        # denseout = self.transitionLayer2(denseout)
        # denseout = self.denseblock3(denseout)
        # denseout = self.transitionLayer3(denseout)

        # flatten
        # xt = denseout.view(-1, 32 * 121)
        # protein_rep_Dense = self.fc_xt1(xt)
        """protein vector with esm-1b."""
        shapes = probert.data.shape
        # features = probert.view(512,2)
        features = torch.Tensor(probert)
        # features = probert.squeeze(1).permute(0, 2, 1)
        # features = features.type(torch.DoubleTensor)
        # features = features.permute(0, 2, 1).contiguous()
        # print('1', features.shape, shapes, features.is_contiguous())
        # features = probert.reshape(1,shapes[1],shapes[0])
        features = features.permute(0, 2, 1)
        features = self.conv_encoder(features)
        # features = features.type(torch.DoubleTensor)
        shapes2 = features.data.shape
        # print(features.view([-1, 32]).shape)
        # features2 = self.multi_CNN(features)
        # pro3d['idtype']='torch.float32'

        protein_rep_bert = self.bert(features)
        protein_rep_bert = protein_rep_bert.view(32, 31)
        features2, head_attn_scores = self.gat_layer(pro3d, features.view([shapes[0] * 500, 32]))
        features2 = features2.view([shapes[0], 500, 32])
        # print('features2.shape, features.shape:', features2.shape, features.shape)
        # z.repeat(1,2).view(shapes[0],2,shapes[1])
        features = features.view(1, 32, 500)
        features2 = features2.view(1, 32, 500)
        # features = torch.cat(features2, features)
        # print(features2.shape, features.shape)
        features = torch.cat((features2, features), 2)
        # features = features.view([shapes[0], 500, 32*2])[t.nonzero(label_idx_onehot.view([shapes[0], 500])==1, as_tuple=True)]
        features = features.view(32, 1000)  # 对一个，view(1, 1000, 32)
        protein_rep_EGRET = self.b(features)

        """smile vector with rdkit."""

        mol = Chem.AddHs(Chem.MolFromSmiles(drugsmile))  # Consider hydrogens.
        atoms = create_atoms(mol)

        adjacency = create_adjacency(mol)
        adjacency = torch.FloatTensor(adjacency)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, 2)  # radius
        fingerprints = torch.from_numpy(fingerprints)
        # compounds.append(fingerprints)

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        # fingerprint_vectors = torch.from_numpy(fingerprint_vectors)
        smile_rep_fingerprint = self.gnn(fingerprint_vectors, adjacency, 3)  # layer_gnn
        # 这bushizaibuxing delete.#(1,100)
        smile_rep_fingerprint = self.d(smile_rep_fingerprint)
        """smile vector with convLSTM."""
        compound_max = 100
        # for i in enumerate(batch_data):
        smiles = torch.from_numpy(label_smiles(drugsmile, CHARISOSMISET, compound_max))

        smile_vectors = self.embed_smile(smiles)
        smile_rep_ConvLSTM = self.rnn(smile_vectors)  # (1,100)
        # keyi le  ,gai wei (1,31)
        smile_rep_ConvLSTM = self.d(smile_rep_ConvLSTM)

        sequence = torch.cat((ligand_rep, protein_rep, protein_rep_bert), dim=0).view(1, -1, 31)

        # sequence = torch.cat((ligand_rep, smile_rep_ConvLSTM, protein_rep, protein_rep_EGRET), dim=0).view(1, -1, 31)
        # sequence = torch.cat((ligand_rep, smile_rep_ConvLSTM, protein_rep, protein_rep_ConvLSTM), dim=0).view(1, -1, 31)
        # sequence = torch.cat((ligand_rep, smile_rep_ConvLSTM, protein_rep, protein_rep_esm), dim=0).view(1, -1, 31)
        # mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).cuda()
        mask = torch.eye(140, dtype=torch.uint8).view(1, 140, 140).to(device)
        mask[0, sequence.size()[1]:140, :] = 0
        mask[0, :, sequence.size()[1]:140] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0, sequence.size()[1] - 1, sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 140 - sequence.size()[1]), mode='constant', value=0)
        sequence = sequence.permute(1, 0, 2)
        sequence = sequence.view(1, 1, 140, 31)
        # h_0 = Variable(torch.zeros(2, 1, 31).cuda())
        # c_0 = Variable(torch.zeros(2, 1, 31).cuda())
        h_0 = Variable(torch.zeros(2, 1, 31).to(device))
        c_0 = Variable(torch.zeros(2, 1, 31).to(device))
        #
        # output, _ = self.bilstm(sequence, (h_0, c_0))
        output = self.resnet(sequence)
        output = output.view(1, 64, 5)
        # output = output.permute(1, 0,  2)
        output = output.permute(0, 2, 1)
        # out = self.attention(output, mask=mask)
        out = self.attention(output)
        # attn_weight_matrix = self.attention_net(output)
        # out = torch.bmm(attn_weight_matrix, output)
        out = F.relu(self.fc_in(out.view(-1, out.size()[1] * out.size()[2])))
        out = self.fc_out(out)
        # out = torch.sigmoid(self.fc_out(out))
        return out


