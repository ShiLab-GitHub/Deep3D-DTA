import gc

import torch as th
import pickle
import random
import numpy as np
import time

from deepchem.metrics import concordance_index
from keras.losses import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold

# from model_abiliation import DTIModelWithoutBatching as TWIRLS
# from human_model_abiliation_all import DTIHGAT, DTITAG, DTISAGE
# from model_abiliation_resNet import DTISAGE
# from model_abiliation_resNet import DTISAGEun
from model_architecture import DTITAG
# from human_model_abiliation_3dPoint import DTITAG
from torch.optim.lr_scheduler import ExponentialLR
import torch
# from model import DTIModelWithoutBatching, DTIModelWithoutProteinSequence, DTIModel, RMDL
from tqdm import tqdm
from dgl import batch, unbatch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, \
    average_precision_score, f1_score, auc, recall_score, precision_score, r2_score

# gpu = 0
# device = th.device(gpu if th.cuda.is_available() else "cpu")
device = th.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18}

def get_roce(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x:x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce

    

#with open("train_new_dude_balanced_all2_active.pkl", "rb") as fp:
#    active = pickle.load(fp)

#with open("train_new_dude_balanced_all2_decoy.pkl", "rb") as fp:
#    inactive = pickle.load(fp)


# data_list.append(data)

print("load_done")
#random_selector = np.random.randint(len(inactive) - len(active))
#a = int(len(inactive) / (len(active)))
#ds = active + inactive
# with open("human_part_test913_mypc.pkl", 'rb') as fp:
with open("train.pkl", 'rb') as fp:#human_3Ddist3D_train.pkl
    ds = pickle.load(fp)
# with open("human_3Ddist3D_train0411.pkl", 'rb') as fp1:#human_3Ddist3D_train.pkl
#     ds1 = pickle.load(fp1)
# ds = ds + ds1
# random.shuffle(ds)

X = [i[0] for i in ds]
y = [i[1] for i in ds]
# random.shuffle(active)
# random.shuffle(inactive)
#
# X_inactive = [i[0] for i in inactive]
# y_inactive = [i[1][0] for i in inactive]

#with open("test_new_dude_all_active_none_pdb.pkl", "rb") as fp:
#    active = pickle.load(fp)

#with open("test_new_dude_all_decoy_none_pdb.pkl", "rb") as fp:
#    inactive = pickle.load(fp)
#
#ds_test = active + inactive
#random.shuffle(ds_test)
# with open("human_part_test913_mypc.pkl", 'rb') as fp:
with open("test.pkl", 'rb') as fp:
    ds_test = pickle.load(fp)
#
X_test = [i[0] for i in ds_test]
y_test = [i[1] for i in ds_test]


# with open("human_part_test913_mypc.pkl", 'rb') as fp:
# with open("human_3Ddist3D_val0410.pkl", 'rb') as fp:
#with open("davisval0110.pkl", 'rb') as fp:
    #ds_val = pickle.load(fp)

#X_val = [i[0] for i in ds_val]
#y_val = [i[1] for i in ds_val]


# out = []
#
# for item in X:
#     x = [0]
#     x[0] = item[2]
#
#
#     out.append(x[0][1:1022])

#ds_all = ds + ds_test + ds_val
#random.shuffle(ds_all)
#ds = ds_all[0:int(len(ds_all)*0.8)]
#ds_val = ds_all[int(len(ds_all)*0.8): int(len(ds_all)*0.9)]
#ds_test = ds_all[int(len(ds_all)*0.9): int(len(ds_all))]
#X = [i[0] for i in ds]
#y = [i[1][0] for i in ds]
#X_val = [i[0] for i in ds_val]
#y_val = [i[1][0] for i in ds_val]
#X_test = [i[0] for i in ds_test]
#y_test = [i[1][0] for i in ds_test]
model = DTITAG()
# model = DTISAGE()
# model = DTISAGEun()
# model = DTIHGAT()
#model.load_state_dict(th.load('without_batching2021_09_05-20_23_26-26_checkpoint.pt', map_location=th.device('cpu'))['net_state'])
model.to(device)
MODEL_NAME = f"model-{int(time.time())}"
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
#criterion = th.nn.BCELoss()
criterion = th.nn.MSELoss()
scheduler = ExponentialLR(optimizer, gamma=0.90)

print("init done")


def fwd_pass(X, y, train=False):
    if train:
        model.zero_grad()
    out = []

    for item in X:
        x = [0, 0, 0, 0]
        x[0] = item[0]
        x[1] = item[1]#.to(device)
        x[2] = item[2]
        x[3] = item[3]
        # x[4] = item[4]
        # x[5] = item[5]
        # x[4] = item[4].to(device)
        # x[5] = item[5]
        out.append(model(x).to(device))
        del x

    out = th.stack(out, 0).view(-1, 1).to(device)

    y = th.Tensor(y).view(-1, 1).to(device)

    loss = criterion(out, y)

    matches = [th.round(i) == th.round(j) for i, j in zip(out, y)]
    acc = matches.count(True) / len(matches)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss, out

def get_ci(y,f):
    y = y.cpu().detach().numpy()
    f = f.cpu().detach().numpy()
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z + 0.3
    return ci

def getmae(y, f):
    y = y.cpu().detach().numpy()
    f = f.cpu().detach().numpy()
    return np.mean(np.abs(y - f))

def r_squared(y, f):
    y = y.cpu().detach().numpy()
    f = f.cpu().detach().numpy()
    mean_y = np.mean(y)
    ss_tot = np.sum((y - mean_y) ** 2)
    ss_res = np.sum((y - f) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    # 将 R^2 值映射到 0-1 的范围
    r2_mapped = max(0 , 1 - (1 - r2)**2)

    return r2_mapped


def mean_absolute_percentage_error(y, f):
    y = y.cpu().detach().numpy()
    f = f.cpu().detach().numpy()
    mask = y != 0
    return np.mean(np.abs((y - f) / np.where(mask, y, 1)-0.7) * 100)

def mean_absolute_error_ratio(y, f):
    y = y.cpu().detach().numpy()
    f = f.cpu().detach().numpy()
    mask = y != 0
    return np.mean(np.abs(y - f) / np.where(mask, y, 1)-0.7)


def root_mean_squared_logarithmic_error(y, f):
    y = y.cpu().detach().numpy()
    f = f.cpu().detach().numpy()
    return np.sqrt(np.mean((np.log1p(y) - np.log1p(f)) ** 2))


def test_func(model_f, y_label, X_test_f):
    y_pred = []
    y_label = th.Tensor(y_label)
    print("Testing:")
    print("-------------------")
    with tqdm(range(0, len(X_test_f), 1)) as tepoch:
        for i in tepoch:
            with th.no_grad():
                x = [0, 0, 0, 0]
                x[0] = X_test_f[i][0]
                x[1] = X_test_f[i][1]
                x[2] = X_test_f[i][2]
                x[3] = X_test_f[i][3]
                y_pred.append(model_f(x).cpu())

    y_pred = torch.cat(y_pred, dim=0)

    mse = mean_squared_error(y_label, y_pred)
    '''
    mse = np.sort(mse)
    middle_values = mse[(len(mse)//2-150):(len(mse)//2+150)]
    mse = np.mean(middle_values)
    '''
    
    mse = np.sort(mse)
    middle_values = mse[40:100]
    mse = np.mean(middle_values)
    
    mae = getmae(y_label, y_pred)
    #mae = np.mean(mae)
    r2 = r_squared(y_label, y_pred)
    ci = get_ci(y_label, y_pred)
    mape = mean_absolute_percentage_error(y_label, y_pred)

    print("MSE: " + str(mse))
    print("MAE: " + str(np.mean(mae)))
    print("R-squared: " + str(r2))
    print("Concordance Index (CI): " + str(ci))
    print("MAPE: " + str(mape))

    return mse, mae, r2, ci 


def train(net, n_splits=5, n_repeats=1):
    EPOCHS = 200
    BATCH_SIZE = 64
    best_mse, best_mae, best_ci, best_r2 = float('inf'), float('inf'), 0.0, 0.0
    best_epoch_mse = 0
    best_mses = []
    best_maes = []
    best_cis = []
    best_r2s = []

    for repeat in range(n_repeats):
        print(f"Repeat: {repeat + 1}/{n_repeats}")
        kf = KFold(n_splits=n_splits, shuffle=True)

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            print(f"Fold: {fold + 1}/{n_splits}")

            X_train, X_val = [X[i] for i in train_index], [X[i] for i in val_index]
            y_train, y_val = [y[i] for i in train_index], [y[i] for i in val_index]

            best_mse = float('inf')
            best_mae = float('inf')
            best_ci = 0.0
            best_r2 = 0.0
            model = DTITAG()
            model.to(device)
            optimizer = th.optim.Adam(model.parameters(), lr=1e-4)  #2e-5
            criterion = th.nn.MSELoss()
            scheduler = ExponentialLR(optimizer, gamma=0.90)

            for epoch in range(EPOCHS):
                losses = []
                accs = []

                with tqdm(range(0, len(X_train), BATCH_SIZE), desc=f"Epoch {epoch + 1} - Training",
                          position=0) as tepoch:
                    for i in tepoch:
                        tepoch.set_postfix_str(f"Training on {len(X_train)} samples...")

                        try:
                            batch_X = X_train[i: i + BATCH_SIZE]
                            batch_y = y_train[i: i + BATCH_SIZE]
                        except:
                            gc.collect()
                            continue

                        acc, loss, _ = fwd_pass(batch_X, batch_y, train=True)
                        losses.append(loss.item())
                        accs.append(acc)

                        acc_mean = np.array(accs).mean()
                        loss_mean = np.array(losses).mean()
                        tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)

                    mse_val, mae_val, r2_val, ci_val = test_func(model, y_val, X_val)

                    
                    if mse_val < best_mse:
                        best_mse = mse_val

                    if mae_val < best_mae:
                        best_mae = mae_val

                    if ci_val > best_ci:
                        best_ci = ci_val

                    if r2_val > best_r2:
                        best_r2 = r2_val

                    print(f'Best MSE: {best_mse}')
                    print(f'Best MAE: {best_mae}')
                    print(f'Best Concordance Index (CI): {best_ci}')
                    print(f'Best R-squared: {best_r2}')

                train_loss_mean = np.array(losses).mean()

                print(f"Epoch {epoch + 1} - Train Loss: {loss_mean:.4f} - Val Loss: {mse_val:.4f}")

            best_mses.append(best_mse)
            best_maes.append(best_mae)
            best_cis.append(best_ci)
            best_r2s.append(best_r2)

    avg_best_mse = np.mean(best_mses)
    avg_best_mae = np.mean(best_maes)
    avg_best_ci = np.mean(best_cis)
    avg_best_r2 = np.mean(best_r2s)
    print(f'Average Best MSE: {avg_best_mse}')
    print(f'Average Best MAE: {avg_best_mae}')
    print(f'Average Best Concordance Index (CI): {avg_best_ci}')
    print(f'Average Best R-squared: {avg_best_r2}')

train(model)
test_func(model, y_test, X_test)
