import gc
import torch as th
import pickle
import random
import numpy as np
import time
import os
from pathlib import Path

from deepchem.metrics import concordance_index
from keras.losses import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold

from model_architecture import DTITAG
from torch.optim.lr_scheduler import ExponentialLR
import torch
from tqdm import tqdm
from dgl import batch, unbatch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, \
    average_precision_score, f1_score, auc, recall_score, precision_score, r2_score

# 设备配置
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建保存目录
save_dir = Path("checkpoints")
save_dir.mkdir(exist_ok=True)

def get_roce(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x:x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce

def load_data():
    """加载数据集"""
    print("Loading datasets...")
    
    # 训练集
    try:
        with open("train.pkl", 'rb') as fp:
            ds_train = pickle.load(fp)
        print(f"Loaded training set: {len(ds_train)} samples")
    except FileNotFoundError:
        print("train.pkl not found!")
        return None, None, None, None, None, None
    
    # 验证集
    try:
        with open("val.pkl", 'rb') as fp:
            ds_val = pickle.load(fp)
        print(f"Loaded validation set: {len(ds_val)} samples")
    except FileNotFoundError:
        print("val.pkl not found, will use train/val split")
        # 从训练集中分割验证集
        random.shuffle(ds_train)
        split_idx = int(0.8 * len(ds_train))
        ds_val = ds_train[split_idx:]
        ds_train = ds_train[:split_idx]
        print(f"Split training set: {len(ds_train)} train, {len(ds_val)} val")
    
    # 测试集
    try:
        with open("test.pkl", 'rb') as fp:
            ds_test = pickle.load(fp)
        print(f"Loaded test set: {len(ds_test)} samples")
    except FileNotFoundError:
        print("test.pkl not found!")
        return None, None, None, None, None, None
    
    # 提取特征和标签
    X_train = [i[0] for i in ds_train]
    y_train = [i[1] for i in ds_train]
    
    X_val = [i[0] for i in ds_val]
    y_val = [i[1] for i in ds_val]
    
    X_test = [i[0] for i in ds_test]
    y_test = [i[1] for i in ds_test]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def fwd_pass(model, X, y, criterion, optimizer=None, train=False):
    """前向传播"""
    if train and optimizer is not None:
        model.zero_grad()
    
    out = []
    for item in X:
        x = [item[0], item[1], item[2], item[3]]
        prediction = model(x).to(device)
        out.append(prediction)
    
    out = th.stack(out, 0).view(-1, 1).to(device)
    y_tensor = th.Tensor(y).view(-1, 1).to(device)
    
    loss = criterion(out, y_tensor)
    
    if train and optimizer is not None:
        loss.backward()
        optimizer.step()
    
    return loss.item(), out

def get_metrics(y_true, y_pred):
    """计算各种评估指标"""
    y_true = y_true.cpu().detach().numpy().flatten()
    y_pred = y_pred.cpu().detach().numpy().flatten()
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # CI (Concordance Index)
    ci = concordance_index(y_true, y_pred)
    
    return {"mse": mse, "mae": mae, "r2": r2, "ci": ci}

def evaluate_model(model, X_val, y_val, criterion):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        loss, predictions = fwd_pass(model, X_val, y_val, criterion, train=False)
        metrics = get_metrics(torch.tensor(y_val), predictions)
        metrics["loss"] = loss
    return metrics

def train_model(X_train, y_train, X_val, y_val, config=None):
    """训练模型"""
    if config is None:
        config = {
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "scheduler_gamma": 0.95,
            "early_stopping_patience": 20
        }
    
    # 初始化模型
    model = DTITAG()
    model.to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss()
    scheduler = ExponentialLR(optimizer, gamma=config["scheduler_gamma"])
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_losses = []
        
        # 训练阶段
        indices = list(range(len(X_train)))
        random.shuffle(indices)
        
        with tqdm(range(0, len(X_train), config["batch_size"]), 
                  desc=f"Epoch {epoch+1}/{config['epochs']}") as pbar:
            
            for i in pbar:
                batch_indices = indices[i:i+config["batch_size"]]
                batch_X = [X_train[idx] for idx in batch_indices]
                batch_y = [y_train[idx] for idx in batch_indices]
                
                try:
                    loss, _ = fwd_pass(model, batch_X, batch_y, criterion, optimizer, train=True)
                    epoch_losses.append(loss)
                    
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
                    
                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    gc.collect()
                    continue
        
        # 验证阶段
        val_metrics = evaluate_model(model, X_val, y_val, criterion)
        
        # 记录历史
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        val_losses.append(val_metrics["loss"])
        val_metrics_history.append(val_metrics)
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val MSE: {val_metrics['mse']:.4f}")
        print(f"  Val MAE: {val_metrics['mae']:.4f}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")
        print(f"  Val CI: {val_metrics['ci']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 早停检查
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics.copy()
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'val_metrics': val_metrics,
                'config': config
            }, save_dir / "best_model.pt")
            
        else:
            patience_counter += 1
            
        if patience_counter >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_metrics, train_losses, val_losses

def test_model(model, X_test, y_test):
    """测试模型"""
    print("\nTesting model...")
    criterion = torch.nn.MSELoss()
    
    model.eval()
    with torch.no_grad():
        loss, predictions = fwd_pass(model, X_test, y_test, criterion, train=False)
        metrics = get_metrics(torch.tensor(y_test), predictions)
        metrics["loss"] = loss
    
    print("Test Results:")
    print(f"  Test Loss: {metrics['loss']:.4f}")
    print(f"  Test MSE: {metrics['mse']:.4f}")
    print(f"  Test MAE: {metrics['mae']:.4f}")
    print(f"  Test R²: {metrics['r2']:.4f}")
    print(f"  Test CI: {metrics['ci']:.4f}")
    
    return metrics

def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    if X_train is None:
        print("Failed to load data!")
        return
    
    # 训练配置
    config = {
        "epochs": 200,
        "batch_size": 32,  # 调整batch size
        "learning_rate": 1e-4,
        "scheduler_gamma": 0.95,
        "early_stopping_patience": 20
    }
    
    # 训练模型
    model, best_val_metrics, train_losses, val_losses = train_model(
        X_train, y_train, X_val, y_val, config
    )
    
    print("\nBest Validation Metrics:")
    print(f"  Val Loss: {best_val_metrics['loss']:.4f}")
    print(f"  Val MSE: {best_val_metrics['mse']:.4f}")
    print(f"  Val MAE: {best_val_metrics['mae']:.4f}")
    print(f"  Val R²: {best_val_metrics['r2']:.4f}")
    print(f"  Val CI: {best_val_metrics['ci']:.4f}")
    
    # 测试模型
    test_metrics = test_model(model, X_test, y_test)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    val_mse = [m['mse'] for m in val_metrics_history]
    val_ci = [m['ci'] for m in val_metrics_history]
    plt.plot(val_mse, label='Val MSE')
    plt.plot(val_ci, label='Val CI')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 