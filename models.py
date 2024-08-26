from GRANDE import GRANDE
import tensorflow as tf
import gc

import os
import xgboost as xgb
import joblib
import optuna

from utils import set_seed, get_metric#, RMSELoss
from torchmetrics import AUROC, R2Score

from sklearn.metrics import r2_score, roc_auc_score

import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rtdl_revisiting_models import MLP, ResNet, FTTransformer, CategoricalEmbeddings
from rtdl_num_embeddings import PeriodicEmbeddings   

from category_encoders.leave_one_out import LeaveOneOutEncoder

# import delu
import time

import catboost as cb
from torchcontrib.optim import SWA
import lightgbm as lgbm

import sklearn
from copy import deepcopy
import category_encoders as ce
import math
from focal_loss import SparseCategoricalFocalLoss

import pickle
import zipfile

import warnings
warnings.filterwarnings("ignore")

def get_model(model_name):

    if model_name=="XGBoost":
        model_class = XGBModel
        model_class.model_class = "tree" 
    elif model_name=="AutoGluon":
        model_class = AutoGluonModel
        model_class.model_class = "autogluon" 
    elif model_name=="ResNet":
        model_class = ResNetModel
        model_class.model_class = "neural_net" 
    elif model_name=="FTTransformer":
        model_class = FTTransformerModel
        model_class.model_class = "neural_net" 
    elif model_name=="CatBoost":
        model_class = CatBoostModel
        model_class.model_class = "tree" 
    elif model_name=="GRANDE":
        model_class = GRANDEModelTF
        model_class.model_class = "hybrid"  # own preprocessing is applied, unlike other neural networks
    elif model_name=="MLP-PLR":
        model_class = MLPPLR
        model_class.model_class = "neural_net" 
    elif model_name=="LightGBM":
        model_class = LightGBMModel
        model_class.model_class = "tree" 
    else:
        raise ValueError(f"Model '{model_name}' not implemented.")
    
    return model_class


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y) + self.eps)

class BaseModel:
    '''
    Possible splittings
        - model agnostic vs. model-dependent params
        - build vs fit (before/after seeing data)
        - tunable hyperparams vs rest
    Currently: Put all in one at init
    
    Required in params: task_type, task_type, cont_indices, cat_indices, cat_cardinalities, device, d_out,
    
    Each model needs to define the following functions: 
        init
        fit
        predict 
        get_default_hyperparameters
        get_optuna_hyperparameters
        
    '''
    def __init__(self, params):    
        
        self.params = params
        
        # Task specific parameters
        # self.task_type = params["task_type"]
        # self.cont_indices = params["cont_indices"]
        # self.cat_indices = params["cat_indices"]
        # self.cat_cardinalities = params["cat_cardinalities"]
        # self.d_out = params["d_out"]  
        # self.device = params["device"] 
        # self.save_path = params["save_path"] 
        
    
class ResNetModel(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            d_hidden=None,
            d_hidden_multiplier=2.0,
            dropout1=0.15,
            dropout2=0.0,
            d_embedding=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        self.model_type = "neural_net"
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       
        
        self.resnet = ResNet(
            d_in=self.params["n_cont_features"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
            d_out=self.params["d_out"],
            n_blocks=self.params["hyperparameters"]["n_blocks"],
            d_block=self.params["hyperparameters"]["d_block"],
            d_hidden=self.params["hyperparameters"]["d_hidden"],
            d_hidden_multiplier=self.params["hyperparameters"]["d_hidden_multiplier"],
            dropout1=self.params["hyperparameters"]["dropout1"],
            dropout2=self.params["hyperparameters"]["dropout2"],
        )                
        self.resnet.to(self.params["device"])
        
        if self.params["n_cat_features"]>0:
            self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
            self.embedding_layer.to(self.params["device"])

    
    def forward(self,X):
        if self.params["n_cat_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            X_cont_catembed = torch.cat([X_cont, X_cat_embed.flatten(1, -1)],dim=1)
            res = self.resnet(X_cont_catembed)
        else:
            res = self.resnet(X)
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], )) 
        
        if self.params["n_cat_features"]>0:
            # Define optimizer
            optimizer = optim.AdamW(
                list(self.resnet.parameters())+list(self.embedding_layer.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
        else:
            optimizer = optim.AdamW(
                self.resnet.parameters(), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "classification":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0


        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.resnet.train()
            if self.params["n_cat_features"]>0:
                self.embedding_layer.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
                history["loss"].append(loss.detach().cpu())
            
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.resnet.eval()
                if self.params["n_cat_features"]>0:
                    self.embedding_layer.eval()
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1

                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions                
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.resnet.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_ResNet.pt")
                    if self.params["n_cat_features"]>0:
                        torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_resnet = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_ResNet.pt")
        self.resnet.load_state_dict(state_dict_resnet)
        if self.params["n_cat_features"]>0:
            state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            self.embedding_layer.load_state_dict(state_dict_embeddings)
        torch.cuda.empty_cache()
        gc.collect()
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.resnet.eval()
        if self.params["n_cat_features"]>0:
            self.embedding_layer.eval()

        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="classification":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, dataset_name="", **kwargs):
        
        if dataset_name == "bnp-paribas-cardif-claims-management":
            max_embedding = 128
        else:
            max_embedding = 512
        
        params = {
            "n_blocks": trial.suggest_int("n_blocks",1,8), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",64,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "d_hidden": trial.suggest_categorical("d_hidden",[None]), # Original paper: (A) UniformInt[64, 512], (B) UniformInt[64, 1024], Kadra: don't use
            "d_hidden_multiplier": trial.suggest_int("d_hidden_multiplier",1,4), # Original paper: [2.0]), (A,B) Uniform[1, 4], Kadra: same
            "dropout1": trial.suggest_float("dropout1",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5], Kadra: same
            "dropout2": trial.suggest_float("dropout2",0.,0.5), # Original paper:  (A,B) {0, Uniform[0, 0.5]}, Kadra: same
            "d_embedding": trial.suggest_int("d_embedding",4,max_embedding), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Original paper: ?, Kadra: 1e-6, 1e-3
            }
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # if large_dataset:
        #     learning_rate = 0.00001
        #     n_blocks = 4
        #     weight_decay=0.000001
        # else:
        #     learning_rate = 0.001
        #     weight_decay=0.0001
        #     n_blocks = 2
        learning_rate = 0.0001
        weight_decay=0.00001
        params = {
            "n_blocks": 2, #2
            "d_block": 192, # 
            "d_hidden": None, # 
            "d_hidden_multiplier": 2.0, # 
            "dropout1": 0.25, # 
            "dropout2": 0.0, # 
            "d_embedding": 8, # 
            "learning_rate": learning_rate, # 0.001
            "weight_decay": weight_decay
            
        }
        return params    


#########################################################       
#########################################################       
#########################################################       

class FTTransformerModel(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            d_hidden=None,
            d_hidden_multiplier=2.0,
            dropout1=0.15,
            dropout2=0.0,
            d_embedding=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       
        else:
            self.params["hyperparameters"]["_is_default"] = False
        
        torch.backends.cudnn.benchmark = True
        if self.params["hyperparameters"]["_is_default"]:
            self.FTTransformer = FTTransformer(
                **self.params["hyperparameters"],
                # d_in=self.params["n_cont_features"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
                n_cont_features=self.params["n_cont_features"],
                cat_cardinalities=self.params["cat_cardinalities"],
                d_out=self.params["d_out"],
                # n_blocks=self.params["hyperparameters"]["n_blocks"],
                # d_block=self.params["hyperparameters"]["d_block"],
                # d_hidden=self.params["hyperparameters"]["d_hidden"],
                # d_hidden_multiplier=self.params["hyperparameters"]["d_hidden_multiplier"],
                # dropout1=self.params["hyperparameters"]["dropout1"],
                # dropout2=self.params["hyperparameters"]["dropout2"],
                linformer_kv_compression_ratio=0.2,           # <---
                linformer_kv_compression_sharing='headwise'
            )                
        else:
            self.FTTransformer = FTTransformer(
                # **self.params["hyperparameters"],
                # d_in=self.params["n_cont_features"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
                n_cont_features=self.params["n_cont_features"],
                cat_cardinalities=self.params["cat_cardinalities"],
                d_out=self.params["d_out"],
                n_blocks=self.params["hyperparameters"]["n_blocks"],
                d_block=self.params["hyperparameters"]["d_block"],
                attention_dropout=self.params["hyperparameters"]["attention_dropout"],
                ffn_d_hidden_multiplier=self.params["hyperparameters"]["ffn_d_hidden_multiplier"],
                ffn_dropout=self.params["hyperparameters"]["ffn_dropout"],
                residual_dropout=self.params["hyperparameters"]["residual_dropout"],
                attention_n_heads=self.params["hyperparameters"]["attention_n_heads"],
                linformer_kv_compression_ratio=0.2,           # <---
                linformer_kv_compression_sharing='headwise'
            )                
            
        self.FTTransformer.to(self.params["device"])
        
        # if self.params["n_cat_features"]>0:
        #     self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
        #     self.embedding_layer.to(self.params["device"])

    
    def forward(self,X):
        if len(self.params["cont_indices"])==0:
            X_cont = None
        else:
            X_cont = X[:,self.params["cont_indices"]]
        if self.params["n_cat_features"]>0:
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
        else:
            X_cat = None
            
        #     X_cat_embed = self.embedding_layer(X_cat)
        #     X_cont_catembed = torch.cat([X_cont, X_cat_embed.flatten(1, -1)],dim=1)
        #     res = self.resnet(X_cont_catembed)
        # else:
        #     res = self.resnet(X)
        
        return self.FTTransformer(X_cont, X_cat)#.squeeze(-1)

        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''

        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4'
        # torch.backends.cudnn.benchmark = True
        
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], )) 
        
        # if self.params["n_cat_features"]>0:
        #     # Define optimizer
        #     if self.params["hyperparameters"]
        #     optimizer = optim.AdamW(
        #         list(self.resnet.parameters())+list(self.embedding_layer.parameters()), 
        #         lr=self.params["hyperparameters"]["learning_rate"], 
        #         weight_decay=self.params["hyperparameters"]["weight_decay"]
        #     )
        #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
        #     # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
        if self.params["hyperparameters"]["_is_default"]:
            optimizer = self.FTTransformer.make_default_optimizer()
        else:
            optimizer = self.FTTransformer.make_default_optimizer()
            for group in optimizer.param_groups:
                group['lr'] = self.params["hyperparameters"]["learning_rate"]
            optimizer.param_groups[1]["weight_decay"] = self.params["hyperparameters"]["weight_decay"]

            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "classification":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0
        # if eval_direction=="minimize":
        #     early_stopping = delu.tools.EarlyStopping(patience, mode="min")
        # else:
        #     early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        #         early_stopping.update(val_eval.detach().cpu())
        #         if early_stopping.should_stop():
        #             print(f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!")
        #             break

        print(f"total_params = {np.round(sum(p.numel() for p in self.FTTransformer.parameters())/1000000,3)}M")
        
        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.FTTransformer.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
                history["loss"].append(loss.detach().cpu())
            
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            ###################


            #############
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.FTTransformer.eval()
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)

                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1
                
                # print(y_val.shape,torch.concatenate(predictions).shape)
                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.FTTransformer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_FTTransformer.pt")
                
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_fttransformer = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_FTTransformer.pt")
        self.FTTransformer.load_state_dict(state_dict_fttransformer)
        
        torch.cuda.empty_cache()
        gc.collect()
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        self.FTTransformer.eval()

        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="classification":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, dataset_name="", **kwargs):
        params = FTTransformer.get_default_kwargs()
        if dataset_name=="santander-value-prediction-challenge":
            params["n_blocks"] = 1
    
        params["attention_dropout"] = trial.suggest_float("attention_dropout",0.0,0.5)
        params["ffn_dropout"] = trial.suggest_float("ffn_dropout",0.,0.5)
        params["residual_dropout"] = trial.suggest_float("residual_dropout",0.,0.2)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        params["_is_default"] = False
        
        # params = {
        #     "attention_dropout": trial.suggest_float("attention_dropout",0.0,0.5), # Original paper: [0,0.5]
        #     # "ffn_d_hidden_multiplier": trial.suggest_float("d_ffn_factor",2/3,8/3), # Original paper: [2/3,8/3]
        #     # "d_block": trial.suggest_categorical("d_block",[96, 128, 192, 256, 320, 384]), # Original paper: Feature embedding size UniformInt[64, 512]  # [96, 128, 192, 256, 320, 384][n_blocks - 1]
        #     # "d_block": trial.suggest_int("d_block",64,512,8), # Original paper: Feature embedding size UniformInt[64, 512]  # [96, 128, 192, 256, 320, 384][n_blocks - 1]            
        #     "ffn_dropout": trial.suggest_float("ffn_dropout",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5]
        #     # "n_blocks": trial.suggest_int("n_blocks",1,3), # Original paper: (A) UniformInt[1, 4], (B) UniformInt[1, 6]
        #     "residual_dropout": trial.suggest_float("residual_dropout",0.,0.2), # Original paper:  (A,B) {0, Uniform[0, 0.5]}
        #     # "attention_n_heads": 8,
        #     "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
        #     "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), # Original paper: [1e-6, 1e-3]
        #     "_is_default": False
        # }

        

        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # Default hyperparameters
        # {'n_blocks': 3,
        #  'd_block': 192,
        #  'attention_n_heads': 8,
        #  'attention_dropout': 0.2,
        #  'ffn_d_hidden': None,
        #  'ffn_d_hidden_multiplier': 1.3333333333333333,
        #  'ffn_dropout': 0.1,
        #  'residual_dropout': 0.0,
         # 'learning_rate': 1e-4,
         # 'weight_decay': 1e-5,
        #  '_is_default': True}

        params = FTTransformer.get_default_kwargs()
        # if large_dataset:
        #     params["n_blocks"] = 1
        
        return params    





#################################################################
#################################################################
#################################################################



class MLPPLR(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            dropout=0.15,
            d_embedding=8,
            d_embedding_num=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       

        self.mlp = MLP(
            d_in=self.params["n_cont_features"]*self.params["hyperparameters"]["d_embedding_num"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
            d_out=self.params["d_out"],
            n_blocks=self.params["hyperparameters"]["n_blocks"],
            d_block=self.params["hyperparameters"]["d_block"],
            dropout=self.params["hyperparameters"]["dropout"],
            
        )                
        self.mlp.to(self.params["device"])
        
        if self.params["n_cat_features"]>0:
            self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
            self.embedding_layer.to(self.params["device"])

        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont = PeriodicEmbeddings(self.params["n_cont_features"], 
                                                           frequency_init_scale = self.params["hyperparameters"]["frequency_init_scale"],
                                                           d_embedding=self.params["hyperparameters"]["d_embedding_num"], lite=self.params["hyperparameters"]["lite"])
            self.embedding_layer_cont.to(self.params["device"])
            

    
    def forward(self,X):
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            X_cont_embed = self.embedding_layer_cont(X_cont)
            X_contembed_catembed = torch.cat([X_cont_embed.flatten(1, -1), X_cat_embed.flatten(1, -1)],dim=1)
            res = self.mlp(X_contembed_catembed)
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X
            
            X_cont_embed = self.embedding_layer_cont(X_cont)
            res = self.mlp(X_cont_embed.flatten(1, -1))        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            X_cat = X.to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            res = self.mlp(X_cat_embed.flatten(1, -1))      
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], ))      
            
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )                
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            # Define optimizer
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "classification":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0
        
        # print(f"total_params = {np.round(sum(p.numel() for p in self.mlp.parameters())/1000000,3)}M")
        
        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.mlp.train()
            if self.params["n_cat_features"]>0:
                self.embedding_layer.train()
            if self.params["n_cont_features"]>0:
                self.embedding_layer_cont.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
                history["loss"].append(loss.detach().cpu())
            
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.mlp.eval()
                if self.params["n_cat_features"]>0:
                    self.embedding_layer.eval()
                if self.params["n_cont_features"]>0:
                    self.embedding_layer_cont.eval()                    
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1
                
                
                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions        
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.mlp.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_ResNet.pt")
                    if self.params["n_cat_features"]>0:
                        torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                    if self.params["n_cont_features"]>0:
                        torch.save(self.embedding_layer_cont.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings_Cont.pt")
                        
                    
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_resnet = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_ResNet.pt")
        self.mlp.load_state_dict(state_dict_resnet)
        if self.params["n_cat_features"]>0:
            state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            self.embedding_layer.load_state_dict(state_dict_embeddings)
        if self.params["n_cont_features"]>0:
            state_dict_embeddings_cont = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings_Cont.pt")
            self.embedding_layer_cont.load_state_dict(state_dict_embeddings_cont)

        
        torch.cuda.empty_cache()
        gc.collect()
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.mlp.eval()
        if self.params["n_cat_features"]>0:
            self.embedding_layer.eval()
        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont.eval()              
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="classification":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, **kwargs):
        params = {
            "n_blocks": trial.suggest_int("n_blocks",1,8), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",1,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "dropout": trial.suggest_float("dropout",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5], Kadra: same
            "d_embedding": trial.suggest_int("d_embedding",1,128), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "d_embedding_num": trial.suggest_int("d_embedding_num",1,128), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "frequency_init_scale": trial.suggest_float("frequency_init_scale", 0.01, 10., log=True), # Original paper:  0.01, 100. - but in docu they recommend to set to max 10 
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 0.005, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), # Original paper: ?, Kadra: 1e-6, 1e-3
            # "frequency_init_scale": 0, 
            "lite": True,
            
        }
        
        # params["d_embedding"] = params["d_embedding_num"]
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # if large_dataset:
        #     learning_rate = 0.00001
        #     n_blocks = 4
        #     weight_decay=0.000001
        # else:
        #     learning_rate = 0.001
        #     weight_decay=0.0001
        #     n_blocks = 2
        learning_rate = 0.001
        weight_decay=0.0001
        params = {
            "n_blocks": 2, #2
            "d_block": 192, # 
            "dropout": 0.25, # 
            "d_embedding": 8, # 
            "d_embedding_num": 8, # 
            "lite": False,
            "frequency_init_scale": 0.01,
            "learning_rate": learning_rate, # 0.001
            "weight_decay": weight_decay
            
        }
        return params    
        
    
#################################################################
#################################################################
#################################################################

    
class AutoGluonModel(BaseModel):
    def __init__(self, params):
        '''
        Model-specific Parameters in params:
            time_limit
            presets
            num_cpus
        '''
        super().__init__(params)
        
        
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        elif self.params["eval_metric"] in ["auc", "gini"]:
            self.params["eval_metric"] = "roc_auc"
        elif self.params["eval_metric"]=="rmse":
            self.params["eval_metric"] = "root_mean_squared_error"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "log_loss"
        elif self.params["eval_metric"]=="mae":
            self.params["eval_metric"] = "mean_absolute_error"
        elif self.params["eval_metric"]=="mlogloss":
            self.params["eval_metric"] = "log_loss"
        elif self.params["eval_metric"]=="logloss":
            self.params["eval_metric"] = "log_loss"
            
        
        if "num_cpus" not in params:
            self.params["num_cpus"] = None
                
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()                            
        
    def fit(self, 
            X_train, y_train, 
            eval_set=None,
           ):
        
        label = y_train.name
        data = pd.concat([X_train,y_train],axis=1)
        
        self.model = TabularPredictor(label, eval_metric=self.params["eval_metric"],
                                      path=f"./logs/AutoGluon/{self.params['dataset_name']}_{self.params['exp_name']}",

                                     )
        self.model.fit(data, 
                       # ag_args_fit={'num_gpus': 1},
                       time_limit=self.params["time_limit"], 
                       presets=self.params["presets"],
                       ag_args_fit={'num_cpus': self.params["num_cpus"]},
                      )

        
        
    def predict(self, X):
        if self.params["task_type"]=="binary":
            pred = self.model.predict_proba(X).iloc[:,1].values            
        elif self.params["task_type"]=="classification":
            pred = self.model.predict_proba(X).values            
        else:
            pred = self.model.predict(X).values
            
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, **kwargs):
        params = {}        
        
        return params    
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {}        
        
        return params    
    

class XGBModel(BaseModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        
        super().__init__(params)
        
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        # Use rmse for r2 as objective is same
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "logloss"
        elif self.params["eval_metric"]=="gini":
            self.params["eval_metric"] = "auc"
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()               
        
        if self.params["task_type"] == "regression":
            self.model = xgb.XGBRegressor(
                **self.params["hyperparameters"],
                booster = "gbtree",
                tree_method = "hist",
                enable_categorical=True,
                sampling_method="gradient_based",
                n_estimators = self.params["n_estimators"],
                early_stopping_rounds=self.params["patience"],
                device=self.params["device"],
                eval_metric = self.params["eval_metric"],
            )
        elif self.params["task_type"] in ["binary", "classification"]:
            self.model = xgb.XGBClassifier(
                **self.params["hyperparameters"],
                booster = "gbtree",
                tree_method = "hist", 
                enable_categorical=True,
                sampling_method="gradient_based",
                n_estimators = self.params["n_estimators"],
                early_stopping_rounds=self.params["patience"],
                device=self.params["device"],
                eval_metric = self.params["eval_metric"],
            )

        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):

        self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns
        self.cat_dtypes = {}
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
        
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    # X_train.loc[:,col] = X_train[col].astype(object)
                    # X_val.loc[:,col] = X_val[col].astype(object)
                    # u_cats = list(X_train[col].unique())+["nan"] #np.unique(list(X_train[col].unique())+list(X_val[col].unique())+["nan"]).tolist()
                    # self.cat_dtypes[col] = pd.CategoricalDtype(categories=u_cats)
                    # X_train.loc[:,col] = X_train.loc[:,col].astype(self.cat_dtypes[col])
                    # X_val.loc[:,col] = X_val.loc[:,col].astype(self.cat_dtypes[col])

                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
                    X_val[col] = X_val[col].astype(self.cat_dtypes[col])            

            eval_set = [(X_val,y_val)]
        else:
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
            eval_set = [(X_train,y_train)]

        h = self.model.fit(X_train, y_train, 
            eval_set=eval_set,
            verbose=50
                          )
        
    def predict(self, X):
        for col in self.cat_col_names:
            if X.loc[:,col].dtype!="category":
                X[col] = X[col].astype(self.cat_dtypes[col])

        self.model.set_params(device="cpu")
        
        if self.params["task_type"]=="regression":
            pred = self.model.predict(X)            
        elif self.params["task_type"]=="binary":
            pred = self.model.predict_proba(X)[:,1]            
        elif self.params["task_type"]=="classification":
            pred = self.model.predict_proba(X)         
        
        self.model.set_params(device="cuda") 
        
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": trial.suggest_int("max_depth", 1, 11, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        

        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 6, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }        
        
        return params    
    
    
class CatBoostModel(BaseModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: iterations, patience, device
        If classification, additionally num_classes has to be given
        '''
        
        super().__init__(params)
        # Not tunable parameters
        if self.params["device"] == "cuda":
            self.params["cb_task_type"] = 'GPU'
        else:
            self.params["cb_task_type"] = None
            
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        elif self.params["eval_metric"] in ["auc", "gini"]:
            self.params["eval_metric"] = "AUC"
        elif self.params["eval_metric"]=="rmse":
            self.params["eval_metric"] = "RMSE"
        elif self.params["eval_metric"]=="rmsle":
            self.params["eval_metric"] = "MSLE"
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "R2"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "Logloss"
        elif self.params["eval_metric"]=="mlogloss":
            self.params["eval_metric"] = "MultiClass"
        elif self.params["eval_metric"]=="logloss":
            self.params["eval_metric"] = "Logloss"
        elif self.params["eval_metric"]=="mae":
            self.params["eval_metric"] = "MAE"
        elif self.params["eval_metric"]=="norm_gini":
            self.params["eval_metric"] = "NormalizedGini"
        elif self.params["eval_metric"]=="multilogloss":
            self.params["eval_metric"] = "MultiLogloss"

        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()       
        
        if self.params["task_type"] == "regression":
            self.model = cb.CatBoostRegressor(
                **self.params["hyperparameters"],
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["folds_parallel"]),
                train_dir='./logs/catboost/'
            )
        elif self.params["task_type"] == "binary":
            self.model = cb.CatBoostClassifier(
                **self.params["hyperparameters"],
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["folds_parallel"]),
                train_dir='./logs'
            )
        elif self.params["task_type"] == "classification":
            self.model = cb.CatBoostClassifier(
                classes_count=self.params["num_classes"],
                **self.params["hyperparameters"],
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["folds_parallel"]),
                train_dir='./logs'
            )
        elif self.params["task_type"] == "multilabel":
            self.model = cb.CatBoostClassifier(
                # classes_count=self.params["num_classes"],
                loss_function='MultiLogloss',
                **self.params["hyperparameters"],
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["folds_parallel"]),
                train_dir='./logs'
            )
            
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        X_train_use = X_train.copy()
        y_train_use = y_train.copy()
        self.cat_col_names = list(X_train_use.iloc[:,self.params["cat_indices"]].columns)
        self.cat_dtypes = {}
        if eval_set is not None:
            X_val_use = eval_set[0][0].copy()
            y_val_use = eval_set[0][1].copy()
        
            for col in self.cat_col_names:
                X_train_use[col] = X_train_use[col].astype(str).fillna("nan")
                X_val_use[col] = X_val_use[col].astype(str).fillna("nan")      
            eval_set = [(X_val_use,y_val_use)]
        else:
            for col in self.cat_col_names:
                X_train_use[col] = X_train_use[col].astype(str).fillna("nan")
            eval_set = [(X_train_use,y_train_use)]

        h = self.model.fit(
            X_train_use, y_train_use, 
            eval_set=eval_set,
            cat_features=self.cat_col_names,
            use_best_model=True
        )
    
    def predict(self, X):
        X_use = X.copy()
        for col in self.cat_col_names:
            X_use.loc[:,col] = X_use.loc[:,col].astype(str).fillna("nan")

        if self.params["task_type"]=="regression":
            pred = self.model.predict(X_use)
        elif self.params["task_type"]=="binary":
            pred = self.model.predict_proba(X_use)[:,1]            
        elif self.params["task_type"]=="classification":
            pred = self.model.predict_proba(X_use)            
        
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, n_features=1, dataset_name="", **kwargs):
        # Limit max_depth for too large datasets
        if dataset_name=="santander-value-prediction-challenge":
            max_depth = 6
        else:
            max_depth = 11
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "depth": trial.suggest_int("depth", 1, max_depth), # Max depth set to 11 because 12 fails for santander value dataset on A6000
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            # "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            # "max_leaves": trial.suggest_categorical("max_leaves", [5,10,15,20,25,30,35,40,45,50,55,60]),
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1),
        }        
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            # "learning_rate": 0.08, #0.08
            # "depth": 5, #5
            # "l2_leaf_reg": 5,
            # "bagging_temperature": 1,
            # "leaf_estimation_iterations": 1
        }        
        
        return params    
        
    

class GRANDEModelTF:
    def __init__(self, params):
        self.params = params
        self.task_type = self.params["task_type"]
        
        # Tunable hyperparameters
        hyperparameters = self.get_default_hyperparameters()    

        if self.task_type == 'regression':
            hyperparameters['loss'] = 'mse'
        
        if "hyperparameters" in self.params:
            hyperparameters.update(self.params["hyperparameters"])

        if "epochs" in self.params:
            hyperparameters["epochs"] = self.params["epochs"]

        if "verbose" in self.params:
            hyperparameters["verbose"] = self.params["verbose"]

        self.params["hyperparameters"] = hyperparameters

        params_grande = {
                'depth': self.params["hyperparameters"]["depth"],
                'n_estimators': self.params["hyperparameters"]["n_estimators"],
        
                'learning_rate_weights': self.params["hyperparameters"]["learning_rate_weights"],
                'learning_rate_index': self.params["hyperparameters"]["learning_rate_index"],
                'learning_rate_values': self.params["hyperparameters"]["learning_rate_values"],
                'learning_rate_leaf': self.params["hyperparameters"]["learning_rate_leaf"],
        
                'optimizer': self.params["hyperparameters"]["optimizer"],
                'cosine_decay_steps': self.params["hyperparameters"]["cosine_decay_steps"],
        
                'loss': self.params["hyperparameters"]["loss"],
                'focal_loss': self.params["hyperparameters"]["focal_loss"],
                'temperature': self.params["hyperparameters"]["temperature"],
        
                'from_logits': self.params["hyperparameters"]["from_logits"],
                'use_class_weights': self.params["hyperparameters"]["use_class_weights"],
        
                'dropout': self.params["hyperparameters"]["dropout"],
        
                'selected_variables': self.params["hyperparameters"]["selected_variables"],
                'data_subset_fraction': self.params["hyperparameters"]["data_subset_fraction"],
        }
        
        args_grande = {
            'epochs': self.params["hyperparameters"]["epochs"],
            'early_stopping_epochs': self.params["hyperparameters"]["early_stopping_epochs"],
            'batch_size': self.params["hyperparameters"]["batch_size"],
        
            'cat_idx': self.params["cat_indices"],
            'objective': self.params["task_type"],
            
            'random_seed': self.params["hyperparameters"]["random_seed"],
            'verbose': self.params["hyperparameters"]["verbose"],
        }
        with tf.device('/GPU:0'):
            self.model = GRANDE(params_grande, args_grande)
        
    def fit(self, 
            X_train, 
            y_train, 
            eval_set):

        X_val = eval_set[0][0]
        y_val = eval_set[0][1]     

        warnings.filterwarnings("ignore")
        self.model.fit(X_train, 
                    y_train, 
                    X_val,
                    y_val)

        

    def predict(self, X):
        preds = self.model.predict(X)       

        if self.task_type == 'binary':
            preds = preds[:,1]
        if self.task_type == 'regression':
            preds = np.squeeze(preds, axis=1)

        return preds
        
    @classmethod
    def get_optuna_hyperparameters(self, trial, **kwargs):
        params = {
            #'depth': trial.suggest_int("depth", 3, 7),
            #'n_estimators': trial.suggest_int("n_estimators", 512, 2048),

            'learning_rate_weights':  trial.suggest_float("learning_rate_weights", 0.0001, 0.25, log=True),
            'learning_rate_index': trial.suggest_float("learning_rate_index", 0.0001, 0.25, log=True),
            'learning_rate_values': trial.suggest_float("learning_rate_values", 0.0001, 0.25, log=True),
            'learning_rate_leaf': trial.suggest_float("learning_rate_leaf", 0.0001, 0.25, log=True),

            'cosine_decay_steps': trial.suggest_categorical("cosine_decay_steps", [0, 100, 1000]),
            
            'dropout': trial.suggest_categorical("dropout", [0, 0.25]),

            'selected_variables': trial.suggest_categorical("selected_variables", [1.0, 0.75, 0.5]),
            #'data_subset_fraction': trial.suggest_categorical("data_subset_fraction", [1.0, 0.8]),
        }

        try:
            if self.params["task_type"] != 'regression':
                params['focal_loss'] = trial.suggest_categorical("focal_loss", [True, False])
                params['temperature'] = trial.suggest_categorical("temperature", [0, 0.25])
        except:
            pass
            
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
  
        params = {
            'depth': 5,
            'n_estimators': 2048,

            'learning_rate_weights': 0.005,
            'learning_rate_index': 0.01,
            'learning_rate_values': 0.01,
            'learning_rate_leaf': 0.01,

            'optimizer': 'adam',
            'cosine_decay_steps': 0,
            'temperature': 0.0,

            'initializer': 'RandomNormal',

            'loss': 'crossentropy',
            'focal_loss': False,

            'from_logits': True,
            'use_class_weights': True,

            'dropout': 0.0,

            'selected_variables': 0.8,
            'data_subset_fraction': 1.0,
            'bootstrap': False,

            'random_seed': 42,
            'epochs': 1_000,
            'early_stopping_epochs': 25,
            'batch_size': 64,         
            'verbose': 1,
        }     
        
        return params    



###################################################################################
###################################################################################
###################################################################################

class LightGBMModel(BaseModel):
    def __init__(self, params):
        
        super().__init__(params)
        # Not tunable parameters
        # Do not use GPU here, overwrite existing parameter
        self.params["device"] = "cpu"

        if self.params["task_type"] == "classification":
            self.params["task_type"] = "multiclass"

        self.feval = None
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        elif self.params["eval_metric"] in ["auc", "gini"]:
            self.params["eval_metric"] = "auc"
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"] in ["logloss", "ams"]:
            self.params["eval_metric"] = "binary"
        elif self.params["eval_metric"] == "mlogloss":
            self.params["eval_metric"] = "multiclass"

        elif self.params["eval_metric"]=="rmsle":
            def rmsle_lgbm(y_pred, data):
            
                y_true = np.array(data.get_label())
                score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))
            
                return 'rmsle', score, False
            self.params["eval_metric"] = "custom"
            self.feval = rmsle_lgbm
            
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()
            
    def fit(self, 
            X_train, y_train, 
            eval_set,
            ):
        # X_train_use = X_train.copy()
        # y_train_use = y_train.copy()
        self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns.tolist()
        self.cat_dtypes = {}
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
        
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    # X_train.loc[:,col] = X_train[col].astype(object)
                    # X_val.loc[:,col] = X_val[col].astype(object)
                    # u_cats = list(X_train[col].unique())+["nan"] #np.unique(list(X_train[col].unique())+list(X_val[col].unique())+["nan"]).tolist()
                    # self.cat_dtypes[col] = pd.CategoricalDtype(categories=u_cats)
                    # X_train.loc[:,col] = X_train.loc[:,col].astype(self.cat_dtypes[col])
                    # X_val.loc[:,col] = X_val.loc[:,col].astype(self.cat_dtypes[col])

                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
                    X_val[col] = X_val[col].astype(self.cat_dtypes[col])            
                    
            eval_set = [(X_val,y_val)]
        else:
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
            eval_set = [(X_train,y_train)]
            
        params = {
            "n_estimators": self.params["n_estimators"],
            "objective": self.params["task_type"],
            "boosting_type": "gbdt",
            "num_class": self.params["d_out"],
            "metric": self.params["eval_metric"],
            "verbosity": -1,
            **self.params["hyperparameters"],
        }

        dtrain = lgbm.Dataset(X_train, y_train, categorical_feature=self.cat_col_names)
        if eval_set is not None:
            X_val_use = eval_set[0][0].copy()
            y_val_use = eval_set[0][1].copy()
            dvalid = lgbm.Dataset(X_val_use, y_val_use, reference=dtrain, categorical_feature=self.cat_col_names)
        else:
            dvalid = None

        if self.params["patience"] != None:
            callbacks = [lgbm.early_stopping(stopping_rounds=self.params["patience"])]
        else:
            callbacks = None

        self.model = lgbm.train(params, dtrain, valid_sets=dvalid, callbacks=callbacks, feval=self.feval)
    
    def predict(self, X):
        for col in self.cat_col_names:
            if X.loc[:,col].dtype!="category":
                X[col] = X[col].astype(self.cat_dtypes[col])
        dpred = lgbm.Dataset(X, categorical_feature=self.params["cat_indices"])

        pred = self.model.predict(X, num_iteration=self.model.best_iteration)         
        
        return pred    
    
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        # Tuning parameter ranges defined based on the descriptions in https://lightgbm.readthedocs.io/en/latest/Parameters.html and https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#add-more-computational-resources as well as the TabR paper
        
        
        
        # # Limit max_depth for too large datasets
        # if n_features>3000:
        #     max_depth = 9
        # else:
        #     max_depth = 11
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            # "num_leaves": trial.suggest_int("num_leaves", 2, (2**params["max_depth"])-1), # Max depth set to 11 because 12 fails for santander value dataset on A6000
            # "min_data_in_leaf": trial.suggest_float("lambda_l2", 0.1, 10., log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        """ porto seguro expert configuration
        params = {
            "learning_rate": 0.1,
                  "num_leaves": 15,
                  "max_bin": 256,
                  "feature_fraction": 0.6,
                  "verbosity": 1,
                  "drop_rate": 0.1,
                  "is_unbalance": False,
                  "max_drop": 50,
                  "min_child_samples": 10,
                  "min_child_weight": 150,
                  "min_split_gain": 0,
                  "subsample": 0.9,
                  "num_iterations": 10000
        }"""

        # default params of LightGBM module
        params = {
            # "learning_rate": 0.1,  Hyperparameters from expert porto-seguro solution
            # "num_leaves": 31,
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  