import os
import pickle
import datetime
import gc
import subprocess
import time
import io
import csv
import kaggle

from math import sqrt
import math
from math import ceil

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, OrdinalEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error

from utils import set_seed, get_metric, get_memory_usage, sizeof_fmt, reduce_mem_usage, merge_by_concat

from models import get_model

import xgboost as xgb

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

import openfe

def get_dataset(dataset_name, toy_example=False):

    if dataset_name=="santander-value-prediction-challenge":
        return SantanderValueDataset(toy_example)
    elif dataset_name=="mercedes-benz-greener-manufacturing":
        return MercedesBenzDataset(toy_example)
    elif dataset_name=="m5-forecasting-accuracy":
        return M5ForecastDataset(toy_example)
    elif dataset_name=="santander-customer-transaction-prediction":
        return SantanderTransactionDataset(toy_example)
    elif dataset_name=="ieee-fraud-detection":
        return IEEEFraudDetectionDataset(toy_example)
    elif dataset_name=="amazon-employee-access-challenge":
        return AmazonEmployeeAccessDataset(toy_example)
    elif dataset_name=="higgs-boson":
        return HiggsBosonDataset(toy_example)
    elif dataset_name=="santander-customer-satisfaction":
        return SantanderSatisfactionDataset(toy_example)
    elif dataset_name=="porto-seguro-safe-driver-prediction":
        return PortoSeguroDriverDataset(toy_example)
    elif dataset_name=="sberbank-russian-housing-market":
        return SberbankHousingDataset(toy_example)
    elif dataset_name == "walmart-recruiting-trip-type-classification":
        return WalmartRecruitingTripType(toy_example)
    elif dataset_name == "allstate-claims-severity":
        return AllstateClaimsSeverity(toy_example)
    elif dataset_name == "bnp-paribas-cardif-claims-management":
        return BNPParibasCardifClaimsManagement(toy_example)
    elif dataset_name == "restaurant-revenue-prediction":
        return RestaurantRevenuePrediction(toy_example)
    elif dataset_name == "home-credit-default-risk":
        return HomeCreditDefaultRisk(toy_example)
    elif dataset_name == "icr-identify-age-related-conditions":
        return ICRIdentifyAgeRelatedConditions(toy_example)
    elif dataset_name == "lish-moa":
        return MoAPrediction(toy_example)
    elif dataset_name == "zillow-prize-1":
        return ZillowPrice(toy_example)
    elif dataset_name == "otto-group-product-classification-challenge":
        return OttoGroupProductClassification(toy_example)
    elif dataset_name == "springleaf-marketing-response":
        return SpringleafMarketingResponse(toy_example)
    elif dataset_name == "prudential-life-insurance-assessment":
        return PrudentialLifeInsuranceAssessment(toy_example)
    elif dataset_name == "microsoft-malware-prediction":
        return MicrosoftMalwarePrediction(toy_example)
    elif dataset_name == "homesite-quote-conversion":
        return HomesiteQuoteConversion(toy_example)
    elif dataset_name == "predicting-red-hat-business-value":
        return PredictingRedHatBusinessValue(toy_example)
    elif dataset_name == "talkingdata-mobile-user-demographics":
        return TalkingdataMobileUserDemographics(toy_example)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not implemented.")

class BaseDataset:
    ''' All implemented datasets should inherit from this base dataset to maintain the same structure. 
    
    All new datasets should define own init methods which include the following dataset-specific instances:
    - dataset_name: name of the dataset (defined by the name of the Kaggle competition)
    - task_type: "regression", "binary", or "classification"
    - eval_metric_name: name of evaluation metric (needs to be a metric included in utils.py)
    - cat_indices: list of indices for categorical features            
    - y_col: name of the target column
    
    Whenever the logic does not follow the BaseDataset, new datasets must implement the following functions:
    - load_data: All steps required to obtain a single table as raw as possible
    - pred_to_submission: Transform predicted test data values to correct format for submitting
    - get_cv_folds: CV procedure representing the dataset-specific expert CV procedure
    
    Ideally, new datasets would also implement dataset-specific expert preprocessing
    - expert_preprocessing: All preprocessing and feature engineering steps in the pipeline of a high-ranked expert solution
    
    
    '''
    def __init__(self, toy_example=False):
        self.toy_example = toy_example
        
        self.dataset_name = ""
        self.cat_indices = []
        self.y_col = ""
        self.heavy_tailed = False
        self.preprocess_states = []
        
        # experimental
        self.trial_budget = 100
        self.batch_size = 128
        self.x_scaled = False
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        y_train = data[self.y_col]
        X_train = data.drop(self.y_col,axis=1)    

        if self.task_type== "classification":
            self.target_label_enc = LabelEncoder()
            y_train = pd.Series(self.target_label_enc.fit_transform(y_train),index=y_train.index, name=y_train.name)
            self.num_classes = y_train.nunique()
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    
    def minimalistic_preprocessing(self, X_train, X_test, y_train, 
                                   scaler=None, one_hot_encode=False, use_test=True):
        '''Preprocessing based on McElfresh et al. 2023
        - Define categorical feature types
        - Fill missing values with mean
        - Optionally: scale numeric features or apply OHE to categoricals
        
        '''

        print("Apply minimalistic preprocessing")
        
        # Encode binary cat features as numeric
        for col in X_train.columns[X_train.nunique()==2]:
            if X_train[col].dtype in [str, "O", "category", "object"]:
                le = LabelEncoder()
                mode = X_train[col].mode()[0]
                X_train[col] = le.fit_transform(X_train[col])

                if len(X_test[col].unique())==2:
                    X_test[col] = le.transform(X_test[col])
                else:
                    X_test[col] = X_test[col].fillna(mode)
                    X_test[col] = le.transform(X_test[col])
                
        
        # Define categorical feature types
        self.cat_indices += list(np.where(X_train.dtypes=="O")[0]) 
        self.cat_indices += list(np.where(X_train.dtypes=="object")[0]) 
        self.cat_indices += list(np.where(X_train.dtypes=="category")[0]) 
        self.cat_indices = np.unique(self.cat_indices).tolist()
        
        for num, col in list(zip(self.cat_indices,X_train.columns[self.cat_indices])):
            # Encode binary categorical features
            if X_train[col].nunique()==2:
                value_1 = X_train[col].dropna().unique()[0]
                X_train[col] = (X_train[col]==value_1).astype(float)
                X_test[col] = (X_test[col]==value_1).astype(float)
                self.cat_indices.remove(num)
            else:
                # Note: The category dtype needs to entail all train categories when applying .astype("category") on test data
                dtype = pd.CategoricalDtype(categories=list(X_train[col].astype(str).fillna("nan").unique()))
                X_train[col] = X_train[col].astype(str).fillna("nan").astype(dtype)
                X_test[col] = X_test[col].astype(str).fillna("nan").astype(dtype)       
                
        
        cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in self.cat_indices])
        cont_col_names = X_train.iloc[:,cont_indices].columns
        
        X_concat = pd.concat([X_train, X_test])
        
        # Fill missing values of continuous columns with mean 
        if X_train.isna().sum().sum()>0:
            if use_test:
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_concat[cont_col_names].mean())
            else:
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_train[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_train[cont_col_names].mean())
            
        # if scaler is not None:
        #     X_train[cont_col_names] = scaler_function.fit_transform(X_train[cont_col_names])
        #     X_test[cont_col_names] = scaler_function.transform(X_test[cont_col_names])

        # if one_hot_encode:
        #     ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        #     new_x1 = ohe.fit_transform(X_train[:, self.cat_indices])
        #     X_train = np.concatenate([new_x1, X_train[:, num_mask]], axis=1)
        #     new_x1_test = ohe.transform(X_test[:, self.cat_indices])
        #     X_test = np.concatenate([new_x1_test, X_test[:, num_mask]], axis=1)
            
        #     self.cat_indices = []
            
        # Drop constant columns
        # drop_cols = X_train.columns[X_train.nunique()==X_train.shape[0]].values.tolist()
        drop_cols = X_train.columns[X_train.nunique()==1].values.tolist()
        if len(drop_cols)>0:
            print(f"Drop {len(drop_cols)} constant/unique features")
            original_categorical_names =  X_train.columns[self.cat_indices]
            X_train.drop(drop_cols,axis=1,inplace=True)
            X_test.drop(drop_cols,axis=1,inplace=True)
            self.cat_indices = [np.where(X_train.columns==i)[0][0] for i in original_categorical_names if i in X_train.columns]
        
        if self.heavy_tailed: # Todo: Might move to minimalistic
            y_train = np.log1p(y_train)

        self.preprocess_states.append("minimalistic")     
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    def minimalistic_postprocessing(self, X_train, y, **kwargs):
        if self.task_type=="regression":
            if self.heavy_tailed:
                y = np.expm1(y)
        return y

    def openfe_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False):
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_openfe.pickle") or overwrite_existing:
            print("Apply OpenFE preprocessing")
            import warnings
            warnings.filterwarnings("ignore")

            task = "regression" if self.task_type == "regression" else "classification"
    
            cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in self.cat_indices])
            cont_col_names = X_train.iloc[:,cont_indices].columns.values.tolist()
            if len(self.cat_indices)>0:
                cat_col_names = X_train.columns[self.cat_indices]
            else:
                cat_col_names = None
            
            candidate_features_list = openfe.get_candidate_features(numerical_features=cont_col_names, categorical_features=cat_col_names, order=1)

            ofe = openfe.OpenFE()
            features = ofe.fit(data=X_train, label=y_train, n_jobs=os.cpu_count(), task=task, n_data_blocks=8, 
                               candidate_features_list=candidate_features_list,
                               stage2_params={"verbose": -1},
                               verbose=True, tmp_save_path=f'./openfe_tmp_data_{self.dataset_name}.feather')
                   

            X_train_new, X_test_new = openfe.transform(X_train, X_test, features, n_jobs=os.cpu_count())

            is_combined = [f.name=='Combine' for f in features]
            if sum(is_combined)>0:
                self.cat_indices += list(np.where([f.name=='Combine' for f in features])[0]+X_train.shape[1])

            self.X_train, self.X_test = X_train_new, X_test_new
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(self.X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'wb'))            
            pickle.dump(self.X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'wb'))


        else:
            print(f"Load existing openFE-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'rb'))
            self.X_train, self.X_test = X_train, X_test

        self.preprocess_states.append("openfe")
        
    def automated_feature_engineering(self, X_train, X_test, y_train):
        '''Preprocessing with openFE'''

        self.preprocess_states.append("autoFE")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, **kwargs):
        print("Expert preprocessing not implemented yet")
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    
        
    def neuralnet_preprocessing(self, X_train, X_test, y_train, use_test=True):
        if self.task_type=="regression":
            # if self.heavy_tailed: # Todo: Might move to minimalistic
            #     y_train = np.log1p(y_train)
            
            self.target_scaler = StandardScaler()
            y_train = pd.Series(self.target_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel(),
              name=self.y_col, index = X_train.index)

        # Drop constant columns
        # drop_cols = X_train.columns[X_train.nunique()==X_train.shape[0]].values.tolist()
        drop_cols = X_train.columns[X_train.nunique()==1].values.tolist()
        if len(drop_cols)>0:
            print(f"Drop {len(drop_cols)} constant/unique features")
            original_categorical_names =  X_train.columns[self.cat_indices]
            X_train.drop(drop_cols,axis=1,inplace=True)
            X_test.drop(drop_cols,axis=1,inplace=True)
            self.cat_indices = [np.where(X_train.columns==i)[0][0] for i in original_categorical_names if i in X_train.columns]

        # Drop constant nan cols
        if len(self.cat_indices)==0: # 
            nan_cols = X_train.columns[X_train.isna().sum()==X_train.shape[0]].values.tolist()
            if len(nan_cols)>0:
                print(f"Drop {len(nan_cols)} all-nan features")
            #     original_categorical_names =  X_train.columns[self.cat_indices]
                X_train.drop(nan_cols,axis=1,inplace=True)
                X_test.drop(nan_cols,axis=1,inplace=True)
                # self.cat_indices = [np.where(X_train.columns==i)[0][0] for i in original_categorical_names if i in X_train.columns]

        X_concat = pd.concat([X_train, X_test])
        
        cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in self.cat_indices])
        cont_col_names = X_train.iloc[:,cont_indices].columns

        X_concat[cont_col_names] = X_concat[cont_col_names].astype(np.float32)
        X_train[cont_col_names] = X_train[cont_col_names].astype(np.float32)
        X_test[cont_col_names] = X_test[cont_col_names].astype(np.float32)
        
        # Apply ordinal encoding to all categorical features
        if len(self.cat_indices)>0:
            cat_col_names = X_train.iloc[:,self.cat_indices].columns
            for col in cat_col_names:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", 
                                     unknown_value=X_train[col].nunique(),
                                     encoded_missing_value=X_train[col].nunique()
                                    )
                X_train[col] = enc.fit_transform(X_train[col].values.reshape(-1,1)).astype(int)
                X_test[col] = enc.transform(X_test[col].values.reshape(-1,1)).astype(int)

        # Fill missing values of continuous columns with mean 
        if X_train.isna().sum().sum()>0 or X_test.isna().sum().sum()>0:
            if use_test:
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_concat[cont_col_names] = X_concat[cont_col_names].fillna(X_concat[cont_col_names].mean())
            else:
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_train[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_train[cont_col_names].mean())
        
        if X_train.shape[1]!=len(self.cat_indices):
            if not self.x_scaled:
                # self.x_scaler = QuantileTransformer(
                #             n_quantiles= 1000
                #         )        
                # X_train[cont_col_names] = self.x_scaler.fit_transform(X_train[cont_col_names])
                # X_test[cont_col_names] = self.x_scaler.transform(X_test[cont_col_names])
    
                quantile_noise = 1e-4
                if use_test:
                    quantile_use = np.copy(X_concat[cont_col_names].values).astype(np.float64)
                else:
                    quantile_use = np.copy(X_train[cont_col_names].values).astype(np.float64)
                    
                stds = np.std(quantile_use, axis=0, keepdims=True)
                noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                quantile_use += noise_std * np.random.randn(*quantile_use.shape)    
                if use_test:
                    quantile_use = pd.DataFrame(quantile_use, columns=cont_col_names, index=X_concat.index)
                else:
                    quantile_use = pd.DataFrame(quantile_use, columns=cont_col_names, index=X_train.index)
                
                self.x_scaler = QuantileTransformer(
                    n_quantiles=min(quantile_use.shape[0], 1000),
                    output_distribution='normal')
    
                self.x_scaler.fit(quantile_use.values.astype(np.float64))
                X_train[cont_col_names] = self.x_scaler.transform(X_train[cont_col_names].values.astype(np.float64))
                X_test[cont_col_names] = self.x_scaler.transform(X_test[cont_col_names].values.astype(np.float64))
            
                self.x_scaled = True
        
        self.preprocess_states.append("neuralnet")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    

    def expert_postprocessing(self, X_train, y, **kwargs):
        return y
    
    def neuralnet_postprocessing(self, X, y):
        if self.task_type=="regression":
            if isinstance(y, pd.Series):
                y = y.values.reshape(-1,1)
            y = pd.Series(self.target_scaler.inverse_transform(y.reshape(-1,1)).ravel(),
              name=self.y_col, index = X.index)
            # if self.heavy_tailed:
            #     y = np.expm1(y)
            
        return y    
    
    def get_cv_folds(self, X_train, y_train, seed=42):
        ### !! Currently not original implemented - original solution used 30-fold CV - but also dicusses 5-fold
        ss = KFold(n_splits=10, random_state=seed, shuffle=True)
        folds = []
        for num, (train,test) in enumerate(ss.split(y_train.copy(), y_train.copy())):
            folds.append([train, test])    
        return folds
        
    def pred_to_submission(self, y_pred):
        try:
            submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sample_submission.csv", engine="pyarrow")
        except:
            submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sampleSubmission.csv", engine="pyarrow")

        if self.toy_example:
            submission = submission.iloc[:1000]
        submission[self.y_col] = y_pred

        return submission

    def submit_data(self, file_name):
        '''
        It might be important to make the users aware of the following warning: 
        'Warning: Your Kaggle API key is readable by other users on this system! To fix this', " you can run 'chmod 600 /home/atschalz/.kaggle/kaggle.json'
        '''
        # submit file to kaggle for evaluation
        os.system(f"kaggle competitions submit -c {self.dataset_name} -f {file_name} -m 'submitted from python script'")
    
        # get all submissions and their scores the user has made
        # wait for 5 seconds to ensure that the submission is processed (Todo: Might better be captured with a while loop)
        processed = False
        cnt = 0
        while not processed and cnt < 10:
            time.sleep(5)
            try:
                command = f"kaggle competitions submissions --csv {self.dataset_name}"
                shell_output = subprocess.check_output(command, shell=True, text=True)
            
                # parse the shell output to a dataframe
                csv_file = io.StringIO(shell_output)
                reader = csv.reader(csv_file)
                data = list(reader)
        
                while data[0][0][:7]=="Warning":
                    print(data[0])
                    data = data[1:]
                if data[0]=="Error":
                    print(data[0])
                    exit()
                # submissions_df = pd.DataFrame(data[1:], columns=data[0])
                submissions_df = pd.DataFrame(kaggle.api.competitions_submissions_list(self.dataset_name))        
        
                public_score = float(submissions_df.loc[submissions_df.fileNameNullable==file_name.split("/")[-1],"publicScoreNullable"].iloc[0])
                private_score = float(submissions_df.loc[submissions_df.fileNameNullable==file_name.split("/")[-1],"privateScoreNullable"].iloc[0])
                processed = True
            except:
                print(f"{cnt} Waited for 5 seconds, but submission was not processed correctly")
                cnt += 1

        
        # get the public and private leaderboard to compute the rank of the submission
        leaderboard_df =pd.read_csv(f"./datasets/{self.dataset_name}/leaderboard.csv")
    
        # compute the public rank and percentile of the submission
        lb_with_own = np.array(sorted(list(leaderboard_df.PublicScore.astype(float))+[public_score],reverse=True))
        if self.eval_metric_direction=="minimize":
            lb_with_own = lb_with_own[::-1]
        public_rank = np.where(lb_with_own==public_score)[0][0]+1
        public_percentile = (public_rank / len(leaderboard_df)) 

        # compute the private rank and percentile of the submission
        lb_with_own = np.array(sorted(list(leaderboard_df.PrivateScore.astype(float))+[private_score],reverse=True))
        if self.eval_metric_direction=="minimize":
            lb_with_own = lb_with_own[::-1]
        private_rank = np.where(lb_with_own==private_score)[0][0]+1
        private_percentile = (private_rank / len(leaderboard_df)) 
    
        return public_score, private_score, public_rank, public_percentile, private_rank, private_percentile
    
##########################################################   
##########################################################   
##########################################################        
        
class MercedesBenzDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "mercedes-benz-greener-manufacturing"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "regression" # "binary", "classification"
        self.eval_metric_name = "r2"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [1,2,3,4,5,6,7,8] 
        self.y_col = "y"
        self.large_dataset = False
        
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, cat_method=None, **kwargs):
        '''
        Solution implemented based on the descriptions in https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing/discussion/37700
        
        1. Preprocessing:
            - instead of throwing out outliers, clipped all y's at 155. 155 was selected from a visual inspection of y's distribution
        2. Feature Engineering
            - 'ID', and 'X0' as a single factorized array (noticed that I would have scored .55571 (private LB) if I had not included 'XO' as a single factorized array)
            - X0: 15 of the 47 unique categories
            - X1: 6 of the 27 unique categories
            - X2: 13 of the 44 unique categories
            - X3: 2 of the 7 unique categories
            - X4: no categories
            - X5: 9 of the 29 unique categories
            - X6: 4 of the 12 unique categories
            - X8: 5 of the 25 unique categories
            - X10 - X385: 78 of the 357 binary features
        3. CV: simple 5-fold KFold with shuffle set to true and the same random seed
            - assumed CV scores within .001 of each other to be effectively equal.
        4. Ensemble: stacked ensemble including GradientBoostingRegressor, RandomForestRegressor, and SVR

        '''
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"

        if cat_method is not None:
            dataset_version = dataset_version+"_"+cat_method
            
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            ### Create interaction features
            X_train["X314-315"] = X_train["X314"]+X_train["X315"]
            X_train["X118-314-315"] = X_train["X118"]+X_train["X314"]+X_train["X315"]
            X_train["X118-314-315-levels110"] = np.logical_and(np.logical_and(X_train["X118"]==1,X_train["X314"]==1), X_train["X315"]==0)*1
            X_train["X47-48"] = X_train["X47"]+X_train["X48"]
            
            X_test["X314-315"] = X_test["X314"]+X_test["X315"]
            X_test["X118-314-315"] = X_test["X118"]+X_test["X314"]+X_test["X315"]
            X_test["X118-314-315-levels110"] = np.logical_and(np.logical_and(X_test["X118"]==1,X_test["X314"]==1), X_test["X315"]==0)*1
            X_test["X47-48"] = X_test["X47"]+X_test["X48"]
            
            ### Create a feature based on considerations of subprocesses in data creation
            X_train["sum_122_128"] = X_train[["X122","X123","X124","X125","X126","X127","X128"]].sum(axis=1)
            X_test["sum_122_128"] = X_test[["X122","X123","X124","X125","X126","X127","X128"]].sum(axis=1)
            
            ### Use only the generated features, X0 and the six features found to be most important
            ### Use index as feature helps
            use_features = ["ID", "X314-315","X118-314-315","X118-314-315-levels110", "sum_122_128", "X47-48", "X0", "X314", "X279", "X232", "X261", "X29", "X127"]
            
            X_train = X_train[use_features]
            X_test = X_test[use_features]
    
            ### One-hot-encode X0
            if cat_method == "model":
                self.cat_indices = list(np.where(X_train.columns=="X0")[0])
    
            else:
                ohe = OneHotEncoder(handle_unknown='ignore')
                cat = "X0"
                x = X_train[cat]
        
                ohe.fit(x.values.reshape(-1,1))
                X_train_ohe = pd.DataFrame(ohe.transform(x.values.reshape(-1,1)).toarray(),
                                     index = X_train.index,
                                     columns=cat+"_"+ohe.categories_[0])
        
                X_train = pd.concat([X_train,X_train_ohe],axis=1)
        
                X_test_ohe = pd.DataFrame(ohe.transform(X_test[[cat]]).toarray(),
                                     index = X_test.index,
                                     columns=cat+"_"+ohe.categories_[0])
        
                X_test = pd.concat([X_test,X_test_ohe],axis=1)        
                
                X_train.drop(cat,inplace=True,axis=1)
                X_test.drop(cat,inplace=True,axis=1)
            
                self.cat_indices = []
            # Transform continuous to floats
            # cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in self.cat_indices])
            # cont_col_names = X_train.iloc[:,cont_indices].columns
            # X_train.loc[:,cont_col_names] = X_train[cont_col_names].astype(float) 
            
            ### One-hot-encode categorical features
    #         cat_features = [f"X{i}" for i in range(9) if i!=7]
    #         bin_features = list(set(X_train.columns)-set(cat_features))
    
    #         for cat in cat_features: 
    #             ohe = OneHotEncoder(handle_unknown='ignore')
    #             ohe.fit(X_train[[cat]])
    #             X_train_ohe = pd.DataFrame(ohe.transform(X_train[[cat]]).toarray(),
    #                                  index = X_train.index,
    #                                  columns=cat+"_"+ohe.categories_[0])
    
    #             X_train = pd.concat([X_train,X_train_ohe],axis=1)
    #             X_train.drop(cat,inplace=True,axis=1)
    
    #             X_test_ohe = pd.DataFrame(ohe.transform(X_test[[cat]]).toarray(),
    #                                  index = X_test.index,
    #                                  columns=cat+"_"+ohe.categories_[0])
    
    #             X_test = pd.concat([X_test,X_test_ohe],axis=1)
    #             X_test.drop(cat,inplace=True,axis=1)
    
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))
                
            ### Clip target to 155 (from second place solution)
    #         y_train[y_train>155] = 155.         
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            try:
                self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
            except:
                self.cat_indices = []
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train  

    def openfe_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False):
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_openfe.pickle") or overwrite_existing:
            print("Apply OpenFE preprocessing")
            import warnings
            warnings.filterwarnings("ignore")

            task = "regression" if self.task_type == "regression" else "classification"
    
            numeric_features = ["X314","X315","X118", "X122", "X123", "X124", "X125", "X126", "X127","X128", "X47","X48", "X279", "X232", "X261", "X29"]
            
            candidate_features_list = openfe.get_candidate_features(numerical_features=numeric_features, categorical_features=["X0"], order=1)

            ofe = openfe.OpenFE()
            features = ofe.fit(data=X_train, label=y_train, n_jobs=os.cpu_count(), task=task, n_data_blocks=8, 
                               candidate_features_list=candidate_features_list,
                               stage2_params={"verbose": -1},
                               verbose=True, tmp_save_path=f'./openfe_tmp_data_{self.dataset_name}.feather')
                   

            X_train_new, X_test_new = openfe.transform(X_train, X_test, features, n_jobs=os.cpu_count())

            is_combined = [f.name=='Combine' for f in features]
            if sum(is_combined)>0:
                self.cat_indices += list(np.where([f.name=='Combine' for f in features])[0]+X_train.shape[1])

            self.X_train, self.X_test = X_train_new, X_test_new
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(self.X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'wb'))            
            pickle.dump(self.X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'wb'))


        else:
            print(f"Load existing openFE-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'rb'))
            self.X_train, self.X_test = X_train, X_test

        self.preprocess_states.append("openfe")

    
##########################################################   
##########################################################   
##########################################################    

# TODO: Correctly implement m5-forecasting-accuracy
class M5ForecastDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "m5-forecasting-accuracy"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "regression"
        self.eval_metric_name = "rmsse"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []            
        self.y_col = "sales"
        self.large_dataset = False
        
    def load_data(self):
        '''
        1. Load the three data tables  'sales_train_evaluation', 'prices_df', and 'releases_df'
        2. Unpivote the main table to a vertical grid - each day becomes a separate row instead of column
        3. Add placeholder test predictions to the grid to be able to make predictions
        4. Add information from prices_df
            - group the prices_df by store_id and item_id to find the week when an item was first sold in a specific store (wm_yr_wk)
            - The result of that is a temporary dataframe release_df
            - The release_df is merged with the grid_df
            - all the rows which report the sale of the item in a store before it was first released in that store are deleted as they are 0
        5. Create 10 new features based on the sell price
        6. Merge 9 of 14 of the columns of the *calender_df* with the *grid_df*
        7. Convert columns to boolean or datetime where necessary
        8. Create datetime features
        '''

        ### Load data
        train_df = pd.read_csv(f'./datasets/{self.dataset_name}/raw/sales_train_evaluation.csv', engine="pyarrow")
        prices_df = pd.read_csv(f'./datasets/{self.dataset_name}/raw/sell_prices.csv', engine="pyarrow")
        calendar_df = pd.read_csv(f'./datasets/{self.dataset_name}/raw/calendar.csv', engine="pyarrow")
        
        
        TARGET = 'sales'         # Our main target
        END_TRAIN = 1941         # Last day in train set
        MAIN_INDEX = ['id','d']  # We can identify item by these columns        
                
        ### Unpivote data / transform to vertical view
        index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
        
        if not os.path.exists(f'./datasets/{self.dataset_name}/processed/grid_part_1.pkl'):
            
        
            grid_df = pd.melt(train_df, 
                            id_vars = index_columns, 
                            var_name = 'd', 
                            value_name = TARGET)
            ### create a temporary dataframe for each day and merge them into one temporary dataframe
            # To be able to make predictions
            # we need to add "test set" to our grid
            add_grid = pd.DataFrame()
            for i in range(1,29):
                temp_df = train_df[index_columns]
                temp_df = temp_df.drop_duplicates()
                temp_df['d'] = 'd_'+ str(END_TRAIN+i)
                temp_df[TARGET] = np.nan
                add_grid = pd.concat([add_grid,temp_df])

            grid_df = pd.concat([grid_df,add_grid])
            grid_df = grid_df.reset_index(drop=True)

            # Remove some temoprary DFs
            del temp_df, add_grid

            # We will not need original train_df
            # anymore and can remove it
            del train_df

            # You don't have to use df = df construction
            # you can use inplace=True instead.
            # like this
            # grid_df.reset_index(drop=True, inplace=True)

            # Let's check our memory usage
            print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

            # We can free some memory 
            # by converting "strings" to categorical
            # it will not affect merging and 
            # we will not lose any valuable data
            for col in index_columns:
                grid_df[col] = grid_df[col].astype('category')

            # Let's check again memory usage
            print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))        
            ### group the *prices_df* by *store_id* and *item_id* to find the  week when an item was first sold in a specific store (*wm_yr_wk*)
            ########################### Product Release date        #################################################################################
            print('Release week')

            # It seems that leadings zero values
            # in each train_df item row
            # are not real 0 sales but mean
            # absence for the item in the store
            # we can safe some memory by removing
            # such zeros

            # Prices are set by week
            # so it we will have not very accurate release week 
            release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
            release_df.columns = ['store_id','item_id','release']

            # Now we can merge release_df
            grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
            del release_df

            # We want to remove some "zeros" rows
            # from grid_df 
            # to do it we need wm_yr_wk column
            # let's merge partly calendar_df to have it
            grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])

            # Now we can cutoff some rows 
            # and safe memory 
            grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
            grid_df = grid_df.reset_index(drop=True)

            # Let's check our memory usage
            print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

            # Should we keep release week 
            # as one of the features?
            # Only good CV can give the answer.
            # Let's minify the release values.
            # Min transformation will not help here 
            # as int16 -> Integer (-32768 to 32767)
            # and our grid_df['release'].max() serves for int16
            # but we have have an idea how to transform 
            # other columns in case we will need it
            grid_df['release'] = grid_df['release'] - grid_df['release'].min()
            grid_df['release'] = grid_df['release'].astype(np.int16)

            # Let's check again memory usage
            print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

            ########################### Save part 1
            #################################################################################
            print('Save Part 1')

            # We have our BASE grid ready
            # and can save it as pickle file
            # for future use (model training)
            grid_df.to_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_1.pkl')
        else:
            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_1.pkl')
        print('Size:', grid_df.shape)        
        
        if not os.path.exists(f'./datasets/{self.dataset_name}/processed/grid_part_2.pkl'):

            ### Create ten new features absed on the sell price
            print('Prices')

            # We can do some basic aggregations
            prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
            prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
            prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
            prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

            # and do price normalization (min/max scaling)
            prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

            # Some items are can be inflation dependent
            # and some items are very "stable"

            prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique') 
            prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

            # I would like some "rolling" aggregations
            # but would like months and years as "window"
            calendar_prices = calendar_df[['wm_yr_wk','month','year']]
            calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk']) # distinct(.keep_all = True)
            prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
            del calendar_prices

            # Now we can add price "momentum" (some sort of)
            # Shifted by week 
            # by month mean
            # by year mean
            prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
            prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
            prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

            del prices_df['month'], prices_df['year']

            grid_df = reduce_mem_usage(grid_df)
            prices_df = reduce_mem_usage(prices_df)

            ########################### Merge prices and save part 2
            #################################################################################
            print('Merge prices and save part 2')

            # Merge Prices
            original_columns = list(grid_df)
            grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
            keep_columns = [col for col in list(grid_df) if col not in original_columns]
            grid_df = grid_df[MAIN_INDEX+keep_columns]
            grid_df = reduce_mem_usage(grid_df)

            # Safe part 2
            grid_df.to_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_2.pkl')
            print('Size:', grid_df.shape)
        else:
            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_2.pkl')

        # We don't need prices_df anymore
        del prices_df
        
        if not os.path.exists(f'./datasets/{self.dataset_name}/processed/grid_part_3.pkl'):

            ### Merge 9 of 14 of the columns of the *calender_df* with the *grid_df*
            grid_df = grid_df[MAIN_INDEX]

            # Merge calendar partly
            icols = ['date',
                    'd',
                    'event_name_1',
                    'event_type_1',
                    'event_name_2',
                    'event_type_2',
                    'snap_CA',
                    'snap_TX',
                    'snap_WI']

            grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')        


            ### 7 columns with categorical values are converted to boolean and numerical values. The *date* column is converted to the datetime format.
            # Minify data
            # 'snap_' columns we can convert to bool or int8
            icols = ['event_name_1',
                    'event_type_1',
                    'event_name_2',
                    'event_type_2',
                    'snap_CA',
                    'snap_TX',
                    'snap_WI']

            for col in icols:
                grid_df[col] = grid_df[col].astype('category')

            # Convert to DateTime
            grid_df['date'] = pd.to_datetime(grid_df['date'])        

            ### Create datetime features
            # Make some features from date
            grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
            grid_df['tm_w'] = grid_df['date'].dt.isocalendar().week.astype(np.int8)
            grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
            grid_df['tm_y'] = grid_df['date'].dt.year
            grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
            grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8) # 오늘 몇째주?

            grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8) 
            grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)

            # Remove date
            del grid_df['date']

            ########################### Save part 3 (Dates)
            #################################################################################
            print('Save part 3')

            # Safe part 3
            grid_df.to_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_3.pkl')
        else:
            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_3.pkl')
        
        
        print('Size:', grid_df.shape)

        # We don't need calendar_df anymore
        del calendar_df
        del grid_df        
        
        SHIFT_DAY = 28
        if not os.path.exists(f'./datasets/{self.dataset_name}/processed/lags_df_'+str(SHIFT_DAY)+'.pkl') or self.toy_example:

            ### converts the days columns from string to int, e.g. d_1 -> 1 - Some additional cleaning
            ## Part 1
            # Convert 'd' to int
            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_1.pkl')
            grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)        

            ### Create lag features
            # Remove 'wm_yr_wk'
            # as test values are not in train set
            del grid_df['wm_yr_wk']
            grid_df.to_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_1.pkl')

            del grid_df

            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_1.pkl')

            # We need only 'id','d','sales'
            # to make lags and rollings
            grid_df = grid_df[['id','d','sales']]

            # Lags
            # with 28 day shift
            print('Create lags')

            LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+15)]
            grid_df = grid_df.assign(**{
                    '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
                    for l in LAG_DAYS
                    for col in [TARGET]
                })

            # Minify lag columns
            for col in list(grid_df):
                if 'lag' in col:
                    grid_df[col] = grid_df[col].astype(np.float16)


            # Rollings
            # with 28 day shift
            print('Create rolling aggs')

            for i in [7,14,30,60,180]:
                print('Rolling period:', i)
                grid_df['rolling_mean_'+str(i)] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
                grid_df['rolling_std_'+str(i)]  = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)

            # Rollings
            # with sliding shift
            for d_shift in [1,7,14]: 
                print('Shifting period:', d_shift)
                for d_window in [7,14,30,60]:
                    col_name = 'rolling_mean_tmp_'+str(d_shift)+'_'+str(d_window)
                    grid_df[col_name] = grid_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)



            ########################### Export
            #################################################################################
            print('Save lags and rollings')
            grid_df.to_pickle(f'./datasets/{self.dataset_name}/processed/lags_df_'+str(SHIFT_DAY)+'.pkl')
        else:
            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/lags_df_'+str(SHIFT_DAY)+'.pkl')
    
            
        if not os.path.exists(f'./datasets/{self.dataset_name}/processed/mean_encoding_df.pkl'):
            
            ########################### Apply on grid_df
            #################################################################################
            # lets read grid from 
            # https://www.kaggle.com/kyakovlev/m5-simple-fe
            # to be sure that our grids are aligned by index
            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/grid_part_1.pkl')
            grid_df['sales'][grid_df['d']>(1941-28)] = np.nan
            base_cols = list(grid_df)

            icols =  [
                        ['state_id'],
                        ['store_id'],
                        ['cat_id'],
                        ['dept_id'],
                        ['state_id', 'cat_id'],
                        ['state_id', 'dept_id'],
                        ['store_id', 'cat_id'],
                        ['store_id', 'dept_id'],
                        ['item_id'],
                        ['item_id', 'state_id'],
                        ['item_id', 'store_id']
                        ]

            for col in icols:
                print('Encoding', col)
                col_name = '_'+'_'.join(col)+'_'
                grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)['sales'].transform('mean').astype(np.float16)
                grid_df['enc'+col_name+'std'] = grid_df.groupby(col)['sales'].transform('std').astype(np.float16)

            keep_cols = [col for col in list(grid_df) if col not in base_cols]
            grid_df = grid_df[['id','d']+keep_cols]    
            
            print('Save Mean/Std encoding')
            grid_df.to_pickle(f'./datasets/{self.dataset_name}/processed/mean_encoding_df.pkl')
        else:
            grid_df = pd.read_pickle(f'./datasets/{self.dataset_name}/processed/mean_encoding_df.pkl')            

        
        ### Reload the data
        FIRST_DAY = 710
        remove_feature = ['id',
                          'state_id',
                          'store_id',
        #                   'item_id',
        #                   'dept_id',
        #                   'cat_id',
                          'date','wm_yr_wk','d','sales']

        cat_var = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
        cat_var = list(set(cat_var) - set(remove_feature))

        grid2_colnm = ['sell_price', 'price_max', 'price_min', 'price_std',
                       'price_mean', 'price_norm', 'price_nunique', 'item_nunique',
                       'price_momentum', 'price_momentum_m', 'price_momentum_y']

        grid3_colnm = ['event_name_1', 'event_type_1', 'event_name_2',
                       'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'tm_d', 'tm_w', 'tm_m',
                       'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end']

        lag_colnm = [ 'sales_lag_28', 'sales_lag_29', 'sales_lag_30',
                     'sales_lag_31', 'sales_lag_32', 'sales_lag_33', 'sales_lag_34',
                     'sales_lag_35', 'sales_lag_36', 'sales_lag_37', 'sales_lag_38',
                     'sales_lag_39', 'sales_lag_40', 'sales_lag_41', 'sales_lag_42',

                     'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
                     'rolling_mean_30', 'rolling_std_30', 'rolling_mean_60',
                     'rolling_std_60', 'rolling_mean_180', 'rolling_std_180']

        mean_enc_colnm = [

            'enc_store_id_dept_id_mean', 'enc_store_id_dept_id_std', 
            'enc_item_id_state_id_mean', 'enc_item_id_state_id_std',


        ]

        grid_1 = pd.read_pickle(f"./datasets/{self.dataset_name}/processed/grid_part_1.pkl")
        grid_2 = pd.read_pickle(f"./datasets/{self.dataset_name}/processed/grid_part_2.pkl")[grid2_colnm]
        grid_3 = pd.read_pickle(f"./datasets/{self.dataset_name}/processed/grid_part_3.pkl")[grid3_colnm]

        grid_df = pd.concat([grid_1, grid_2, grid_3], axis=1)
        del grid_1, grid_2, grid_3; gc.collect()

        grid_df = grid_df[grid_df['d'] >= FIRST_DAY]

        lag = pd.read_pickle(f"./datasets/{self.dataset_name}/processed/lags_df_28.pkl")[lag_colnm]

        lag = lag[lag.index.isin(grid_df.index)]

        grid_df = pd.concat([grid_df,
                         lag],
                        axis=1)

        del lag; gc.collect()


        mean_enc = pd.read_pickle(f"./datasets/{self.dataset_name}/processed/mean_encoding_df.pkl")[mean_enc_colnm]
        mean_enc = mean_enc[mean_enc.index.isin(grid_df.index)]

        grid_df = pd.concat([grid_df,
                             mean_enc],
                            axis=1)    
        del mean_enc; gc.collect()

        grid_df = reduce_mem_usage(grid_df)        
        
        ### Obtain data in our common format
        # Store days to retrieve them later
        X_test = grid_df.loc[grid_df["sales"].isna()]
        self.test_d = X_test.d
        X_test.drop(["sales", "id",'d'],axis=1,inplace=True)
        
        X_train = grid_df.loc[~grid_df["sales"].isna()]
        y_train = X_train["sales"]
        self.train_d = X_train.d
        X_train.drop(["sales","id",'d'],axis=1,inplace=True)
        
        cat_var = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
        
        self.cat_indices = [num for num,col in enumerate(X_train.columns) if col in cat_var]            
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, **kwargs):
        '''
        Solution implemented based on kernel of th first place solution.
        
        1. A lot of preprocessing done in loading function - hard to disentangle. Hence, it is not possible to evaluate standardized preprocessing for this dataset.
        
        
        
        '''
        print("Expert preprocessing not implemented yet")

    def get_cv_folds(self, X_train, y_train, seed=42):
        ### Time based CV - last split was public leaderboard
        FIRST_DAY = 710
        validation = {
            'cv1' : [1551, 1610],
            'cv2' : [1829,1857],
            'cv3' : [1857, 1885],
            'cv4' : [1885,1913],
            'cv5' : [1913, 1941]}    

        folds = []
        for cv in validation:
            folds.append([X_train.index[(self.train_d <= validation[cv][0]) & (self.train_d >= FIRST_DAY)], 
                          X_train.index[(self.train_d > validation[cv][0]) & (self.train_d <= validation[cv][1])]
                         ])    
        return folds
        
 
        
    

    
    
################################################################
################################################################
################################################################

class SantanderTransactionDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "santander-customer-transaction-prediction"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []            
        self.y_col = "target"
        self.large_dataset = False
        
    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', index_col=0, engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', index_col=0, engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        y_train = data[self.y_col]
        X_train = data.drop(self.y_col,axis=1)    
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented (https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
        1. Feature Engineering: 200 categorical Features (one per raw feature) based on counting (unique) values resulting in five categories per feature:
        - This value appears at least another time in data with target==1 and no 0;
        - This value appears at least another time in data with target==0 and no 1;
        - This value appears at least two more time in data with target==0 & 1;
        - This value is unique in data;
        - This value is unique in data + test (only including real test samples);
        2. Feature Engineering: 200 numerical features with replacing values of raw features that are unique in data + test with the mean of the feature
        The other 200 (one per raw feature) features are numerical, let's call them "not u
        3. Apply StandardScaler to numerical features
        4. Shuffle augmentation: duplicate and shuffle 16 times samples with target == 1, 4 for target ==0
        5. Pseudo label: 2700 highest predicted test points as 1 and 2000 lowest as 0
        6. 10 fold Stratified cross validation with multiple seeds for final blend
        7. LGBM
        8. NN: Custom architecture
        - embed all the features belonging to the same group(raw / has one / not unique) independently and in the same way (i.e using same set of weights)
        - weighted average of those 200 embeddings which we then feed to a dense layer for final output. This ensure that every feature is treated in the same way. Weightes were generated by another NN. Idea similar to attention networks 
        - all end-to-end
        - added on the fly augmentation (for every batch, shuffle the features values that belong to target == 1 / target == 0)
        - Adding pseudo label (5000 highest and 3000 lowest) 
        9. Ensembling: 2.1NN, 1LGBM
        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            data = pd.concat([y_train,X_train],axis=1)
    
            not_used = []
            cat_feat = []
            target = 'target'
            features = [i for i in X_train.columns]
            orig = [f'var_{i}' for i in range(200)]
            has_one = [f'var_{i}_has_one' for i in range(200)]
            has_zero = [f'var_{i}_has_zero' for i in range(200)]
            not_u = [f'var_{i}_not_unique' for i in range(200)]
    
            # Create new features: 1 if value is unique in test data
            for f in orig:
                unique_v = X_test[f].value_counts()
                unique_v = unique_v.index[unique_v == 1]
                new_feat = X_test[f].isin(unique_v)
                new_feat.name = f + '_u'
                X_test = pd.concat([X_test,new_feat],axis=1)
    
            # Create feature: 1 if sample has any unique values in a feature in test data
            X_test['has_unique'] = X_test[[f + '_u' for f in orig]].any(axis=1)
            print(X_test['has_unique'].sum())    
    
            # Those samples are  real samples
            real_samples = X_test.loc[X_test['has_unique'], orig]
            ref = pd.concat([data, real_samples], axis=0)
            print(ref.shape)    
    
            # For each feature in the train data, generate a categorical feature corresponding to counts:
            for f in orig:
                data[f + '_has_one'] = 0
                data[f + '_has_zero'] = 0
                
                f_1 = data.loc[data[target] == 1, f].value_counts()
    
                f_1_1 = set(f_1.index[f_1 > 1])
                f_0_1 = set(f_1.index[f_1 > 0])
    
                f_0 = data.loc[data[target] == 0, f].value_counts()
                f_0_0 = set(f_0.index[f_0 > 1])
                f_1_0 = set(f_0.index[f_0 > 0])
    
                data.loc[data[target] == 1, f + '_has_one'] = data.loc[data[target] == 1, f].isin(f_1_1).astype(int)
                data.loc[data[target] == 0, f + '_has_one'] = data.loc[data[target] == 0, f].isin(f_0_1).astype(int)
    
                data.loc[data[target] == 1, f + '_has_zero'] = data.loc[data[target] == 1, f].isin(f_1_0).astype(int)
                data.loc[data[target] == 0, f + '_has_zero'] = data.loc[data[target] == 0, f].isin(f_0_0).astype(int)

                data = data.copy()
            
            data.loc[:, has_one] = 2*data.loc[:, has_one].values + data.loc[:, has_zero].values    
    
            # ???Apply the same to the fake test data
            for f in orig:
                X_test[f + '_has_one'] = 0
                X_test[f + '_has_zero'] = 0
                f_1 = data.loc[data[target] == 1, f].unique()
                f_0 = data.loc[data[target] == 0, f].unique()
                X_test.loc[:, f + '_has_one'] = X_test[f].isin(f_1).astype(int)
                X_test.loc[:, f + '_has_zero'] = X_test[f].isin(f_0).astype(int)

                X_test = X_test.copy()
    
            X_test.loc[:, has_one] = 2*X_test.loc[:, has_one].values + X_test.loc[:, has_zero].values    
    
            #??? Add a feature that indicates whether the feature value of a sample occurs uniquely (in the train data and among the real test samples) 
            for f in orig:
                if use_test:
                    v = ref[f].value_counts()
                else:
                    v = data[f].value_counts()
    
                non_unique_v = v.index[v != 1]
    
                m_trd = data[f].isin(non_unique_v)
                data[f + '_not_unique'] = m_trd  * data[f] + (~m_trd) * data[f].mean()
    
                m_X_test = X_test[f].isin(non_unique_v)
                X_test[f + '_not_unique'] = m_X_test  * X_test[f] + (~m_X_test) * data[f].mean()
                
                data.loc[~m_trd, f + '_has_one'] = 4
                X_test.loc[~m_X_test, f + '_has_one'] = 4    

                data = data.copy()
                X_test = X_test.copy()

            
            y_train = data["target"]
            X_train = data.drop("target",axis=1)    

            # Get unique value counts (added to expert solution, doesnt help - even worse)
            # for f in orig:
            #     unique = X_train[f].value_counts().to_dict()
            #     X_train[f+"_count"] = X_train[f].map(unique)
            #     X_test[f+"_count"] = X_test[f].map(unique)
            #     X_test[f+"_count"].fillna(0)
    
            ######## Preprocessing B
            has_one = [f'var_{i}_has_one' for i in range(200)]
            orig = [f'var_{i}' for i in range(200)]
            not_u = [f'var_{i}_not_unique' for i in range(200)]
    
            cont_vars = orig + not_u
            cat_vars = has_one
    
            for f in cat_vars:
                X_train[f] = X_train[f].astype('category').cat.as_ordered()
                X_test[f] = X_test[f].astype('category').cat.as_ordered()
                X_test[f] = pd.Categorical(X_test[f], categories=X_train[f].cat.categories, ordered=True)
    
            # constant feature to replace feature index information
            # feat = ['intercept']
            # X_train['intercept'] = 1
            # X_train['intercept'] = X_train['intercept'].astype('category')
            # X_test['intercept'] = 1
            # X_test['intercept'] = X_test['intercept'].astype('category')
    
            # cat_vars += feat
    
            ref = pd.concat([X_train[cont_vars + cat_vars], X_test[cont_vars + cat_vars]])
    
            # ss = StandardScaler()
            # ref[cont_vars] = ss.fit_transform(ref[cont_vars].values)
            # self.x_scaled = True
    
            if self.toy_example:
                X_train = ref.iloc[:1000]
                X_test = ref.iloc[1000:]
            else:
                X_train = ref.iloc[:200000]
                X_test = ref.iloc[200000:]
    
            y_train = y_train.astype('int')    
    
            self.cat_indices = list(np.where(X_train.dtypes=="category")[0])#[np.where(X_train.columns==i)[0][0] for i in cat_vars]
    
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    def get_cv_folds(self, X_train, y_train, seed=42):
        ss = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        folds = []
        for num, (train,test) in enumerate(ss.split(X_train.copy(), y_train.copy())):
            folds.append([train, test])  

        return folds

    def openfe_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False):
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_openfe.pickle") or overwrite_existing:
            print("Apply OpenFE preprocessing")
            import warnings
            warnings.filterwarnings("ignore")

            task = "regression" if self.task_type == "regression" else "classification"
    
            cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in self.cat_indices])
            cont_col_names = X_train.iloc[:,cont_indices].columns.values.tolist()
            if len(self.cat_indices)>0:
                cat_col_names = X_train.columns[self.cat_indices]
            else:
                cat_col_names = None
            
            candidate_features_list = openfe.get_candidate_features(numerical_features=cont_col_names, categorical_features=cat_col_names, order=1)
            candidate_features_list = [i for i in candidate_features_list if i.name in ["freq","abs","log", "sqrt","square","sigmoid","round","residual"]]


            ofe = openfe.OpenFE()
            features = ofe.fit(data=X_train, label=y_train, n_jobs=os.cpu_count(), task=task, n_data_blocks=8, 
                               candidate_features_list=candidate_features_list,
                               stage2_params={"verbose": -1},
                               verbose=True, tmp_save_path=f'./openfe_tmp_data_{self.dataset_name}.feather')
                   

            X_train_new, X_test_new = openfe.transform(X_train, X_test, features, n_jobs=os.cpu_count())

            is_combined = [f.name=='Combine' for f in features]
            if sum(is_combined)>0:
                self.cat_indices += list(np.where([f.name=='Combine' for f in features])[0]+X_train.shape[1])

            self.X_train, self.X_test = X_train_new, X_test_new
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(self.X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'wb'))            
            pickle.dump(self.X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'wb'))


        else:
            print(f"Load existing openFE-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'rb'))
            self.X_train, self.X_test = X_train, X_test

        self.preprocess_states.append("openfe")
            
################################################################
################################################################
################################################################

class IEEEFraudDetectionDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        
        self.dataset_name = "ieee-fraud-detection"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []            
        self.y_col = "isFraud"        
        self.large_dataset = False

        # load precomputed month object for cross validation
        self.DT_M = pd.read_csv(f"./datasets/{self.dataset_name}/DT_M.csv",index_col=0)["DT_M"]
    
    def load_data(self):
        
        dtypes = {}

        # LOAD TRAIN
        X_train = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train_transaction.csv', index_col='TransactionID', engine="pyarrow")
        train_id = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train_identity.csv', index_col='TransactionID', engine="pyarrow")
        X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)
        
        # LOAD TEST
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test_transaction.csv', index_col='TransactionID', engine="pyarrow")
        test_id = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test_identity.csv', index_col='TransactionID', engine="pyarrow")
        fix = {o:n for o, n in zip(test_id.columns, train_id.columns)}
        test_id.rename(columns=fix, inplace=True)
        X_test = X_test.merge(test_id, how='left', left_index=True, right_index=True)

        if self.toy_example:
            # Need samples from each month
            X_train = pd.concat([
                X_train.iloc[100000:100200],
                X_train.iloc[200000:200200],
                X_train.iloc[300000:300200],
                X_train.iloc[400000:400200],
                X_train.iloc[500000:500200],
                X_train.iloc[550000:550200],
                                ])
            self.DT_M = pd.concat([
                self.DT_M.iloc[100000:100200],
                self.DT_M.iloc[200000:200200],
                self.DT_M.iloc[300000:300200],
                self.DT_M.iloc[400000:400200],
                self.DT_M.iloc[500000:500200],
                self.DT_M.iloc[550000:550200],
                                ])
            
            X_test = X_test.iloc[:1000]
        
        # TARGET
        y_train = X_train['isFraud'].copy()
        del train_id, test_id, X_train['isFraud']; x = gc.collect()
        # PRINT STATUS
        # print('Train shape',X_train.shape,'test shape',X_test.shape)   
        
        # Obtain correct cat indices
        cat_cols = ['ProductCD',"addr1","addr2","P_emaildomain","R_emaildomain","DeviceType","DeviceInfo"]
        cat_cols += [f"card{i}" for i in range(1,7)]
        cat_cols += [f"M{i}" for i in range(1,10)]
        cat_cols += [f"id_{i}" for i in range(12,39)]
        self.cat_indices = [np.where(X_train.columns==i)[0][0] for i in cat_cols]
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, cat_method=None, **kwargs):
        '''
        Summary of the solution implemented (First place solution: https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600/notebook)

        '''
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"

        if cat_method is not None:
            dataset_version = dataset_version+"_"+cat_method
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            ### Define utility functions
            # FREQUENCY ENCODE TOGETHER
            def encode_FE(df1, df2, cols, use_test=True):
                for col in cols:
                    if use_test:
                        df = pd.concat([df1[col], df2[col]])
                        vc = df.value_counts(dropna=True, normalize=True).to_dict()
                    else:
                        vc = df1.value_counts(dropna=True, normalize=True).to_dict()                        
                    vc[-1] = -1
                    nm = col + '_FE'
                    df1[nm] = df1[col].map(vc)
                    df1[nm] = df1[nm].astype('float32')
                    df2[nm] = df2[col].map(vc)
                    df2[nm] = df2[nm].astype('float32')
                    # print(nm, ', ', end='')
    
    
            # LABEL ENCODE
            # def encode_LE(col, train, test, verbose=True):
            #     df_comb = pd.concat([train[col], test[col]], axis=0)
            #     df_comb, _ = df_comb.factorize(sort=True)
            #     nm = col
            #     if df_comb.max() > 32000:
            #         train[nm] = df_comb[:len(train)].astype('int32')
            #         test[nm] = df_comb[len(train):].astype('int32')
            #     else:
            #         train[nm] = df_comb[:len(train)].astype('int16')
            #         test[nm] = df_comb[len(train):].astype('int16')
            #     if verbose: print(nm, ', ', end='')
    
    
            # GROUP AGGREGATION MEAN AND STD
            # https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
            def encode_AG(main_columns, uids, aggregations, train_df, test_df,
                          fillna=True, usena=False, use_test=True):
                # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
                for main_column in main_columns:
                    for col in uids:
                        for agg_type in aggregations:
                            new_col_name = main_column + '_' + col + '_' + agg_type
                            if use_test:
                                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                            else:
                                temp_df = train_df[[col, main_column]]
                            
                            if usena: temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                            temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                columns={agg_type: new_col_name})
    
                            temp_df.index = list(temp_df[col])
                            temp_df = temp_df[new_col_name].to_dict()
    
                            train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                            test_df[new_col_name] = test_df[col].map(temp_df).astype('float32')
    
                            if fillna:
                                train_df[new_col_name] = train_df[new_col_name].fillna(-1)
                                train_df[new_col_name] = test_df[new_col_name].fillna(-1)

                            # print("'" + new_col_name + "'", ', ', end='')
    
    
            # COMBINE FEATURES
            def encode_CB(col1, col2, df1, df2):
                nm = col1 + '_' + col2
                df1[nm] = df1[col1].astype(str) + '_' + df1[col2].astype(str)
                df2[nm] = df2[col1].astype(str) + '_' + df2[col2].astype(str)
                df1 = df1.copy()
                df2 = df2.copy()
            
            #     print(nm, ', ', end='')
    
    
            # GROUP AGGREGATION NUNIQUE
            def encode_AG2(main_columns, uids, train_df, test_df, use_test=True):
                for main_column in main_columns:
                    for col in uids:
                        if use_test:
                            comb = pd.concat([train_df[[col] + [main_column]], test_df[[col] + [main_column]]], axis=0)
                        else:
                            comb = train_df[[col] + [main_column]]
                        mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
                        train_df[col + '_' + main_column + '_ct'] = train_df[col].map(mp).astype('float32')
                        test_df[col + '_' + main_column + '_ct'] = test_df[col].map(mp).astype('float32')
                        # print(col + '_' + main_column + '_ct, ', end='')
                        
            ### Apply first preprocessing
            # The codes for data preparation is from https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600
            # COLUMNS WITH STRINGS
            str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
                        'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
                        'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
        #     str_type += ['id-12', 'id-15', 'id-16', 'id-23', 'id-27', 'id-28', 'id-29', 'id-30',
        #                 'id-31', 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38']
    
            # FIRST 53 COLUMNS
            cols = ['TransactionDT', 'TransactionAmt',
                   'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                   'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                   'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                   'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
                   'M5', 'M6', 'M7', 'M8', 'M9']
    
            # COLUMNS FROM ID TABLE
            id_cols = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',
                   'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
                   'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
                   'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
                   'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
                   'DeviceInfo']
            # V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
            # https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
            v =  [1, 3, 4, 6, 8, 11]
            v += [13, 14, 17, 20, 23, 26, 27, 30]
            v += [36, 37, 40, 41, 44, 47, 48]
            v += [54, 56, 59, 62, 65, 67, 68, 70]
            v += [76, 78, 80, 82, 86, 88, 89, 91]
    
            # v += [96, 98, 99, 104] # relates to groups, no NAN
            v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN
            v += [124, 127, 129, 130, 136] # relates to groups, no NAN
    
            # LOTS OF NAN BELOW
            v += [138, 139, 142, 147, 156, 162] #b1
            v += [165, 160, 166] #b1
            v += [178, 176, 173, 182] #b2
            v += [187, 203, 205, 207, 215] #b2
            v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2
            v += [218, 223, 224, 226, 228, 229, 235] #b3
            v += [240, 258, 257, 253, 252, 260, 261] #b3
            v += [264, 266, 267, 274, 277] #b3
            v += [220, 221, 234, 238, 250, 271] #b3
    
            v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN
            v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN
            v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN
            #v += [332, 325, 335, 338] # b4 lots NAN
    
            cols += ['V'+str(x) for x in sorted(v)]
            dtypes = {}
            for c in cols+['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,34)]:
                    dtypes[c] = 'float32'
            for c in str_type: dtypes[c] = 'category'
    
            cols += id_cols
    
            X_train = X_train[cols]
            X_test = X_test[cols]
    
            X_train = X_train.astype(dtypes)
            X_test = X_test.astype(dtypes)
    
            # NORMALIZE D COLUMNS    
            for i in range(1,16):
                if i in [1,2,3,5,9]: continue
                X_train['D'+str(i)] =  X_train['D'+str(i)] - X_train.TransactionDT/np.float32(24*60*60)
                X_test['D'+str(i)] = X_test['D'+str(i)] - X_test.TransactionDT/np.float32(24*60*60)
    
            for i,f in enumerate(X_train.columns):
                # FACTORIZE CATEGORICAL VARIABLES
                if (str(X_train[f].dtype)=='category')|(X_train[f].dtype=='object'):
                    if use_test:
                        df_comb = pd.concat([X_train[f],X_test[f]],axis=0)
                        df_comb,_ = df_comb.factorize(sort=True)
                        # if df_comb.max()>32000: print(f,'needs int32')
                        X_train[f] = df_comb[:len(X_train)].astype('int16')
                        X_test[f] = df_comb[len(X_train):].astype('int16')
                    else:
                        comb,ind = X_train[f].astype(str).factorize(sort=True)
                        # if comb.max()>32000: print(f,'needs int32')
                        X_train[f] = comb.astype('int16')
                        map_dict = {i: num for num, i in enumerate(ind)}
                        X_test[f] = X_test[f].astype(str).map(map_dict)
                        if X_test[f].isna().sum()>0:
                            X_test[f] = X_test[f].fillna(-1.).astype('int16')
                X_train = X_train.copy()
                X_test = X_test.copy()
            
            # COMBINE COLUMNS CARD1+ADDR1
            encode_CB('card1', 'addr1', df1=X_train, df2=X_test)
            X_train = X_train.copy()
            X_test = X_test.copy()
            
            START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
            X_train['DT_M'] = X_train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
            X_train['DT_M'] = (X_train['DT_M'].dt.year - 2017) * 12 + X_train['DT_M'].dt.month
    
            X_test['DT_M'] = X_test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
            X_test['DT_M'] = (X_test['DT_M'].dt.year - 2017) * 12 + X_test['DT_M'].dt.month
    
            X_train['day'] = X_train.TransactionDT / (24 * 60 * 60)
            X_train['uid'] = X_train.card1_addr1.astype(str) + '_' + np.floor(X_train.day - X_train.D1).astype(str)
    
            X_test['day'] = X_test.TransactionDT / (24 * 60 * 60)
            X_test['uid'] = X_test.card1_addr1.astype(str) + '_' + np.floor(X_test.day - X_test.D1).astype(str)
            # encode_LE('uid', train=X_train, test=X_test)            
    
            ### Apply feature engineering
            # ExpertFE is from https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600

            # TRANSACTION AMT CENTS
            X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
            X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
            # print('cents, ', end='')
            # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
            encode_FE(X_train, X_test, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
            encode_CB('card1_addr1', 'P_emaildomain', X_train, X_test)
            X_train = X_train.copy()
            X_test = X_test.copy()
            
            # FREQUENCY ENCODE
            encode_FE(X_train, X_test, ['card1_addr1', 'card1_addr1_P_emaildomain'])
            # GROUP AGGREGATE
            encode_AG(['TransactionAmt', 'D9', 'D11'], ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'],
                      train_df=X_train, test_df=X_test, fillna=False, usena=True)
            
            # FREQUENCY ENCODE UID
            encode_FE(X_train, X_test, ['uid'])
            # AGGREGATE
            encode_AG(['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['uid'], ['mean', 'std'],
                      train_df=X_train, test_df=X_test, usena=True) # might use fillna=True
            # AGGREGATE
            encode_AG(['C' + str(x) for x in range(1, 15) if x != 3], ['uid'], ['mean'],
                      train_df=X_train, test_df=X_test, fillna=False, usena=True)
            # AGGREGATE
            encode_AG(['M' + str(x) for x in range(1, 10)], ['uid'], ['mean'],
                      train_df=X_train, test_df=X_test, fillna=False, usena=True)
            
            # AGGREGATE
            encode_AG2(['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents'], ['uid'],
                       train_df=X_train, test_df=X_test)
            # AGGREGATE
            encode_AG(['C14'], ['uid'], ['std'],
                      train_df=X_train, test_df=X_test, fillna=False, usena=True)
            # AGGREGATE
            encode_AG2(['C13', 'V314'], ['uid'], train_df=X_train, test_df=X_test)
            # AGGREATE
            encode_AG2(['V127', 'V136', 'V309', 'V307', 'V320'], ['uid'], train_df=X_train, test_df=X_test)
            # NEW FEATURE
            X_train['outsider15'] = (np.abs(X_train.D1 - X_train.D15) > 3).astype('int8')
            X_test['outsider15'] = (np.abs(X_test.D1 - X_test.D15) > 3).astype('int8')            

            X_train = X_train.copy()
            X_test = X_test.copy()
            
            ### Apply additional feature engineering
            # Apparently, an important preprocessing step is to treat categorical data as numeric.
            # An explanation why this can work for that is given here: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/104134 and https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/104796
            for i, f in enumerate(X_train.columns):
                if (str(X_train[f].dtype)=='category')|(X_train[f].dtype=='object'):
                    if use_test:
                        df_comb = pd.concat([X_train[f], X_test[f]], axis=0)
                        df_comb, _ = df_comb.factorize(sort=True)
                        # if df_comb.max() > 32000: print(f, 'needs int32')
                        X_train[f] = df_comb[:len(X_train)].astype('int16')
                        X_test[f] = df_comb[len(X_train):].astype('int16')
                    else:
                        comb,ind = X_train[f].factorize(sort=True)
                        # if comb.max()>32000: print(f,'needs int32')
                        X_train[f] = comb.astype('int16')
                        map_dict = {i: num for num, i in enumerate(ind)}
                        X_test[f] = X_test[f].astype(str).map(map_dict)
                        if X_test[f].isna().sum()>0:
                            X_test[f] = X_test[f].fillna(-1.).astype('int16')

                # SHIFT ALL NUMERICS POSITIVE. SET NAN to -1
                elif f not in ['TransactionAmt', 'TransactionDT']:
                    if use_test:
                        mn = np.min((X_train[f].min(), X_test[f].min()))
                    else:
                        mn = X_train[f].min()
                    X_train[f] -= np.float32(mn)
                    X_test[f] -= np.float32(mn)
                    # X_train[f].fillna(-1, inplace=True)
                    # X_test[f].fillna(-1, inplace=True)    

            cols = list(X_train.columns)
            # FAILED TIME CONSISTENCY TEST
            for c in ['C3', 'M5', 'id_08', 'id_33']:
                cols.remove(c)
            for c in ['card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']:
                cols.remove(c)
            for c in ['id_' + str(x) for x in range(22, 28)]:
                cols.remove(c)
    
            cols.remove('TransactionDT')
            for c in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
                cols.remove(c)
            for c in ['DT_M', 'day', 'uid']:
                cols.remove(c)            
        
            X_train = X_train[cols]
            X_test = X_test[cols]

            # Added because 'TransactionAmt_uid_mean' has only NAs - original solution imputed them
            # na_only = X_train.columns[X_train.isna().sum()==X_train.shape[0]]
            # X_train = X_train.drop(na_only,axis=1)
            # X_test = X_test.drop(na_only,axis=1) 

            if cat_method == "model":
                cat_features = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'card1', 'card2', 'card3', 'card5', 'M1', 'M4', 'id_13', 'id_15', 'id_17', 'id_18', 'id_19', 'id_20', 'id_31']
                self.cat_indices = [np.where(X_train.columns==i)[0][0] for i in cat_features if i in X_train.columns]
                # Encode binary cat features as numeric
                for col in X_train.columns[X_train.nunique()==2]:
                    if X_train[col].dtype in [str, "O", "category", "object"]:
                        le = LabelEncoder()
                        mode = X_train[col].mode()[0]
                        X_train[col] = le.fit_transform(X_train[col])
        
                        if len(X_test[col].unique())==2:
                            X_test[col] = le.transform(X_test[col])
                        else:
                            X_test[col] = X_test[col].fillna(mode)
                            X_test[col] = le.transform(X_test[col])
                        
                
                # Define categorical feature types
                self.cat_indices += list(np.where(X_train.dtypes=="O")[0]) 
                self.cat_indices += list(np.where(X_train.dtypes=="object")[0]) 
                self.cat_indices += list(np.where(X_train.dtypes=="category")[0]) 
                self.cat_indices = np.unique(self.cat_indices).tolist()
                
                for num, col in list(zip(self.cat_indices,X_train.columns[self.cat_indices])):
                    # Encode binary categorical features
                    if X_train[col].nunique()==2:
                        value_1 = X_train[col].dropna().unique()[0]
                        X_train[col] = (X_train[col]==value_1).astype(float)
                        X_test[col] = (X_test[col]==value_1).astype(float)
                        self.cat_indices.remove(num)
                    else:
                        # Note: The category dtype needs to entail all train categories when applying .astype("category") on test data
                        dtype = pd.CategoricalDtype(categories=list(X_train[col].astype(str).fillna("nan").unique()))
                        X_train[col] = X_train[col].astype(str).fillna("nan").astype(dtype)
                        X_test[col] = X_test[col].astype(str).fillna("nan").astype(dtype)       
                
            else:
                self.cat_indices = []
            # Using categorical features as numeric was a very relevant step to reduce overfitting!
       #      cat_features = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
       # 'DeviceInfo', 'card1', 'card2', 'card3', 'card5', 'M1', 'M4',
       # 'id_13', 'id_15', 'id_17', 'id_18', 'id_19', 'id_20', 'id_31']
       #      cat_features = ['ProductCD', 'P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'M1', 'M4',
       # 'id_15', 'id_31']
       #      self.cat_indices = [np.where(X_train.columns==i)[0][0] for i in cat_features if i in X_train.columns]
       #      for col in X_train.columns[self.cat_indices]:
       #          dtype = pd.CategoricalDtype(categories=list(X_train[col].astype(str).fillna("nan").unique()))
       #          X_train[col] = X_train[col].astype(str).fillna("nan").astype(dtype)
       #          X_test[col] = X_test[col].astype(str).fillna("nan").astype(dtype)       
                    
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices.pickle', 'wb'))
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            try:
                self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
            except:
                self.cat_indices = []            # self.cat_indices = []
            # cat_features = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'card1', 'card2', 'card3', 'card5', 'M1', 'M4', 'id_13', 'id_15', 'id_17', 'id_18', 'id_19', 'id_20', 'id_31']
            # self.cat_indices =  [np.where(X_train.columns==i)[0][0] for i in cat_features if i in X_train.columns]
        
        self.preprocess_states.append("expert")
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

  
    
    
    def expert_postprocessing(self, X_train, y, test=True, **kwargs):
        if test:
            X_train = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train_transaction.csv', usecols=['TransactionID',"isFraud"], index_col='TransactionID', engine="pyarrow")
            X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test_transaction.csv', usecols=['TransactionID'], index_col='TransactionID', engine="pyarrow")
            if self.toy_example:
                # Need samples from each month
                X_train = pd.concat([
                    X_train.iloc[100000:100200],
                    X_train.iloc[200000:200200],
                    X_train.iloc[300000:300200],
                    X_train.iloc[400000:400200],
                    X_train.iloc[500000:500200],
                    X_train.iloc[550000:550200],
                                    ])
                X_test = X_test.iloc[:1000]
        
            X_test['isFraud'] = y
            
            comb = pd.concat([X_train[['isFraud']],X_test[['isFraud']]],axis=0)
            
            uids = pd.read_csv(f'./datasets/{self.dataset_name}/uids_v4_no_multiuid_cleaning.csv',usecols=['TransactionID','uid'],engine="pyarrow").rename({'uid':'uid2'},axis=1)
            comb = comb.merge(uids,on='TransactionID',how='left')
            mp = comb.groupby('uid2').isFraud.agg(['mean'])
            comb.loc[comb.uid2>0,'isFraud'] = comb.loc[comb.uid2>0].uid2.map(mp['mean'])
            
            uids = pd.read_csv(f'./datasets/{self.dataset_name}/uids_v1_no_multiuid_cleaning.csv',usecols=['TransactionID','uid'],engine="pyarrow").rename({'uid':'uid3'},axis=1)
            comb = comb.merge(uids,on='TransactionID',how='left')
            mp = comb.groupby('uid3').isFraud.agg(['mean'])
            comb.loc[comb.uid3>0,'isFraud'] = comb.loc[comb.uid3>0].uid3.map(mp['mean'])
            
            y_new = comb.iloc[len(X_train):].isFraud.values
        
        
            return y_new
        else:
            return y
    
    def get_cv_folds(self, X_train, y_train, seed=42):
        # Note from experts:
        # We will predict test.csv using GroupKFold with months as groups. 
        # The training data are the months December 2017, January 2018, February 2018, March 2018, April 2018, and May 2018. 
        # We refer to these months as 12, 13, 14, 15, 16, 17. 
        # Fold one in GroupKFold will train on months 13 thru 17 and predict month 12. 
        # Note that the only purpose of month 12 is to tell XGB when to early_stop we don't actual care about the backwards time predictions. 
        # The model trained on months 13 thru 17 will also predict test.csv which is forward in time.        
        
        skf = GroupKFold(n_splits=6)
        folds = []
        for num, (train,test) in enumerate(skf.split(X_train.copy(), y_train.copy(), groups=self.DT_M)):
            folds.append([train, test])    
            
        return folds            

    def openfe_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False):
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_openfe.pickle") or overwrite_existing:
            print("Apply OpenFE preprocessing")
            import warnings
            warnings.filterwarnings("ignore")

            task = "regression" if self.task_type == "regression" else "classification"
    
            # cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in self.cat_indices])
            # cont_col_names = X_train.iloc[:,cont_indices].columns.values.tolist()
            # if len(self.cat_indices)>0:
            #     cat_col_names = X_train.columns[self.cat_indices]
            # else:
            #     cat_col_names = None

            #######<<< From openFE
            idxT = X_train.index[:3 * len(X_train) // 4]
            idxV = X_train.index[3 * len(X_train) // 4:]
    
            categorical_features = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
                                    'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',
                                    'id_30',
                                    'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo',
                                    # 'uid', # Can't use as this is considered part of expert FE engineered in another way 
                                    'card1_addr1']
            to_remove = ['C3', 'M5', 'id_08', 'id_33', 'card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34',
                         'TransactionDT', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'DT_M', 'day', 'uid']
            to_remove.extend(['id_' + str(x) for x in range(22, 28)])
            
            params = {"n_estimators": 1000, "importance_type": "gain", "num_leaves": 64,
                      "seed": 1, "n_jobs": os.cpu_count()}
            
            ordinal_features = []
            for f in X_train.columns:
                if f not in categorical_features: ordinal_features.append(f)
        
            candidate_features_list = openfe.get_candidate_features(numerical_features=[],
                                                        categorical_features=categorical_features,
                                                             ordinal_features=ordinal_features)
            
            ofe = openfe.OpenFE()
            features = ofe.fit(data=X_train.loc[label.index], label=label,
                               init_scores=oof_proba,
                               candidate_features_list=candidate_features_list,
                               metric='rmse',
                               train_index=train_index, val_index=val_index,
                               categorical_features=categorical_features,
                               min_candidate_features=30000,
                               stage2_params=params,
                               drop_columns=to_remove,
                               n_jobs=n_jobs, n_data_blocks=8, task='regression')
            new_features_list = [feature for feature in features[:600]]
            X_train_new, X_test_new = openfe.transform(X_train, X_test, new_features_list, n_jobs=os.cpu_count())

            #######<<< From openFE
            
            # candidate_features_list = openfe.get_candidate_features(numerical_features=cont_col_names, categorical_features=cat_col_names, order=1)

            # ofe = openfe.OpenFE()
            # features = ofe.fit(data=X_train, label=y_train, n_jobs=os.cpu_count(), task=task, n_data_blocks=8, 
            #                    candidate_features_list=candidate_features_list,
            #                    stage2_params={"verbose": -1},
            #                    verbose=True, tmp_save_path=f'./openfe_tmp_data_{self.dataset_name}.feather')
                   

            # X_train_new, X_test_new = openfe.transform(X_train, X_test, features, n_jobs=os.cpu_count())

            is_combined = [f.name=='Combine' for f in features]
            if sum(is_combined)>0:
                self.cat_indices += list(np.where([f.name=='Combine' for f in features])[0]+X_train.shape[1])

            self.X_train, self.X_test = X_train_new, X_test_new
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(self.X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'wb'))            
            pickle.dump(self.X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'wb'))


        else:
            print(f"Load existing openFE-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_openfe.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_openfe.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_openfe.pickle', 'rb'))
            self.X_train, self.X_test = X_train, X_test

        self.preprocess_states.append("openfe")
        

    
################################################################
################################################################
################################################################



class SantanderValueDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "santander-value-prediction-challenge"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "regression"
        self.eval_metric_name = "rmsle"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []            
        self.y_col = "target"
        self.large_dataset = True
        self.heavy_tailed = True

        self.test_leaks = pd.read_csv(f"./datasets/{self.dataset_name}/leak_only_7837.csv", engine="pyarrow")
        
    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        y_train = data[self.y_col]
        X_train = data.drop(self.y_col,axis=1)    

        # Load leak data - was originally produced and stored by the expert preprocessing pipeline and is available as csv as part of our benchmark
        
        ### Obtain final dataset
        self.original_indices = X_train.index
        X_train = pd.concat([X_train, X_test[self.test_leaks.target != 0]], axis = 0)
        X_train.reset_index(drop=True,inplace=True)
        y_train = pd.Series(list(y_train.values) + list(self.test_leaks['target'][self.test_leaks.target != 0]), index=X_train.index, name=self.y_col)

        if self.toy_example:
            X_train = X_train.iloc[:1000]
            y_train = y_train.iloc[:1000]
            X_test = X_test.iloc[:1000]        
            self.test_leaks = self.test_leaks.iloc[:1000]    
            self.original_indices = X_train.index

        # X_train = X_train.drop("ID",axis=1)
        # X_test = X_test.drop("ID",axis=1)
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Solution implemented based on the descriptions in https://www.kaggle.com/competitions/santander-value-prediction-challenge/discussion/63919
        
        1. There was a data leak where the target variable is equal to the value in some columns 
            - Exploit the data leak in the test data
        2. Two things were important:
            - Removing all the ugly/fake rows before leak predictions.
            - Using the leaks for data augmentation.
        3. Used some public kernels to generate features
        4. 10-fold CV and LightGBM as model
        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            raw_data_dir = os.getcwd() + f"/datasets/{self.dataset_name}/raw/"
            processed_data_dir = os.getcwd() + f"/datasets/{self.dataset_name}/processed/"
            
            ### Adapt to data the expert used
            test = X_test
            train = X_train.loc[self.original_indices].copy()
            train["target"] = y_train.loc[self.original_indices]

            ########### 1. Find corresponding feature sets (each set of features is unique) ###########
            ### Exploit the target leak
            #This code is borrowed from a kernel. Not sure
            all_features = [c for c in test.columns if c not in ['ID']]
            def has_ugly(row):
                for v in row.values[row.values > 0]:
                    if str(v)[::-1].find('.') > 2:
                        return True
                return False
            test['has_ugly'] = test[all_features].apply(has_ugly, axis=1)
            test_og = test[['ID']].copy()
            test_og['nonzero_mean'] = test[[c for c in test.columns if c not in ['ID', 'has_ugly']]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
            test = test[test.has_ugly == False]
    
            train_t = train.drop(['target'], axis = 1, inplace=False)
            train_t.set_index('ID', inplace=True)
            train_t = train_t.T
            test_t = test.set_index('ID', inplace=False)
            test_t = test_t.T
    
            gc.collect()
    
            features = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
                    '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
                    'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
                    '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
                    'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
                    '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
                    '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
                    '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']
    
            extra_features = []
            
            #run this iteratively until you have no more links. Then prune
            def chain_pairs(ordered_items):
                ordered_chains = []
                links_found = 0
                for i_1, op_chain in enumerate(ordered_items.copy()[:]):
                    if op_chain[0] != op_chain[1]:
                        end_chain = op_chain[-1]
                        for i_2, op in enumerate(ordered_items.copy()[:]):
                            if (end_chain == op[0]):
                                links_found += 1
                                op_chain.extend(op[1:])
                                end_chain = op_chain[-1]
    
                        ordered_chains.append(op_chain)
                return links_found, ordered_chains
    
            def prune_chain(ordered_chain):
    
                ordered_chain = sorted(ordered_chain, key=len, reverse=True)
                new_chain = []
                id_lookup = {}
                for oc in ordered_chain:
                    id_already_in_chain = False
                    for idd in oc:
                        if idd in id_lookup:
                            id_already_in_chain = True
                        id_lookup[idd] = idd
    
                    if not id_already_in_chain:
                        new_chain.append(oc)
                return sorted(new_chain, key=len, reverse=True)
            
            def find_new_ordered_features(ordered_ids, data_t):
                data = data_t.copy()
    
                f1 = ordered_ids[0][:-1]
                f2 = ordered_ids[0][1:]
                for ef in ordered_ids[1:]:
                    f1 += ef[:-1]
                    f2 += ef[1:]
    
                d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
                d1['ID'] = data.index
                gc.collect()
                d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
                d2['ID'] = data.index
                gc.collect()
                d3 = d2[~d2.duplicated(['key'], keep=False)]
                d4 = d1[~d1.duplicated(['key'], keep=False)]
                d5 = d4.merge(d3, how='inner', on='key')
    
                d_feat = d1.merge(d5, how='left', on='key')
                d_feat.fillna(0, inplace=True)
    
                ordered_features = list(d_feat[['ID_x', 'ID_y']][d_feat.ID_x != 0].apply(list, axis=1))
                del d1,d2,d3,d4,d5,d_feat
                gc.collect()
    
                links_found = 1
                #print(ordered_features)
                while links_found > 0:
                    links_found, ordered_features = chain_pairs(ordered_features)
                    #print(links_found)
    
                ordered_features = prune_chain(ordered_features)
    
                #make lookup of all features found so far
                found = {}
                for ef in extra_features:
                    found[ef[0]] = ef
                    #print (ef[0])
                found [features[0]] = features
    
                #make list of sets of 40 that have not been found yet
                new_feature_sets = []
                for of in ordered_features:
                    if len(of) >= 40:
                        if of[0] not in found:
                            new_feature_sets.append(of)
    
                return new_feature_sets
            
            def add_new_feature_sets(data, data_t):
    
                # print ('\nData Shape:', data.shape)
                f1 = features[:-1]
                f2 = features[1:]
    
                for ef in extra_features:
                    f1 += ef[:-1]
                    f2 += ef[1:]
    
                d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
                d1['ID'] = data['ID']    
                gc.collect()
                d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
                d2['ID'] = data['ID']
                gc.collect()
                #print('here')
                d3 = d2[~d2.duplicated(['key'], keep=False)]
                del d2
                d4 = d1[~d1.duplicated(['key'], keep=False)]
                #print('here')
                d5 = d4.merge(d3, how='inner', on='key')
                del d4
                d = d1.merge(d5, how='left', on='key')
                d.fillna(0, inplace=True)
                #print('here')
                ordered_ids = list(d[['ID_x', 'ID_y']][d.ID_x != 0].apply(list, axis=1))
                del d1,d3,d5,d
                gc.collect()
    
                links_found = 1
                while links_found > 0:
                    links_found, ordered_ids = chain_pairs(ordered_ids)
                    #print(links_found)
    
                # print ('OrderedIds:', len(ordered_ids))
                #Make distinct ordered id chains
                ordered_ids = prune_chain(ordered_ids)
                # print ('OrderedIds Pruned:', len(ordered_ids))
    
                #look for ordered features with new ordered id chains
                new_feature_sets = find_new_ordered_features(ordered_ids, data_t)    
    
                extra_features.extend(new_feature_sets)
                # print('New Feature Count:', len(new_feature_sets))
                # print('Extra Feature Count:', len(extra_features))
            
            add_new_feature_sets(train,train_t)
            add_new_feature_sets(test,test_t)
            add_new_feature_sets(train,train_t)
            add_new_feature_sets(test,test_t)
            add_new_feature_sets(train,train_t)
    
            with open((processed_data_dir + "extra_features_{}.txt".format(len(extra_features))), "w") as text_file:
                for ef in extra_features:
                    text_file.write(','.join(ef) + '\n')
            
            del train_t, test_t, test
            gc.collect()
    
            #now that memory is cleared we can get back full test
            test = pd.read_csv(raw_data_dir + 'test.csv', engine="pyarrow")
            cols = test.columns
            test['has_ugly'] = test[all_features].apply(has_ugly, axis=1)
            test.loc[test.has_ugly == True, cols] = 0.        
            if self.toy_example:
                test = test.iloc[:1000]
        
        ################### 2. Get predictions for test leaks ############# 
        ### Use the copied code to create ???; The expert uses an offset of 39 for the data leakage exploit code.
            def get_log_pred(data, feats, extra_feats, offset = 2):
                f1 = feats[:(offset * -1)]
                f2 = feats[offset:]
                for ef in extra_feats:
                    f1 += ef[:(offset * -1)]
                    f2 += ef[offset:]
    
                d1 = data[f1].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
                d2 = data[f2].apply(tuple, axis=1).apply(hash).to_frame().rename(columns={0: 'key'})
                d2['pred'] = data[feats[offset-2]]
                d2 = d2[d2['pred'] != 0] # Keep?
                d3 = d2[~d2.duplicated(['key'], keep=False)]
                d4 = d1[~d1.duplicated(['key'], keep=False)]
                d5 = d4.merge(d3, how='inner', on='key')
    
                d = d1.merge(d5, how='left', on='key')
                return np.log1p(d.pred).fillna(0)
            
            end_offset = 39
            pred_test = []
            pred_train = []
            efs = extra_features
            for o in list(range(2, end_offset)):
                # print('Offset:', o)
    
                log_pred = get_log_pred(train, features, extra_features, o)
                pred_train.append(np.expm1(log_pred))
                have_data = log_pred != 0
                train_count = have_data.sum()
                score = sqrt(mean_squared_error(np.log1p(train.target[have_data]), log_pred[have_data]))
                # print(f'Score = {score} on {have_data.sum()} out of {train.shape[0]} training samples')
    
    
                log_pred_test = get_log_pred(test, features, efs, o)
                pred_test.append(np.expm1(log_pred_test))
                have_data = log_pred_test != 0
                test_count = have_data.sum()
                # print(f'Have predictions for {have_data.sum()} out of {test.shape[0]} test samples')
    
            pred_train_final = pred_train[0].copy()
            for r in range(1,len(pred_train)):
                pred_train_final[pred_train_final == 0] = pred_train[r][pred_train_final == 0]
    
            train_leak_match_count = (pred_train_final!=0).sum()
            no_match_count = (pred_train_final==0).sum()
            # print ("Train leak count: ", train_leak_match_count, "Train no leak count: ",  no_match_count)
    
            pred_train_temp = pred_train_final.copy()
            nonzero_mean = train[[f for f in train.columns if f not in ["ID", "target","nonzero_mean"]]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
            nonzero_mean.name = "nonzero_mean"
            train = pd.concat([train, nonzero_mean],axis=1)
            
            pred_train_temp[pred_train_temp==0] = train['nonzero_mean'][pred_train_temp==0]
            # print(f'Baseline Train Score = {sqrt(mean_squared_error(np.log1p(train.target), np.log1p(pred_train_temp)))}')
    
            pred_test_final = pred_test[0].copy()
            for r in range(1,len(pred_test)):
                pred_test_final[pred_test_final == 0] = pred_test[r][pred_test_final == 0]
    
            test_leak_match_count = (pred_test_final!=0).sum()
            no_match_count = (pred_test_final==0).sum()
            # print ("Test leak count: ", test_leak_match_count, "Test no leak count: ",  no_match_count)
    
            ##Make Leak Baseline
            pred_test_temp = pred_test_final.copy()
            test_og["nonzero_mean"] = test_og[[f for f in test_og.columns if f not in ["ID", "target", "nonzero_mean", "has_ugly"]]].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
            pred_test_temp[pred_test_temp==0] = test_og['nonzero_mean'][pred_test_temp==0]
            test_og['target']=pred_test_temp
            # test_og[['ID', 'target']].to_csv((processed_data_dir + 'leak_baseline_{}.csv'.format(test_leak_match_count)), index=False)
            
            # test_leaks = pd.read_csv(raw_data_dir + "sample_submission.csv")
            # del test_leaks['target']
            # test_leaks['target']=pred_test_final
            # test_leaks.to_csv((processed_data_dir + 'leak_only_{}.csv'.format(test_leak_match_count)), index=False)
             
            ### Create 40 features
            extra_features_list = []
    
            for ef in extra_features:
                extra_features_list.extend(ef)
    
            extra_features_list.extend(features)
            len(extra_features_list)
    
            #This makes the 100 40 length feature groups into 40 100 length feature groups. 
            #These 100 size vectors is what I would have liked to feed into an LSTM\CNN but I never got a chance to try this
            feats = pd.DataFrame(extra_features) 
            time_features = []
            for c in feats.columns[:]:    
                time_features.append([f for f in feats[c].values if f is not None])
    
    
            #Make a bunch of different feature groups to build aggregates from
            agg_features = []
            all_cols = train.columns.drop(['ID', 'target', 'nonzero_mean'])
            agg_features.append(all_cols)
            agg_features.append([c for c in all_cols if c not in extra_features_list])
            agg_features.append(extra_features_list)
            agg_features.extend(time_features)
            agg_features.extend(extra_features)
    
            #I made more aggregate feature to select from in model\feature selection. 
            #See this thread for some more aggregate ideas
            #https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/62446
    
            def add_new_features(source, dest, feats):
                high = source[feats].max(axis=1)
                high.name = 'high_{}_{}'.format(feats[0], len(feats))                
                mean = source[feats].replace(0, np.nan).mean(axis=1)
                mean.name = 'mean_{}_{}'.format(feats[0], len(feats))
                low = source[feats].replace(0, np.nan).min(axis=1)
                low.name = 'low_{}_{}'.format(feats[0], len(feats))
                median = source[feats].replace(0, np.nan).median(axis=1)
                median.name = 'median_{}_{}'.format(feats[0], len(feats))
                sum = source[feats].sum(axis=1)
                sum.name = 'sum_{}_{}'.format(feats[0], len(feats))
                stddev = source[feats].std(axis=1)
                stddev.name = 'stddev_{}_{}'.format(feats[0], len(feats))
                first_nonZero= np.log1p(source[feats].replace(0, np.nan).bfill(axis=1).iloc[:, 0])
                first_nonZero.name = 'first_nonZero_{}_{}'.format(feats[0], len(feats))
                last_nonZero = np.log1p(source[feats[::-1]].replace(0, np.nan).bfill(axis=1).iloc[:, 0])
                last_nonZero.name = 'last_nonZero_{}_{}'.format(feats[0], len(feats))
                nb_nans =  source[feats].replace(0, np.nan).isnull().sum(axis=1)
                nb_nans.name = 'nb_nans_{}_{}'.format(feats[0], len(feats))
                unique = source[feats].nunique(axis=1)
                unique.name = 'unique_{}_{}'.format(feats[0], len(feats))
                
                dest = pd.concat([dest, high, mean, low, median, sum, stddev, first_nonZero, last_nonZero, nb_nans, unique],axis=1)
                return dest
                # ############
                # dest['high_{}_{}'.format(feats[0], len(feats))] = source[feats].max(axis=1)
                # dest['mean_{}_{}'.format(feats[0], len(feats))] = source[feats].replace(0, np.nan).mean(axis=1)
                # dest['low_{}_{}'.format(feats[0], len(feats))] = source[feats].replace(0, np.nan).min(axis=1)
                # dest['median_{}_{}'.format(feats[0], len(feats))] = source[feats].replace(0, np.nan).median(axis=1)
                # dest['sum_{}_{}'.format(feats[0], len(feats))] = source[feats].sum(axis=1)
                # dest['stddev_{}_{}'.format(feats[0], len(feats))] = source[feats].std(axis=1)
    
                # dest['first_nonZero_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats].replace(0, np.nan).bfill(axis=1).iloc[:, 0])
                # dest['last_nonZero_{}_{}'.format(feats[0], len(feats))] = np.log1p(source[feats[::-1]].replace(0, np.nan).bfill(axis=1).iloc[:, 0])
    
    
                # dest['nb_nans_{}_{}'.format(feats[0], len(feats))] =  source[feats].replace(0, np.nan).isnull().sum(axis=1)
                # dest['unique_{}_{}'.format(feats[0], len(feats))] = source[feats].nunique(axis=1)

    
            #now that leak is done we should get back ugly data for feature engineering. This might not be necessary.
            del test
            gc.collect
            test = pd.read_csv(raw_data_dir + 'test.csv',engine="pyarrow")
            if self.toy_example:
                test = test.iloc[:1000]
                
            
            train_feats = pd.DataFrame()
            test_feats = pd.DataFrame()
    
            for i, ef in list(enumerate(agg_features)):        
                train_feats = add_new_features(train, train_feats, ef)
                test_feats = add_new_features(test, test_feats, ef)        

            ### Obtain final dataset
            cols = train_feats.columns 
            train_feat_final = pd.concat([train_feats[cols], test_feats[cols][self.test_leaks.target != 0]], axis = 0)
            train_feat_id = pd.concat([train['ID'], test['ID'][self.test_leaks.target != 0]], axis = 0)
            test_feat_final = test_feats[cols]    
            y_train = pd.Series(list(np.log1p(train.target.values)) + list(np.log1p(self.test_leaks['target'][self.test_leaks.target != 0])), index=train_feat_final.index, name=self.y_col)
            self.heavy_tailed = False
    
            X_train = train_feat_final
            X_test = test_feat_final
    
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = []
            self.heavy_tailed = False

        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_postprocessing(self, X, y, **kwargs):
        return np.expm1(y)

    def pred_to_submission(self, y_pred):

        y_pred[y_pred<0] = 1e-5
        
        submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sample_submission.csv", engine="pyarrow")
        if self.toy_example:
            submission = submission.iloc[:1000]
        submission[self.y_col] = y_pred
        
        submission.loc[self.test_leaks.target != 0,'target'] = self.test_leaks.loc[self.test_leaks.target != 0,'target'].astype(submission["target"].dtype)

        return submission

################################################################
################################################################
################################################################



class AmazonEmployeeAccessDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "amazon-employee-access-challenge"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [0,1,2,3,4,5,6,7,8]       
        self.y_col = "Action"
        self.large_dataset = False
    
    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        y_train = data["ACTION"]
        y_train.name = self.y_col
        X_train = data.drop("ACTION",axis=1)    
        
        X_test.drop("id",axis=1,inplace=True)    

        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Solution implemented based on the descriptions in https://www.kaggle.com/competitions/amazon-employee-access-challenge/discussion/5283
        '''

        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            # X_train_orig = X_train.copy()
            # X_test_orig = X_test.copy()
            
            ### BEN FEATURES
            # X_train = X_train.drop(['ROLE_CODE'], axis=1)
            # X_test = X_test.drop(['ROLE_CODE'], axis=1)
            X_all = pd.concat([X_test, X_train], ignore_index=True)
            
            # I want to combine role_title as a subset of role_familia and
            # X_all['ROLE_TITLE'] = X_all['ROLE_TITLE'] + (1000 * X_all['ROLE_FAMILY'])
            # X_all['ROLE_ROLLUPS'] = X_all['ROLE_ROLLUP_1'] + (10000 * X_all['ROLE_ROLLUP_2'])
            # X_all = X_all.drop(['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_FAMILY'], axis=1)

            # self.cat_indices = [0,1,2,3,4,5]
            original_cols = X_all.columns[self.cat_indices]
            
            # Count/freq
            for col in X_all.columns:
                if use_test:
                    counts = X_all.groupby(col)[col].transform('count')
                    X_all['cnt'+col] = np.log(counts+1)
                else:
                    cnt_dict = dict(X_train[col].value_counts())
                    X_train[col].map(cnt_dict)                    
                    X_test[col].map(cnt_dict).fillna(0.)                    
            
            # Percent of dept that is this resource
            # And Counts of dept/resource occurancesa (tested, not used)
            # Group by the column and 'RESOURCE', then calculate the size of each group
            for col in ['MGR_ID', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY']:
                if use_test:
                    group_sizes = X_all.groupby([col, 'RESOURCE']).transform('size')
                    # Calculate the size of each group by the column only
                    col_group_sizes = X_all.groupby(col).transform('size')
                    # Calculate the ratio
                    X_all['Duse' + col] = group_sizes / col_group_sizes
                else:
                    group_sizes_dict = dict(X_train[[col, "RESOURCE"]].value_counts())
                    
                    # Calculate the size of each group by the column only
                    col_group_sizes_dict = dict(X_train[col].value_counts())
                    # Calculate the ratio
                    X_train['Duse' + col] =  X_train[col].map(group_sizes_dict)   /  X_train[col].map(col_group_sizes_dict)  
                    X_test['Duse' + col] =  X_test[col].map(group_sizes_dict)   /  (X_test[col].map(col_group_sizes_dict)+1e-5)  
                    X_test['Duse' + col] = X_test['Duse' + col].fillna(0)
            # Number of resources that a manager manages
            # Count the number of unique entries in `col` for each manager
            if use_test:
                X_all['Mdeps' +'RESOURCE'] = X_all.groupby('MGR_ID')['RESOURCE'].transform(lambda x: x.nunique())
            else:
                cnt_dict = dict(X_train.groupby('MGR_ID').nunique()['RESOURCE'])
                X_train['Mdeps' +'RESOURCE'] = X_train['MGR_ID'].map(cnt_dict)
                X_test['Mdeps' +'RESOURCE'] = X_test['MGR_ID'].map(cnt_dict).fillna(0)
            # X_all = X_all.drop(X_all.columns[0:6], axis=1)
    
            # Adding cross-tabulated count features
            def add_crosstab_features(df1, col1, col2, df2=None):
                # Creating a crosstab count matrix for col1 and col2
                crosstab_counts = pd.crosstab(df1[col1], df1[col2])
                # Flattening the crosstab to enable mapping
                crosstab_flat = crosstab_counts.stack().reset_index(name=f'{col1}_{col2}_count')
                # Merging the counts back into the original dataframe
                new_feat1 = df1.merge(crosstab_flat, how='left', left_on=[col1, col2], right_on=[col1, col2])[f'{col1}_{col2}_count']
                if df2 is not None:
                    new_feat2 = df2.merge(crosstab_flat, how='left', left_on=[col1, col2], right_on=[col1, col2])[f'{col1}_{col2}_count']
                else:
                    new_feat2 = None
                
                return new_feat1, new_feat2        

            if use_test:
                new_cols = []
            else:
                new_cols_train = []
                new_cols_test = []
            for col_1 in original_cols:
                for col_2 in original_cols:
                    # print(f"Get new feature: {col_1}_{col_2}_count")
                    if (col_1!=col_2) and (f'{col_2}_{col_1}_count' not in X_all.columns):
                        if use_test:
                            new_col, _ = add_crosstab_features(X_all, col_1, col_2)
                            new_cols.append(new_col)
                            new_col_log = np.log(new_col+1)
                            new_col_log.name = f'{col_1}_{col_2}_count_log' 
                            new_cols.append(new_col_log)
                        
                            # X_all[f'{col_1}_{col_2}_count_log'] = np.log(X_all[f'{col_1}_{col_2}_count']+1).copy()
                        else:
                            new_col_train, new_col_test = add_crosstab_features(X_train, col_1, col_2, X_test)
                            
                            new_cols_train.append(new_col_train)
                            new_col_log_train = np.log(new_col_train+1)
                            new_col_log_train.name = f'{col_1}_{col_2}_count_log' 
                            new_cols_train.append(new_col_log_train)
                            
                            new_cols_test.append(new_col_test)
                            new_col_log_test = np.log(new_col_test+1)
                            new_col_log_test.name = f'{col_1}_{col_2}_count_log' 
                            new_cols_test.append(new_col_log_test)

                            
                            # X_train[f'{col_1}_{col_2}_count_log'] = np.log(X_train[f'{col_1}_{col_2}_count']+1)
                            # X_test[f'{col_1}_{col_2}_count_log'] = np.log(X_test[f'{col_1}_{col_2}_count']+1)

            if use_test:
                X_all = pd.concat([X_all]+new_cols, axis=1)
            else:
                X_train = pd.concat([X_train]+new_cols_train, axis=1)
                X_test = pd.concat([X_test]+new_cols_test, axis=1)
                
            # Add original columns as numeric (from 3rd place solution) - not used
            # for col in self.cat_indices:
            #     X_all[str(X_all.columns[col])+"_numeric"] = X_all.iloc[:,col].astype(float) 
            
            
            if use_test:
                new_cols = []
            else:
                new_cols_train = []
                new_cols_test = []
            # Add two-way-interactions
            counter = 0
            for col_1 in original_cols:
                for col_2 in original_cols:
                    new_col_name = f'{col_1}x{col_2}'
                    # print(f"Get new feature: {col_1}x{col_2}")
                    if use_test:
                        if (col_1!=col_2) and (new_col_name not in X_all.columns):
                            new_col =  (X_all[col_1].astype(str)+X_all[col_2].astype(str)).astype(int).copy()
                            new_col.name = new_col_name
                            new_cols.append(new_col)
                            self.cat_indices.append(X_all.shape[1]+counter)
                            counter += 1
                    else:
                        # X_train[f'{col_1}x{col_2}'] =  (X_train[col_1].astype(str)+X_train[col_2].astype(str)).astype(int).copy()
                        # X_test[f'{col_1}x{col_2}'] =  (X_test[col_1].astype(str)+X_test[col_2].astype(str)).astype(int).copy()
                        # self.cat_indices.append(np.where(X_train.columns==f'{col_1}x{col_2}')[0][0])
                        if (col_1!=col_2) and (new_col_name not in X_train.columns):
                            new_col_train = (X_train[col_1].astype(str)+X_train[col_2].astype(str)).astype(int)
                            new_col_train.name = new_col_name
                            new_cols_train.append(new_col_train)
                            self.cat_indices.append(X_train.shape[1]+counter)
                            
                            new_col_test = (X_test[col_1].astype(str)+X_test[col_2].astype(str)).astype(int)
                            new_col_test.name = new_col_name
                            new_cols_test.append(new_col_test)    
                        counter += 1
        
            
            if use_test:
                X_all = pd.concat([X_all]+new_cols, axis=1)
            else:
                X_train = pd.concat([X_train]+new_cols_train, axis=1)
                X_test = pd.concat([X_test]+new_cols_test, axis=1)
            
            # Add three-way interactions and counts
            # Generate all possible combinations of three categorical columns
            three_way_combinations = combinations(original_cols, 3)

            if use_test:
                new_cols = []
            else:
                new_cols_train = []
                new_cols_test = []

            # Iterate over each combination to create interaction features
            for num, cols in enumerate(three_way_combinations):
                # Create a new column name based on the columns being combined
                new_col_name = '_X_'.join(cols)
                
                # Concatenate the values of the three columns for each row to create the interaction feature
                # The interaction is represented as a string, with values separated by a special character (e.g., ':')
                if use_test:
                    new_col = X_all[list(cols)].apply(lambda row: ':'.join(row.values.astype(str)), axis=1).copy()
                    new_col.name = new_col_name
                    new_cols.append(new_col)
                    self.cat_indices.append(X_all.shape[1]+num)
                    
                    # X_all[new_col_name] = X_all[list(cols)].apply(lambda row: ':'.join(row.values.astype(str)), axis=1).copy()
                    # self.cat_indices.append(X_all.shape[1]+1+num)
                else:
                    # X_train[new_col_name] = X_train[list(cols)].apply(lambda row: ':'.join(row.values.astype(str)), axis=1).copy()
                    # X_test[new_col_name] = X_test[list(cols)].apply(lambda row: ':'.join(row.values.astype(str)), axis=1).copy()
                    # self.cat_indices.append(np.where(X_train.columns==new_col_name)[0][0])
                    new_col_train = X_train[list(cols)].apply(lambda row: ':'.join(row.values.astype(str)), axis=1).copy()
                    new_col_train.name = new_col_name
                    new_cols_train.append(new_col_train)
                    self.cat_indices.append(X_train.shape[1]+num)
                    
                    new_col_test = X_test[list(cols)].apply(lambda row: ':'.join(row.values.astype(str)), axis=1).copy()
                    new_col_test.name = new_col_name
                    new_cols_test.append(new_col_test)
            
            if use_test:
                X_all = pd.concat([X_all]+new_cols, axis=1)
            else:
                X_train = pd.concat([X_train]+new_cols_train, axis=1)
                X_test = pd.concat([X_test]+new_cols_test, axis=1)

            ### Counts (overfits)
                # # Create a unique identifier for each combination by concatenating their values
                # interaction_col = X_all[list(cols)].astype(str).apply(lambda x: '_'.join(x), axis=1)
                # # Calculate the count of each unique combination
                # interaction_counts = interaction_col.value_counts()#.rename(lambda x: '__'.join(list(cols)) + '_count')
                
                # # Map the counts back to the original DataFrame based on the interaction identifier
                # X_all['__'.join(cols) + '_count'] = interaction_col.map(interaction_counts)
                # X_all['__'.join(cols) + '_count_log'] = np.log(X_all['__'.join(cols) + '_count'])
    
            # Add numeric version of base features (doesnt help)
            # X_all[[i+"_numeric" for i in X_all.columns[:6]]] = X_all[X_all.columns[:6]].astype(float)

            # create some polynomial features  (doesnt help)
            # for i in range(X_all.shape[1]): 
            #     if X_all.iloc[:,i].dtype=="float" or X_all.iloc[:,i].dtype=="int": 
            #         X_all["poly_1_"+str(i)] = [round(a/(b + 1), 3) for a, b in zip(X_all.iloc[:,i], X_all.iloc[:,0])]
            #         X_all["poly_1_"+str(i)] = X_all["poly_1_"+str(i)].astype(float)
                    
            #         X_all["poly_2_"+str(i)] = [round(a/(b + 1), 3) for a, b in zip(X_all.iloc[:,0], X_all.iloc[:,i])]
            #         X_all["poly_2_"+str(i)] = X_all["poly_2_"+str(i)].astype(float)
            
            #         X_all["poly_3_"+str(i)] = [a*b for a, b in zip(X_all.iloc[:,0], X_all.iloc[:,i])]
            #         X_all["poly_3_"+str(i)] = X_all["poly_3_"+str(i)].astype(float)
            
            # Now X is the train, X_test is test and X_all is both together
            if use_test:
                X_train = X_all[:][X_all.index >= len(X_test.index)]
                X_test = X_all[:][X_all.index < len(X_test.index)]
                X_train.reset_index(inplace=True,drop=True) 

           #  # Drop original features
           #  X_train.drop(COLNAMES,axis=1,inplace=True)
           #  X_test.drop(COLNAMES,axis=1,inplace=True)
            
            # remove constant features
            constant = X_train.columns[np.logical_and(X_train.nunique()==1,X_test.nunique()==1)]
            X_train.drop(constant,axis=1,inplace=True)
            X_test.drop(constant,axis=1,inplace=True)        
            
            
            for col in X_train.columns[self.cat_indices]:
                # Note: The category dtype needs to entail all train categories when applying .astype("category") on test data
                dtype = pd.CategoricalDtype(categories=list(X_train[col].astype(str).fillna("nan").unique()))
                X_train[col] = X_train[col].astype(str).fillna("nan").astype(dtype)
                X_test[col] = X_test[col].astype(str).fillna("nan").astype(dtype)       

                
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))
        
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
            for col in X_train.columns[self.cat_indices]:
                # Note: The category dtype needs to entail all train categories when applying .astype("category") on test data
                dtype = pd.CategoricalDtype(categories=list(X_train[col].astype(str).fillna("nan").unique()))
                X_train[col] = X_train[col].astype(str).fillna("nan").astype(dtype)
                X_test[col] = X_test[col].astype(str).fillna("nan").astype(dtype)       

        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    def get_cv_folds(self, X_train, y_train, seed=42):
        ## !! Currently not original implemented - original solution used 30-fold CV - but also dicusses 5-fold
        ss = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        folds = []
        for num, (train,test) in enumerate(ss.split(X_train.copy(), y_train.copy())):
            folds.append([train, test])  

        return folds
        


################################################################
################################################################
################################################################



################################################################
################################################################
################################################################

class SantanderSatisfactionDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "santander-customer-satisfaction"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []
        self.y_col = "TARGET"
        self.large_dataset = False

    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Solution implemented based on one of the descriptions in https://www.kaggle.com/competitions/santander-customer-satisfaction/discussion/20978:

        Following data cleaning and feature engineering tricks were used:
        1. Removing almost all indicator variables (ind_*); they cover same info provided by num * variables.
        2. Removing constant variables.
        3. Removing variables which non-0 attributes have 0,1,2,3 TARGET=1 instances.
        4. Setting var38==117310.979016494 ⇒ var38=NA (missing at random); assumption that Santander mean imputed NA’s.
        5. Summarizing how many 0’s, 3’s, 6’s, 9’s, {X mod 3 == 0}’s appear in each row.
        6. Calculating var38 percentile rank within same var15 for each instance.
        7. Calculating var38 percentile rank within +/-1 var15 for each instance.
        8. Calculating var38 percentile rank within +/-2 var15 for each instance.
        9. Calculating ratios of ult1/ult3 variables.
        10. Calculating ratios of hace2/hace3 variables.
        11. Calculating if X mod 3 == 0 (for money type variables).
        12. Special feature seperating 2 distint population segments within data
        (file data/features/population_split.csv)

        Elements that were not listed in the description but found in the code are also implemented.
        What others did:
        ○ Ikki Tanaka:
            § Counts of integer variables
            § Dummy, boolean, log
            § Limit variables in test based on train (only for NNs)
            § Log transform for NNs
        ○ Marios
            § Cat counts
            § Likelihood of cats
            § Tsne, PCA & KNN Feats
        ○ Mathias: Heavy feature selection

        '''
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            
            # from tsne import bh_sne
            
            OUTPUT_PATH = "./datasets/santander-customer-satisfaction/processed/"
            
            # Second preprocessing version for the santander customer satisfaction data set ("Dmitry")
            
            
            def process_base(train, test):
                train.loc[
                    (train["var38"] > 117310.979) & (train["var38"] < 117310.98), "var38"
                ] = -999.0
                test.loc[
                    (test["var38"] > 117310.979) & (test["var38"] < 117310.98), "var38"
                ] = -999.0
            
                train.loc[train["var3"] == -999999, "var3"] = -999.0
                test.loc[test["var3"] == -999999, "var3"] = -999.0
            
                for f in [
                    "imp_op_var40_comer_ult1",
                    "imp_op_var40_efect_ult3",
                    "imp_op_var41_comer_ult3",
                    "imp_sal_var16_ult1",
                ]:
                    train.loc[train[f] == 0.0, f] = -999.0
                    test.loc[test[f] == 0.0, f] = -999.0
            
                return train, test
            
            
            def drop_sparse(train, test):
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
                for f in flist:
                    if len(np.unique(train[f])) < 2:
                        train.drop(f, axis=1, inplace=True)
                        test.drop(f, axis=1, inplace=True)
                return train, test
            
            
            def drop_duplicated(train, test):
                # drop var6 variable (it is similar to var29)
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
                train.drop([x for x in flist if "var6" in x], axis=1, inplace=True)
                test.drop([x for x in flist if "var6" in x], axis=1, inplace=True)
            
                # remove repeated columns with _0 in the name
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
                flist_remove = []
                for i in range(len(flist) - 1):
                    v = train[flist[i]].values
                    for j in range(i + 1, len(flist)):
                        if np.array_equal(v, train[flist[j]].values):
                            if "_0" in flist[j]:
                                flist_remove.append(flist[j])
                            elif "_0" in flist[i]:
                                flist_remove.append(flist[i])
                train.drop(flist_remove, axis=1, inplace=True)
                test.drop(flist_remove, axis=1, inplace=True)
            
                flist_remove = [
                    "saldo_medio_var13_medio_ult1",
                    "delta_imp_reemb_var13_1y3",
                    "delta_imp_reemb_var17_1y3",
                    "delta_imp_reemb_var33_1y3",
                    "delta_imp_trasp_var17_in_1y3",
                    "delta_imp_trasp_var17_out_1y3",
                    "delta_imp_trasp_var33_in_1y3",
                    "delta_imp_trasp_var33_out_1y3",
                ]
                train.drop(flist_remove, axis=1, inplace=True)
                test.drop(flist_remove, axis=1, inplace=True)
            
                return train, test
            
            
            def add_features(train, test, features, use_test=True):
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
                if "SumZeros" in features:
                    train.insert(1, "SumZeros", (train[flist] == 0).astype(int).sum(axis=1))
                    test.insert(1, "SumZeros", (test[flist] == 0).astype(int).sum(axis=1))
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
            
                """
                if "tsne" in features:
                    try:
                        tsne_feats = pd.read_csv(OUTPUT_PATH + "tsne_feats.csv")
                        train = pd.merge(train, tsne_feats, on="ID", how="left")
                        test = pd.merge(test, tsne_feats, on="ID", how="left")
                    except FileNotFoundError:
                        train, test = tsne_features(train, test)
                """
            
                if "pca" in features:
                    try:
                        pca_feats = pd.read_csv(OUTPUT_PATH + "pca_feats.csv")
                        train = pd.merge(train, pca_feats, on="ID", how="left")
                        test = pd.merge(test, pca_feats, on="ID", how="left")
                    except FileNotFoundError:
                        train, test = pca_features(train, test)
            
                if "kmeans" in features:
                    # try:
                    #     kmeans_feats = pd.read_csv(OUTPUT_PATH + "kmeans_feats.csv")
                    #     train = pd.merge(train, kmeans_feats, on="ID", how="left")
                    #     test = pd.merge(test, kmeans_feats, on="ID", how="left")
                    # except FileNotFoundError:
                    train, test = kmeans_features(train, test, use_test)
            
                return train, test
            
            
            def normalize_features(train, test, use_test=True):
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
                for f in flist:
                    if train[f].max() == 9999999999.0:
                        fmax = train.loc[train[f] < 9999999999.0, f].max()
                        train.loc[train[f] == 9999999999.0, f] = fmax + 1
            
                    train[f] = train[f].astype("float64")
                    test[f] = test[f].astype("float64")
            
                    if len(train.loc[train[f] < 0, f].value_counts()) == 1:
                        train.loc[train[f] < 0, f] = -1.0
                        test.loc[test[f] < 0, f] = -1.0
                        if use_test:
                            fmax = max(np.max(train[f]), np.max(test[f]))
                        else:
                            fmax = np.max(train[f])
                        if fmax > 0:
                            train.loc[train[f] > 0, f] = train.loc[train[f] > 0, f] / fmax
                            test.loc[test[f] > 0, f] = test.loc[test[f] > 0, f] / fmax
            
                    if len(train.loc[train[f] < 0, f]) == 0:
                        if use_test:
                            fmax = max(np.max(train[f]), np.max(test[f]))
                        else:
                            fmax = np.max(train[f])
                        if fmax > 0:
                            train.loc[train[f] > 0, f] = train.loc[train[f] > 0, f] / fmax
                            test.loc[test[f] > 0, f] = test.loc[test[f] > 0, f] / fmax
            
                    if len(train.loc[train[f] < 0, f].value_counts()) > 1:
                        if use_test:
                            fmax = max(np.max(train[f]), np.max(test[f]))
                        else:
                            fmax = np.max(train[f])
                        if fmax > 0:
                            train[f] = train[f] / fmax
                            test[f] = test[f] / fmax
            
                return train, test
            
            
            def add_likelihood_feature(fname, train_likeli, test_likeli, flist):
                tt_likeli = pd.DataFrame()
                np.random.seed(1232345)
                skf = StratifiedKFold(
                    train_likeli["TARGET"].values, n_folds=5, shuffle=True, random_state=21387
                )
                for train_index, test_index in skf:
                    ids = train_likeli["ID"].values[train_index]
                    train_fold = train_likeli.loc[train_likeli["ID"].isin(ids)].copy()
                    test_fold = train_likeli.loc[~train_likeli["ID"].isin(ids)].copy()
                    global_avg = np.mean(train_fold["TARGET"].values)
                    feats_likeli = (
                        train_fold.groupby(fname)["TARGET"]
                        .agg({"sum": np.sum, "count": len})
                        .reset_index()
                    )
                    feats_likeli[fname + "_likeli"] = (feats_likeli["sum"] + 30.0 * global_avg) / (
                        feats_likeli["count"] + 30.0
                    )
                    test_fold = pd.merge(
                        test_fold, feats_likeli[[fname, fname + "_likeli"]], on=fname, how="left"
                    )
                    test_fold[fname + "_likeli"] = test_fold[fname + "_likeli"].fillna(global_avg)
                    tt_likeli = tt_likeli.append(
                        test_fold[["ID", fname + "_likeli"]], ignore_index=True
                    )
                train_likeli = pd.merge(train_likeli, tt_likeli, on="ID", how="left")
            
                global_avg = np.mean(train_likeli["TARGET"].values)
                feats_likeli = (
                    train_likeli.groupby(fname)["TARGET"]
                    .agg({"sum": np.sum, "count": len})
                    .reset_index()
                )
                feats_likeli[fname + "_likeli"] = (feats_likeli["sum"] + 30.0 * global_avg) / (
                    feats_likeli["count"] + 30.0
                )
                test_likeli = pd.merge(
                    test_likeli, feats_likeli[[fname, fname + "_likeli"]], on=fname, how="left"
                )
                test_likeli[fname + "_likeli"] = test_likeli[fname + "_likeli"].fillna(global_avg)
                return train_likeli, test_likeli, flist + [fname + "_likeli"]
            
            
            """
            def tsne_features(train, test):
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
            
                ### add TSNE features
                X = train[flist].append(test[flist], ignore_index=True).values.astype("float64")
                svd = TruncatedSVD(n_components=30)
                X_svd = svd.fit_transform(X)
                X_scaled = StandardScaler().fit_transform(X_svd)
                feats_tsne = bh_sne(X_scaled)
                feats_tsne = pd.DataFrame(feats_tsne, columns=["tsne1", "tsne2"])
                feats_tsne["ID"] = (
                    train[["ID"]].append(test[["ID"]], ignore_index=True)["ID"].values
                )
                train = pd.merge(train, feats_tsne, on="ID", how="left")
                test = pd.merge(test, feats_tsne, on="ID", how="left")
            
                feat = train[["ID", "tsne1", "tsne2"]].append(
                    test[["ID", "tsne1", "tsne2"]], ignore_index=True
                )
                feat.to_csv(OUTPUT_PATH + "tsne_feats.csv", index=False)
            
                return train, test
            """
            
            
            def pca_features(train, test):
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
            
                pca = PCA(n_components=2)
                x_train_projected = pca.fit_transform(normalize(train[flist], axis=0))
                x_test_projected = pca.transform(normalize(test[flist], axis=0))
                train.insert(1, "PCAOne", x_train_projected[:, 0])
                train.insert(1, "PCATwo", x_train_projected[:, 1])
                test.insert(1, "PCAOne", x_test_projected[:, 0])
                test.insert(1, "PCATwo", x_test_projected[:, 1])
            
                pca_feats = pd.concat(
                    [train[["ID", "PCAOne", "PCATwo"]], test[["ID", "PCAOne", "PCATwo"]]],
                    ignore_index=True,
                )
                pca_feats.to_csv(OUTPUT_PATH + "pca_feats.csv")
            
                return train, test
            
            
            def kmeans_features(train, test, use_test=True):
                train, test = normalize_features(train, test, use_test)
            
                flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
            
                flist_kmeans = []
                if use_test:
                    train_test = pd.concat([train,test],axis=0)
                for ncl in range(2, 11):
                    cls = KMeans(n_clusters=ncl)
                    if use_test:
                        cls.fit_predict(train_test[flist].values)
                    else:
                        cls.fit_predict(train[flist].values)
                    key = "kmeans_cluster" + str(ncl)
                    train[key] = cls.predict(train[flist].values)
                    # train_cluster[key] = cls.predict(train[flist].values)
                    test[key] = cls.predict(test[flist].values)
                    # test_cluster[key] = cls.predict(test[flist].values)
                    flist_kmeans.append("kmeans_cluster" + str(ncl))
            
                # print(flist_kmeans[0] in train.columns)
            
                pd.concat(
                    [train[["ID"] + flist_kmeans], test[["ID"] + flist_kmeans]], ignore_index=True
                ).to_csv(OUTPUT_PATH + "kmeans_feats.csv", index=False)
            
                return train, test

            def tsne_features(train, test):
                if not os.path.exists(OUTPUT_PATH + 'tsne_feats.csv'):
                    train, test = process_base(train, test)
                    train, test = drop_sparse(train, test)
                    train, test = drop_duplicated(train, test)
                    train, test = add_features(train, test, ['SumZeros'])
                    train, test = normalize_features(train, test)
                    
                    flist = [x for x in train.columns if x not in ["ID", "TARGET"]]
                    
                    X = pd.concat([train[flist],test[flist]],axis=0, ignore_index=True).fillna(0).values.astype('float64')
                    
                    svd = TruncatedSVD(n_components=30)
                    X_svd = svd.fit_transform(X)
                    
                    X_scaled = StandardScaler().fit_transform(X_svd)
                    
                    tsne = TSNE(n_components=2, verbose=0)
                    X_2d = tsne.fit_transform(X_scaled)
                    feats_tsne = pd.DataFrame(X_2d, columns=['tsne1', 'tsne2'])
                    
                    feats_tsne['ID'] = pd.concat([train["ID"],test["ID"]], ignore_index=True).values
                    feats_tsne.to_csv(OUTPUT_PATH + 'tsne_feats.csv', index=False)

                else:
                    feats_tsne = pd.read_csv(OUTPUT_PATH + 'tsne_feats.csv')        
                test = feats_tsne.iloc[train.shape[0]:]
                train = feats_tsne.iloc[:train.shape[0]]
                
                return train, test            
            
            
            def second_expert_preprocessing(train, test, feature_set, use_test=True):
                train, test = process_base(train, test)
                train, test = drop_sparse(train, test)
                train, test = drop_duplicated(train, test)
                train, test = add_features(train, test, feature_set, use_test)
            
                # print(len(train.columns), train.head(3))
            
                return train, test

            # X_train_second_expert, X_test_second_expert = second_expert_preprocessing(
            #             X_train, X_test, ["SumZeros", "pca", "kmeans"]
            #         )

            # X_train_pca, X_test_pca = second_expert_preprocessing(
            #     X_train.copy(), X_test.copy(), ["pca"], use_test
            # )
            # X_train_pca = X_train_pca[["ID", "PCAOne",	"PCATwo"]] 
            # X_test_pca = X_test_pca[["ID", "PCAOne",	"PCATwo"]]
            

            
            X_train_kmeans, X_test_kmeans = second_expert_preprocessing(
                X_train.copy(), X_test.copy(), ["kmeans"], use_test
            )
            X_train_kmeans = X_train_kmeans[["ID", "kmeans_cluster2", "kmeans_cluster3",	"kmeans_cluster4",	"kmeans_cluster5",	"kmeans_cluster6",	"kmeans_cluster7",	"kmeans_cluster8",	"kmeans_cluster9",	"kmeans_cluster10"]]
            X_test_kmeans = X_test_kmeans[["ID", "kmeans_cluster2", "kmeans_cluster3",	"kmeans_cluster4",	"kmeans_cluster5",	"kmeans_cluster6",	"kmeans_cluster7",	"kmeans_cluster8",	"kmeans_cluster9",	"kmeans_cluster10"]]

            # X_train_tsne, X_test_tsne = tsne_features(
            #     X_train.copy(), X_test.copy()
            # )
            
#####################################################
            
            # Combine train and test for preprocessing, potentially split them up for more realistic handling later
            X_train_with_y = pd.concat([X_train, y_train], axis=1)
            X_test_with_y = X_test
            X_test_with_y[self.y_col] = np.nan
    
            combined_data = pd.concat([X_train_with_y, X_test_with_y])
    
            target_counts = combined_data.groupby(self.y_col).size()
            print(target_counts)
    
            # 0. Data Cleaning (Move to laod data)
            combined_data.loc[combined_data["var3"]==-999999, "var3"] = -1
            combined_data.loc[combined_data["delta_num_aport_var13_1y3"]==9999999999,"delta_num_aport_var13_1y3"] = 10
            combined_data.loc[combined_data["delta_imp_aport_var13_1y3"]==9999999999,"delta_imp_aport_var13_1y3"] = 10 
            
            combined_data["sum_zeros"] = (combined_data.drop(["ID", "TARGET"],axis=1)==0).sum(axis=1)
            
            # 1. Remove all indicator variables
            def remove_indicator_vars(data):  
                ind_var_cols = [col for col in data.columns if col.startswith('ind_')]

              #   # Originally dropped 
              #   ['ind_var1_0','ind_var1','ind_var2_0','ind_var2', 'ind_var5_0','ind_var5','ind_var6_0','ind_var6', 
              #   'ind_var7_emit_ult1','ind_var7_recib_ult1','ind_var8_0','ind_var8','ind_var12_0','ind_var12',
              #   'ind_var13_0','ind_var13', 'ind_var13_corto_0','ind_var13_corto', 'ind_var13_medio_0','ind_var13_medio',
              #   'ind_var13_largo_0','ind_var13_largo', 'ind_var14_0','ind_var14','ind_var17_0','ind_var17',
              # 'ind_var18_0','ind_var18','ind_var20_0','ind_var20', 'ind_var24_0','ind_var24','ind_var25_0','ind_var25',         
              # 'ind_var26_0','ind_var26','ind_var27_0','ind_var27','ind_var28_0','ind_var28','ind_var29_0','ind_var29',
              # 'ind_var30_0','ind_var30','ind_var31_0','ind_var31','ind_var32_0','ind_var32','ind_var33_0','ind_var33',
              # 'ind_var34_0','ind_var34','ind_var37_0','ind_var37','ind_var39_0','ind_var39','ind_var40_0','ind_var40',
              # 'ind_var41_0','ind_var41','ind_var43_emit_ult1','ind_var43_recib_ult1',
              # 'ind_var44_0','ind_var44','ind_var46_0','ind_var46']
                
                data.drop(columns=ind_var_cols, inplace=True)

        
            
            # print("STEP 1")
            # remove_indicator_vars(combined_data)
    
            # 2. Remove constant variables
            def remove_const_vars(data):
                const_var_cols = [col for col in data.columns if data[col].nunique() == 1]
                data.drop(columns=const_var_cols, inplace=True)
    
            # print("STEP 2")
            # remove_const_vars(combined_data)
    
            # # 2.5 (not mentioned in pdf) Identify numeric columns with values > 0 that all point to TARGET 0 and aggregate them
            # def aggregate_target0_cols(data):
            #     target0cols = []
            #     for col in data.columns:
            #         if np.max(data[self.y_col][data[col] != 0].dropna()) == 0:
            #             target0cols.append(col)
    
            #     print(data[target0cols].shape)
    
            #     data['s0'] = data[target0cols].apply(lambda x: np.sum(np.where(x != 0, 1, 0)))
            #     data['s0'] = np.where(data.s0 > 1, 1, 0)
            #     data = data.drop(columns=target0cols)
    
            # aggregate_target0_cols(combined_data)
    
            # 3. Remove low-frequency variable-positive class combinations
            # def remove_low_freq_var_pos_class_combinations(data):
            #     # non-0 attributes have only 1 targets
            #     target1cols = []
            #     for col in set(data.columns) - {'s0'}:
            #         if data[data[col] != 0][self.y_col].sum(axis=0) == 1:
            #             target1cols.append(col)
    
            #     data['s1'] = data[target1cols].apply(lambda x: np.sum(np.where(x != 0, 1, 0)))
            #     data = data.drop(columns=target1cols)
    
            #     # non-0 attributes have only 2 targets
            #     target2cols = []
            #     for col in set(data.columns) - {'s0', 's1'}:
            #         if data[data[col] != 0][self.y_col].sum(axis=0) == 2:
            #             target2cols.append(col)
    
            #     data['s2'] = data[target2cols].apply(lambda x: np.sum(np.where(x != 0, 1, 0)))
            #     data = data.drop(columns=target2cols)
    
            #     # non-0 attributes have only 3 targets
            #     target3cols = []
            #     for col in set(data.columns) - {'s0', 's1', 's2'}:
            #         if data[data[col] != 0][self.y_col].sum(axis=0) == 3:
            #             target3cols.append(col)
    
            #     data['s3'] = data[target3cols].apply(lambda x: np.sum(np.where(x != 0, 1, 0)))
            #     data = data.drop(columns=target3cols)
    
            # print("STEP 3")
            # # Move drop at the end
            # # remove_low_freq_var_pos_class_combinations(combined_data)
            # usecols=combined_data.columns.drop("TARGET")
            # dropcols = usecols[[combined_data["TARGET"][combined_data[col]!=0].sum()<4 for col in usecols]]
            # # combined_data = combined_data.drop(dropcols,axis=1)

            # 4. Setting var38==117310.979016494 ⇒ var38=NA (+ adding bounds for var15 which is only added in the code)
            print("STEP 4") 
            combined_data.replace({'var38': 117310.979016494}, np.nan, inplace=True)
            combined_data.loc[combined_data['var15'] <= 22, 'var15'] = 22
            combined_data.loc[combined_data['var15'] >= 95, 'var15'] = 95

            # 5. Summarizing how many 0’s, 3’s, 6’s, 9’s, {X mod 3 == 0}’s appear in each row.
            def mod3_feature(data):
                def count_number(x, number):
                    return np.sum((x == number) & (~pd.isna(x)))
    
                def count_mod3(x):
                    return np.sum((x > 0) & (np.round(x, 2) % 3 == 0) & (~pd.isna(x)))
                
                for number in [0, 3, 6, 9]:
                    data[f'n_{number}'] = data.drop(columns=[self.y_col]).apply(lambda x: count_number(x, number))
    
                data['n_mod3'] = data.drop(columns=[self.y_col]).apply(count_mod3, axis=1)
            print("STEP 5")
            # mod3_feature(combined_data)
            combined_data["n_0"] = (combined_data.drop(self.y_col,axis=1)==0).sum(axis=1)
            combined_data["n_3"] = (combined_data.drop(self.y_col,axis=1)==3).sum(axis=1)
            combined_data["n_6"] = (combined_data.drop(self.y_col,axis=1)==6).sum(axis=1)
            combined_data["n_9"] = (combined_data.drop(self.y_col,axis=1)==9).sum(axis=1)
            combined_data['n_mod3'] = combined_data[["n_0","n_3","n_6","n_9"]].sum(axis=1)
            combined_data = combined_data.drop(["n_0","n_3","n_6","n_9"],axis=1)
            
            # # 6/7/8 Calculating var38 eCDF within same/+-1/+-2 var15 for each instance.
            # def var38_percentile_features(data):
            #     ecdf = data.reset_index()[['ID', 'var15', 'var38']].dropna(subset=['var38'])
            #     ecdf['rank'] = ecdf.groupby('var15')['var38'].rank(method='dense')
            #     ecdf['ecdf1'] = ecdf.groupby('var15')['rank'].transform(lambda x: x / x.max())
            #     ecdf = ecdf[['ID', 'ecdf1']]
    
            #     data = data.merge(ecdf, on='ID', how='left')
    
            #     # AUC calculation for ECDF within same age group
            #     from sklearn.metrics import roc_auc_score
            #     auc_same_age = roc_auc_score(data.loc[~data['TARGET'].isna() & ~data['var38'].isna(), 'TARGET'],
            #                                 data.loc[~data['TARGET'].isna() & ~data['var38'].isna(), 'var38'])
                
            #     auc_same_age1 = roc_auc_score(data.loc[~data['TARGET'].isna() & ~data['var38'].isna(), 'TARGET'],
            #                                 data.loc[~data['TARGET'].isna() & ~data['var38'].isna(), 'ecdf1'])
                
            #     # Control values
            #     print(auc_same_age, auc_same_age1)
    
            #     """
            #     def calculate_ecdf_in_range(data, x, col_name):
            #         ecdf = data.reset_index()[['ID', 'var15', 'var38']].dropna(subset=['var38'])
            #         def calc_ecdf_for_var15(var15):
            #             lower_bound = var15 - x
            #             upper_bound = var15 + x
            #             mask = (ecdf['var15'] >= lower_bound) & (ecdf['var15'] <= upper_bound)
            #             filtered_data = ecdf[mask]
            #             rank = filtered_data['var38'].rank(method='dense')
            #             ecdf_pm_range = rank / len(filtered_data)
            #             return ecdf_pm_range.reindex(ecdf.index, fill_value=np.nan)
            #         ecdf[col_name] = ecdf['var15'].apply(calc_ecdf_for_var15)
            #         ecdf = ecdf[['ID', col_name]]
            #         data = data.merge(ecdf, on='ID', how='left')
    
            #     calculate_ecdf_in_range(data, 1,'ecdf2')
            #     calculate_ecdf_in_range(data, 2, 'ecdf3')
            #     """
                
            #     # Repeat initial proess for var36 as an additional criterion
            #     ecdf = data.reset_index()[['ID', 'var15', 'var36', 'var38']].dropna(subset=['var38'])
            #     ecdf['rank'] = ecdf.groupby(['var15', 'var36'])['var38'].rank(method='dense')
            #     ecdf['ecdf4'] = ecdf.groupby(['var15', 'var36'])['rank'].transform(lambda x: x / x.max())
            #     ecdf = ecdf[['ID', 'ecdf4']]
    
            #     data = data.merge(ecdf, on='ID', how='left')
    
            #     # AUC calculation for ECDF within same age group
            #     auc_same_age_new = roc_auc_score(data.loc[~data['TARGET'].isna() & ~data['var38'].isna(), 'TARGET'],
            #                                 data.loc[~data['TARGET'].isna() & ~data['var38'].isna(), 'ecdf4'])
                
            #     print(auc_same_age_new)
    
            #     # 8.5 Clean resulting data from na
            #     # data.loc[data['var38'].isna(), ['var38', 'ecdf1', 'ecdf2', 'ecdf3', 'ecdf4']] = 0, 0, 0, 0, 0
            #     data.loc[data['var38'].isna(), ['var38', 'ecdf1', 'ecdf4']] = 0, 0, 0

            #     return data
    
            print("STEP 6/7/8")
            # combined_data = var38_percentile_features(combined_data)
            if use_test:
                combined_data["ecdf_0"] = 0
                combined_data["ecdf_1"] = 0
                combined_data["ecdf_2"] = 0
    
                for diff in [0,1,2]:
                    for i in combined_data.var15.unique():
                        ranked = combined_data.var38[combined_data.var15.isin([i-diff,i,i+diff])].rank()
                        combined_data.loc[combined_data.var15.isin([i-diff,i,i+diff]), f"ecdf_{diff}"] = (ranked/ranked.max()).values
            else:
                X_train["ecdf_0"] = 0
                X_train["ecdf_1"] = 0
                X_train["ecdf_2"] = 0
                X_test["ecdf_0"] = 0
                X_test["ecdf_1"] = 0
                X_test["ecdf_2"] = 0
    
                for diff in [0,1,2]:
                    for i in X_train.var15.unique():
                        # print(i)
                        ranked = X_train.var38[X_train.var15.isin([i-diff,i,i+diff])].rank()
                        ranked = (ranked/ranked.max()).values
                        X_train.loc[X_train.var15.isin([i-diff,i,i+diff]), f"ecdf_{diff}"] = ranked
                        
                        train_var38 = X_train.var38[X_train.var15.isin([i-diff,i,i+diff])]
                        test_var38 = X_test.var38[X_test.var15.isin([i-diff,i,i+diff])]
                        rank_map = dict(zip(train_var38,ranked))
                        
                        closest = [rank_map[np.unique(train_var38)[np.abs(np.unique(train_var38)-t).argmin()]] for t in np.unique(test_var38)]
                        test_rank_map = dict(zip(np.unique(test_var38),closest))
                        X_test.loc[X_test.var15.isin([i-diff,i,i+diff]), f"ecdf_{diff}"] = test_var38.apply(lambda x: test_rank_map[x])
                
                combined_data[["ecdf_0","ecdf_1","ecdf_2"]] = pd.concat([X_train[["ecdf_0","ecdf_1","ecdf_2"]],X_test[["ecdf_0","ecdf_1","ecdf_2"]]])
                

            # print("Step 6")
            # ecdf = combined_data.reset_index()[['ID', 'var15', 'var38']].dropna(subset=['var38'])
            # ecdf['rank'] = ecdf.groupby('var15')['var38'].rank(method='dense')
            # combined_data['ecdf1'] = ecdf.groupby('var15')['rank'].transform(lambda x: x / x.max())
            
            # # # 9/10 Calculate ult_1/ult_3 and hace2/hace3 ratio
            def ratio_features(data,index):
                ratio_tuples = [
                    ('imp_op_var39_comer_ult_ratio', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3'),
                    ('imp_op_var41_comer_ult_ratio', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3'),
                    ('imp_op_var39_efect_ult1_ratio', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3'),
                    ('imp_op_var41_efect_ult1_ratio', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3'),
                    ('num_op_var39_comer_ult_ratio', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3'),
                    ('num_op_var41_comer_ult_ratio', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3'),
                    ('num_op_var39_efect_ult1_ratio', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3'),
                    ('num_op_var41_efect_ult1_ratio', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3'),
                    ('num_op_var39_ult_ratio', 'num_op_var39_ult1', 'num_op_var39_ult3'),
                    ('num_op_var41_ult_ratio', 'num_op_var41_ult1', 'num_op_var41_ult3'),
                    ('num_var22_ult_ratio', 'num_var22_ult1', 'num_var22_ult3'),
                    ('num_var45_ult_ratio', 'num_var45_ult1', 'num_var45_ult3'),
                    ('saldo_medio_var5_ult_ratio', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3'),
                    ('num_var22_hace_ratio', 'num_var22_hace2', 'num_var22_hace3'),
                    ('num_var45_hace_ratio', 'num_var45_hace2', 'num_var45_hace3'),
                    ('saldo_medio_var5_hace_ratio', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3')
                ]
                new_cols = []
                for new, col1, col2 in ratio_tuples:
                    new_col = data[col1] / data[col2]
                    new_col.index = index
                    new_col.name = new
                    
                    new_col[new_col==np.inf] = np.nan
                    new_col[new_col==-np.inf] = np.nan
                    new_col = new_col.fillna(0.)
                    new_cols.append(new_col)
                
                return new_cols
            
            print("STEP 9/10")
            new_cols = ratio_features(pd.concat([X_train_with_y, X_test_with_y]), index = combined_data.index)
            combined_data = pd.concat([combined_data]+new_cols,axis=1)
            
            # # 11. Calculate mod 3 == 0 for each money variable
            def money_mod3_feature(data):
                relevant_cols = [
                        'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 
                        'imp_op_var40_comer_ult1', 'imp_op_var40_comer_ult3', 'imp_op_var40_ult1', 
                        'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 
                        'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 
                        'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 
                        'saldo_var1', 'saldo_var5', 'saldo_var8', 'saldo_var12', 'saldo_var13_corto', 
                        'saldo_var13', 'saldo_var14', 'saldo_var24', 'saldo_var25', 'saldo_var26', 
                        'saldo_var30', 'saldo_var31', 'saldo_var37', 'saldo_var40', 'saldo_var42', 
                        'imp_aport_var13_hace3', 'imp_aport_var13_ult1', 'imp_var7_recib_ult1', 
                        'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'saldo_medio_var5_hace2', 
                        'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 
                        'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 
                        'saldo_medio_var8_ult3', 'saldo_medio_var12_hace2', 'saldo_medio_var12_hace3', 
                        'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 
                        'saldo_medio_var13_corto_hace3', 'saldo_medio_var13_corto_ult1', 'saldo_medio_var13_corto_ult3'
                    ]


                mod_feats = pd.DataFrame(((data[relevant_cols] % 3)==0)*1)
                mod_feats.columns = [i+"_mod3" for i in relevant_cols]
                mod_feats.index = data.index    
                return mod_feats
            
            print("STEP 11")
            mod_feats = money_mod3_feature(combined_data.copy())
            combined_data = pd.concat([combined_data, mod_feats],axis=1)

            
            # 12. Population split
            # def create_zero_segment(data):
            #     # Define columns to exclude from calculation
            #     exclude_cols = ['ID', 'TARGET', 'idx', 'var3', 'var15', 'var36', 'var38',
            #                     'num_var4', 'num_var1_0', 'num_var2_0', 'num_var5_0', 'num_var6_0',
            #                     'num_var8_0', 'num_var13_corto_0', 'num_var13_medio_0', 
            #                     'num_var13_largo_0', 'num_var14_0', 'num_var18_0', 'num_var20_0',
            #                     'num_var22_hace2', 'num_var22_hace3', 'num_med_var22_ult3', 
            #                     'num_var22_ult1', 'num_var22_ult3', 'num_var24_0', 'num_var25_0',
            #                     'num_var26_0', 'num_var27_0', 'num_var28_0', 'num_var29_0', 
            #                     'num_var30_0', 'num_var32_0', 'num_var35_0', 'num_var37_0',
            #                     'num_var39_0', 'num_meses_var39_vig_ult3', 'num_var40_0',
            #                     'num_var41_0', 'num_var42_0', 'num_var44_0', 'num_var45_hace2',
            #                     'num_var45_hace3', 'num_med_var45_ult3', 'num_var45_ult1',
            #                     'num_var45_ult3', 'num_var46_0']
    
            #     # Calculate the sum excluding specified columns
            #     data['zeroSegment'] = data[list(set(data.columns) - set(exclude_cols))].fillna(0).sum()
    
            #     # Set specific conditions to 1 in zeroSegment
            #     data.loc[data['zeroSegment'] != 0, 'zeroSegment'] = 1
            #     data.loc[(data['var36'] != 99) & (data['zeroSegment'] == 0), 'zeroSegment'] = 1
            #     data.loc[(data['var15'] <= 22) & (data['zeroSegment'] == 0), 'zeroSegment'] = 1
            #     data.loc[(data['var3'] != 2) & (data['zeroSegment'] == 0), 'zeroSegment'] = 1
            #     data.loc[(data['num_var22_ult1'] != 0) & (data['zeroSegment'] == 0), 'zeroSegment'] = 1
            #     data.loc[(data['num_meses_var39_vig_ult3'] == 1) & (data['zeroSegment'] == 0), 'zeroSegment'] = 1
    
            #     return data
    
            # print("STEP 12")
            # combined_data = create_zero_segment(combined_data)
            pop_split = pd.read_csv("datasets/santander-customer-satisfaction/population_split.csv")
            combined_data = pd.merge(combined_data,pop_split,on="ID")



            # 3. Remove low-frequency variable-positive class combinations
            # def remove_low_freq_var_pos_class_combinations(data):
            #     # non-0 attributes have only 1 targets
            #     target1cols = []
            #     for col in set(data.columns) - {'s0'}:
            #         if data[data[col] != 0][self.y_col].sum(axis=0) == 1:
            #             target1cols.append(col)
    
            #     data['s1'] = data[target1cols].apply(lambda x: np.sum(np.where(x != 0, 1, 0)))
            #     data = data.drop(columns=target1cols)
    
            #     # non-0 attributes have only 2 targets
            #     target2cols = []
            #     for col in set(data.columns) - {'s0', 's1'}:
            #         if data[data[col] != 0][self.y_col].sum(axis=0) == 2:
            #             target2cols.append(col)
    
            #     data['s2'] = data[target2cols].apply(lambda x: np.sum(np.where(x != 0, 1, 0)))
            #     data = data.drop(columns=target2cols)
    
            #     # non-0 attributes have only 3 targets
            #     target3cols = []
            #     for col in set(data.columns) - {'s0', 's1', 's2'}:
            #         if data[data[col] != 0][self.y_col].sum(axis=0) == 3:
            #             target3cols.append(col)
    
            #     data['s3'] = data[target3cols].apply(lambda x: np.sum(np.where(x != 0, 1, 0)))
            #     data = data.drop(columns=target3cols)
    
            # Drop some cols
            print("STEP 1")
            remove_indicator_vars(combined_data)
            
            print("STEP 2")
            remove_const_vars(combined_data)
            
            print("STEP 3")
            # Move drop at the end
            # remove_low_freq_var_pos_class_combinations(combined_data)
            usecols=combined_data.columns.drop("TARGET")
            dropcols = usecols[[combined_data["TARGET"][combined_data[col]!=0].sum()<4 for col in usecols]]
            combined_data = combined_data.drop(dropcols,axis=1)            


            # Apply log-transform
            # combined_data = np.log(combined_data)
        
            # Separate train and test again
            X_train = combined_data[~combined_data[self.y_col].isna()].drop(self.y_col, axis=1)
            X_test = combined_data[combined_data[self.y_col].isna()].drop(self.y_col, axis=1)

            X_train = pd.merge(X_train,X_train_kmeans,on = "ID")
            X_test = pd.merge(X_test,X_test_kmeans,on = "ID") 
            
            # X_train = pd.merge(X_train,X_train_tsne, on = "ID")
            # X_test = pd.merge(X_test,X_test_tsne, on = "ID") 

            ### From Dmitry solution: Encode saldo_var13 - would need to include in CV loop
            # loo = LeaveOneOutEncoder()
            # X_train['saldo_var13_loo'] = loo.fit_transform(X_train['saldo_var13'].astype(str),y_train)['saldo_var13'].values
            # X_test['saldo_var13_loo'] = loo.transform(X_test['saldo_var13'].astype(str))['saldo_var13'].values            

            # X_train = pd.concat([X_train,X_train_pca],axis=1)
            # X_test = pd.concat([X_test,X_test_pca],axis=1)
            
            self.X_train, self.X_test, self.y_train = X_train, X_test, y_train

            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True) 
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))


        
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    def get_cv_folds(self, X_train, y_train, seed=42):
        ss = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        folds = []
        for num, (train,test) in enumerate(ss.split(X_train.copy(), y_train.copy())):
            folds.append([train, test])  

        return folds
    
    def expert_postprocessing(self, X_train, y, test=True, **kwargs):
        if test:
            X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
            y[X_test["num_aport_var13_hace3"] >= 6] = 0
            y[X_test["num_meses_var13_largo_ult3"] >= 1] = 0
            y[X_test["var15"] < 23] = 0
            y[X_test["var36"] == 0] = 0
        
        return y

      
    
################################################################
################################################################
################################################################
    
class PortoSeguroDriverDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "porto-seguro-safe-driver-prediction"

        self.task_type = "binary"
        self.eval_metric_name = "gini"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [1,  4, 21, 24, 26, 29, 30, 31]            
        self.y_col = "target"
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', index_col=0, engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', index_col=0, engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        y_train = data[self.y_col]
        X_train = data.drop(self.y_col,axis=1)  

        X_train[X_train==-1] = np.nan
        X_test[X_test==-1] = np.nan
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    # based on 2nd best Kaggle solution: https://github.com/xiaozhouwang/kaggle-porto-seguro/
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, neural_net=False, **kwargs):
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}_{"nn" if neural_net else "tree"}.pickle') or overwrite_existing:
            if neural_net:
                print(f"Apply expert preprocessing for neural nets")
            else:
                print(f"Apply expert preprocessing for tree-based models")

            train_id = X_train.index
            test_id = X_test.index

            # remove calculated features
            def remove_calc_features(data):
                calc_feats = [col for col in data.columns if 'calc' in col]
                data = data.drop(calc_feats, axis=1)
                return data

            X_train = remove_calc_features(X_train)
            X_test = remove_calc_features(X_test)
        
            cat_fea = [x for x in X_train.columns if 'cat' in x]
    
            # basic feature block
            X_train = X_train.replace(np.nan, -1)
            X_test = X_test.replace(np.nan, -1)
    
            # missing features
            X_train['missing'] = (X_train==-1).sum(axis=1).astype(float)
            X_test['missing'] = (X_test==-1).sum(axis=1).astype(float)
    
            # string cat count feature
            def string_label_attributes(substr, df):
                colset = [c for c in df.columns if substr in c]
                df['new_'+substr] = df.apply(lambda row: '_'.join(str(row[col]) for col in colset), axis=1)
    
            string_label_attributes('ind', X_train)
            string_label_attributes('ind', X_test)
    
            # cat count features
            cat_count_features = []
            for c in cat_fea + ['new_ind']:
                if use_test:
                    d = pd.concat([X_train[c],X_test[c]]).value_counts().to_dict()
                else:
                    d = X_train[c].value_counts().to_dict()
                    
                X_train['%s_count'%c] = X_train[c].apply(lambda x:d.get(x,0))
                X_test['%s_count'%c] = X_test[c].apply(lambda x:d.get(x,0))
                cat_count_features.append('%s_count'%c)
    
            X_train = X_train.drop(['new_ind'], axis=1)
            X_test = X_test.drop(['new_ind'], axis=1)

            # One-hot encode cat features for tree based methods
            if not neural_net:
                feature_names = X_train.columns.tolist()
                cat_features = [c for c in feature_names if ('cat' in c and 'count' not in c)]
                for c in cat_features:
                    le = LabelEncoder()
                    le.fit(X_train[c])
                    X_train[c] = le.transform(X_train[c])
                    X_test[c] = le.transform(X_test[c])
                
                enc = OneHotEncoder()
                enc.fit(X_train[cat_features])
                X_train_cat = enc.transform(X_train[cat_features])
                X_train_cat = pd.DataFrame(X_train_cat.toarray(), columns=enc.get_feature_names_out(cat_features)).set_index(X_train.index)
                X_train = pd.concat([X_train, X_train_cat], axis=1)

                X_test_cat = enc.transform(X_test[cat_features])
                X_test_cat = pd.DataFrame(X_test_cat.toarray(), columns=enc.get_feature_names_out(cat_features)).set_index(X_test.index)
                X_test = pd.concat([X_test, X_test_cat], axis=1)

                X_train = X_train.drop(cat_features, axis=1)
                X_test = X_test.drop(cat_features, axis=1)

                self.cat_indices = []

            if neural_net:
                # interaction features
                def interaction_features(train, test, fea1, fea2, prefix):
                    # Divide by small number to prevent division by 0 (maybe change approach later)
                    train['inter_{}*'.format(prefix)] = train[fea1] * train[fea2]
                    train['inter_{}/'.format(prefix)] = np.where(train[fea2] == 0, 0, train[fea1] / train[fea2])
                
                    test['inter_{}*'.format(prefix)] = test[fea1] * test[fea2]
                    test['inter_{}/'.format(prefix)] = np.where(test[fea2] == 0, 0, test[fea1] / test[fea2])
                
                    return train, test
                
                for e, (x, y) in enumerate(combinations(['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01'], 2)):
                    X_train, X_test = interaction_features(X_train, X_test, x, y, e)
        
                # XGBoost features
                def xgb_features(X_train, X_test, y_train):
                    #print(X_train.columns)
                    train_label = y_train
                    train_id = X_train.index
                    #del X_train["id"]
        
                    test_id = X_test.index
                    #del X_test["id"]
        
                    params = {
                        "objective": "reg:linear",
                        "booster": "gbtree",
                        "eta": 0.1,
                        "max_depth": int(6),
                        "subsample": 0.9,
                        "colsample_bytree": 0.85,
                        "min_child_weight": 55,
                        # Add GPU settings by default
                        "device": "cuda",
                        "tree_method": "hist"
                    }
        
                    num_boost_round = 500
        
                    if use_test:
                        data = pd.concat([X_train, X_test])
                    else:
                        data = X_train
                    train_rows = X_train.shape[0]
                    
                    data.reset_index(inplace=True,drop=True)
                    
                    feature_results = []
                    feature_results_test = []
                            
                    for target_g in ['car', 'ind', 'reg']:
                        features = [x for x in list(data) if target_g not in x]
                        target_list = [x for x in list(data) if target_g in x]
                        train_fea = data[features]
                        for target in target_list:
                            print(target)
                            train_label = data[target]
                            kfold = KFold(n_splits=5, random_state=218, shuffle=True)
                            kf = kfold.split(data)
                            cv_train = np.zeros(shape=(data.shape[0], 1))
                            if not use_test:
                                cv_test = np.zeros(shape=(X_test.shape[0], 1))
                                
                            for i, (train_fold, validate) in enumerate(kf):
                                X_train, X_validate, label_train, label_validate = \
                                    train_fea.iloc[train_fold], train_fea.iloc[validate], train_label[train_fold], train_label[validate]
                                dtrain = xgb.DMatrix(X_train, label_train)
                                dvalid = xgb.DMatrix(X_validate, label_validate)
                                watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
                                bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=50,
                                                early_stopping_rounds=10)
                                cv_train[validate, 0] += bst.predict(xgb.DMatrix(X_validate))

                                if not use_test:
                                    ### TODO
                                    cv_test[:, 0] += bst.predict(xgb.DMatrix(X_test[features]))/5
                            feature_results.append(cv_train)
                            if not use_test:
                                feature_results_test.append(cv_test)
                            # print(cv_train.mean(),cv_test.mean())
                    
                    if use_test:
                        feature_results = np.hstack(feature_results)
                        train_features = feature_results[:train_rows, :]
                        test_features = feature_results[train_rows:, :]
                    else:
                        feature_results = np.hstack(feature_results)
                        feature_results_test = np.hstack(feature_results_test)
                        train_features = feature_results
                        test_features = feature_results_test
                        
                    # pickle.dump([train_features, test_features], open(f'./datasets/porto-seguro-safe-driver-prediction/processed/xgb_features_{dataset_version}.pickle', 'wb'))
        
                    return [train_features, test_features]
        
                # if os.path.exists(f'./datasets/porto-seguro-safe-driver-prediction/processed/xgb_features_{dataset_version}.pickle'):
                #     [train_feat, test_feat] = pickle.load(open(f'./datasets/porto-seguro-safe-driver-prediction/processed/xgb_features_{dataset_version}.pickle', 'rb'))
                # else:
                train_feat, test_feat = xgb_features(X_train.copy(), X_test.copy(), y_train.copy())
        
                xgb_cols = ['xgb_'+str(i) for i in range(len(train_feat[0]))]
                X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(train_feat, columns=xgb_cols)], axis=1)
                X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(test_feat, columns=xgb_cols)], axis=1)
        
                
                # feature aggregation
                def proj_num_on_cat(train_df, test_df, target_column, group_column):
                    """
                    :param train_df: train data frame
                    :param test_df:  test data frame
                    :param target_column: name of numerical feature
                    :param group_column: name of categorical feature
                    """

                    train_df['row_id'] = range(train_df.shape[0])
                    test_df['row_id'] = range(test_df.shape[0])
                    train_df['train'] = 1
                    test_df['train'] = 0
        
                    if use_test:
                        all_df = pd.concat([train_df[['row_id', 'train', target_column, group_column]], test_df[['row_id','train', target_column, group_column]]])
                    else:
                        all_df = train_df[['row_id', 'train', target_column, group_column]]
                        
                    grouped = all_df[[target_column, group_column]].groupby(group_column)
                    the_size = pd.DataFrame(grouped.size()).reset_index()
                    the_size.columns = [group_column, '%s_size' % target_column]
                    the_mean = pd.DataFrame(grouped.mean()).reset_index()
                    the_mean.columns = [group_column, '%s_mean' % target_column]
                    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
                    the_std.columns = [group_column, '%s_std' % target_column]
                    the_median = pd.DataFrame(grouped.median()).reset_index()
                    the_median.columns = [group_column, '%s_median' % target_column]
                    the_stats = pd.merge(the_size, the_mean)
                    the_stats = pd.merge(the_stats, the_std)
                    the_stats = pd.merge(the_stats, the_median)
                
                    the_max = pd.DataFrame(grouped.max()).reset_index()
                    the_max.columns = [group_column, '%s_max' % target_column]
                    the_min = pd.DataFrame(grouped.min()).reset_index()
                    the_min.columns = [group_column, '%s_min' % target_column]
                
                    the_stats = pd.merge(the_stats, the_max)
                    the_stats = pd.merge(the_stats, the_min)
                
                    if use_test:
                        all_df = pd.merge(all_df, the_stats, how='left')
                        selected_train = all_df.copy()[all_df['train'] == 1]
                        selected_test = all_df.copy()[all_df['train'] == 0]
                    else:
                        selected_train = pd.merge(train_df[['row_id', 'train', target_column, group_column]], the_stats, how='left').copy()
                        selected_test = pd.merge(test_df[['row_id', target_column, group_column]], the_stats, how='left').copy()
                        
                    selected_train.sort_values('row_id', inplace=True)
                    selected_test.sort_values('row_id', inplace=True)
                    selected_train.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
                    if use_test:
                        selected_test.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
                    else:
                        selected_test.drop([target_column, group_column, 'row_id'], axis=1, inplace=True)
                
                    selected_train, selected_test = np.array(selected_train), np.array(selected_test)
                    print(selected_train.shape, selected_test.shape)
                    return selected_train, selected_test
                
                agg_train = []
                agg_test = []
        
                for t in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01']:
                    for g in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat']:
                        if t != g:
                            print(t,g)
                            s_train, s_test = proj_num_on_cat(X_train, X_test, target_column=t, group_column=g)
                            agg_train = s_train if len(agg_train) == 0 else np.concatenate([agg_train, s_train], axis=1)
                            agg_test = s_test if len(agg_test) == 0 else np.concatenate([agg_test, s_test], axis=1)
        
                print(X_train.shape, X_test.shape)

                agg_cols = ['agg_'+str(i) for i in range(len(agg_train[0]))]
                X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(agg_train, columns=agg_cols)], axis=1)
                X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(agg_test, columns=agg_cols)], axis=1)

                # update cat indices after df modification
                self.cat_indices = [i for i, col in enumerate(X_train.columns) if 'cat' in col]
    
            print(X_train.shape, X_test.shape)
            
            X_train = X_train.set_index(train_id)
            X_test = X_test.set_index(test_id)
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True) 
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'wb'))
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'rb'))
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}_{"nn" if neural_net else "tree"}.pickle', 'rb'))
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train

    # def get_cv_folds(self, X_train, y_train, seed=42):
    #     ### !! Currently not original implemented - original solution used 30-fold CV - but also dicusses 5-fold
    #     ss = KFold(n_splits=5, random_state=seed, shuffle=True)
    #     folds = []
    #     for num, (train,test) in enumerate(ss.split(y_train.copy(), y_train.copy())):
    #         folds.append([train, test])    
    #     return folds
        
    def get_cv_folds(self, X_train, y_train, seed=42):
        ss = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        folds = []
        for num, (train,test) in enumerate(ss.split(X_train.copy(), y_train.copy())):
            folds.append([train, test])  

        return folds


################################################################
################################################################
################################################################

class SberbankHousingDataset(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "sberbank-russian-housing-market"

        self.task_type = "regression"
        self.eval_metric_name = "rmsle"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [1, 11, 12, 29, 33, 34, 35, 36, 37, 38, 39, 40, 106, 114, 118, 152]           
        self.y_col = "price_doc"
        self.large_dataset = False
        # self.heavy_tailed = True

        self.expert_postprocessing = False

    def load_data(self):
        data_train = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        
        y_train = data_train[self.y_col]
        data_train = data_train.drop([self.y_col], axis=1)
        X_train = data_train

        X_train[X_train==-1] = np.nan
        X_test[X_test==-1] = np.nan

        # create date related variable
        def create_date_var(data, time):
            data[time] = pd.to_datetime(data[time])
            data['year'] = data[time].dt.year 
            data['month'] = data[time].dt.month 
            data['day'] = data[time].dt.day 
            data['dayofweek'] = data[time].dt.dayofweek 
            data['days_in_month'] = data[time].dt.days_in_month 
            data['weekofyear'] = data[time].dt.days_in_month
            data['quarter'] = data[time].dt.quarter
            data['week'] = data[time].dt.isocalendar().week

        create_date_var(X_train, 'timestamp')
        X_train['timestamp_1'] = X_train.timestamp.apply(lambda x: x - pd.DateOffset(months=1))
        X_train['month_1'] = X_train['timestamp_1'].dt.month
        X_train['year_1'] = X_train['timestamp_1'].dt.year
        X_train['quarter_1'] = X_train['timestamp_1'].dt.quarter
        X_train['week_1'] = X_train['timestamp_1'].dt.isocalendar().week
        X_train.drop(['timestamp_1'], axis=1, inplace=True)
        
        create_date_var(X_test, 'timestamp')
        X_test['timestamp_1'] = X_test.timestamp.apply(lambda x: x - pd.DateOffset(months=1))
        X_test['month_1'] = X_test['timestamp_1'].dt.month
        X_test['year_1'] = X_test['timestamp_1'].dt.year
        X_test['quarter_1'] = X_test['timestamp_1'].dt.quarter
        X_test['week_1'] = X_test['timestamp_1'].dt.isocalendar().week
        X_test.drop(["timestamp_1"], axis=1, inplace=True)


        #
        macro_df = pd.read_csv(f'./datasets/{self.dataset_name}/raw/macro.csv', engine="pyarrow")
        macro_df["oil_urals*gdp_quart_growth"] = macro_df["oil_urals"] * macro_df["gdp_quart_growth"]
        scaler = MinMaxScaler()
        for col in ["micex_rgbi_tr", "gdp_quart_growth", "oil_urals*gdp_quart_growth"]:
            macro_df[col] = scaler.fit_transform(macro_df[[col]])
            macro_df[col] = np.log1p(macro_df[col])
        macro_df = macro_df.filter(["timestamp", "micex_rgbi_tr", "gdp_quart_growth", "oil_urals*gdp_quart_growth"])
        create_date_var(macro_df, "timestamp")
        
        self.X_train, self.X_test, self.y_train, self.macro = X_train, X_test, y_train, macro_df

    # based on 9th best (private) Kaggle solution: https://www.kaggle.com/competitions/sberbank-russian-housing-market/discussion/35912
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            
            # create statistical variables
            def get_stats_target(df, group_column, target_column, drop_raw_col=False):
                df_old = df.copy()
                grouped = df_old.groupby(group_column)
                the_stats = grouped[target_column].agg(['mean','median','max','min','std']).reset_index()
                
                the_stats.columns = [group_column[0], 
                                '_%s_mean_by_%s' % (target_column[0], group_column[0]),
                                '_%s_median_by_%s' % (target_column[0], group_column[0]),
                                '_%s_max_by_%s' % (target_column[0], group_column[0]),
                                '_%s_min_by_%s' % (target_column[0], group_column[0]),
                                '_%s_std_by_%s' % (target_column[0], group_column[0])]
                
                df_old = pd.merge(left=df_old, right=the_stats, on=group_column, how='left')
                if drop_raw_col:
                    df_old.drop(group_column, axis=1, inplace=True)
                return df_old
    
            # big part of feature engineering
            def preprocess_data(rubbish_in, keep_is_missing=False):
                impute_missing = -1
                impute_missing_0 = 0
                impute_missing_1 = 1
                
                df_new = rubbish_in.copy()
                '''----------------------------------------------------------------------------------
                compute house count of last month, last week 
                ----------------------------------------------------------------------------------'''
                last_month_year = (df_new['month_1'] + df_new["year_1"] * 100)
                last_month_year_cnt_map = last_month_year.value_counts().to_dict()    
                df_new['_last_month_year_cnt'] = last_month_year.map(last_month_year_cnt_map)
                
                last_week_year = (df_new["week_1"] + df_new["year_1"] * 100)
                last_week_year_cnt_map = last_week_year.value_counts().to_dict()    
                df_new['_last_week_year_cnt'] = last_week_year.map(last_week_year_cnt_map)
                
                '''----------------------------------------------------------------------------------
                compute house count of this month, last week 
                ----------------------------------------------------------------------------------'''
                month_year = (df_new["month"] + df_new["year"] * 100)
                month_year_cnt_map = month_year.value_counts().to_dict()
                df_new['_month_year_cnt'] = month_year.map(month_year_cnt_map)
                
                week_year = (df_new["week"] + df_new["year"] * 100)
                week_year_cnt_map = week_year.value_counts().to_dict()
                df_new['_week_year_cnt'] = week_year.map(week_year_cnt_map)
    
                # drop useless variables
                df_new.drop(['month_1', 'year_1', 'quarter_1'], axis=1, inplace=True)
                df_new['_num_of_missing'] = df_new.isnull().sum(axis=1)
                
                if keep_is_missing: 
                    df_new['_missing_hospital_beds_raion'] = df_new['hospital_beds_raion'].isnull().astype(int)
                df_new['hospital_beds_raion'].fillna(impute_missing_0, inplace=True)
    
                if keep_is_missing: 
                    df_new['_missing_cafe_500_info'] = df_new['cafe_avg_price_500'].isnull().astype(int)
                df_new['cafe_avg_price_500'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_500_max_price_avg'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_500_min_price_avg'].fillna(impute_missing_0, inplace=True)
    
                if keep_is_missing:
                    df_new['_missing_cafe_1000_info'] = df_new['cafe_avg_price_1000'].isnull().astype(int)
                df_new['cafe_avg_price_1000'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_1000_max_price_avg'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_1000_min_price_avg'].fillna(impute_missing_0, inplace=True)
    
                if keep_is_missing:
                    df_new['_missing_cafe_1500_info'] = df_new['cafe_avg_price_1500'].isnull().astype(int)
                df_new['cafe_avg_price_1500'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_1500_max_price_avg'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_1500_min_price_avg'].fillna(impute_missing_0, inplace=True)
    
                if keep_is_missing:
                    df_new['_missing_cafe_2000_info'] = df_new['cafe_avg_price_2000'].isnull().astype(int)
                df_new['cafe_avg_price_2000'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_2000_max_price_avg'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_2000_min_price_avg'].fillna(impute_missing_0, inplace=True)
    
                if keep_is_missing:
                    df_new['_missing_cafe_3000_info'] = df_new['cafe_avg_price_3000'].isnull().astype(int)
                df_new['cafe_avg_price_3000'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_3000_max_price_avg'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_3000_min_price_avg'].fillna(impute_missing_0, inplace=True)
    
                if keep_is_missing:
                    df_new['_missing_cafe_5000_info'] = df_new['cafe_avg_price_5000'].isnull().astype(int)
                df_new['cafe_avg_price_5000'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_5000_max_price_avg'].fillna(impute_missing_0, inplace=True)
                df_new['cafe_sum_5000_min_price_avg'].fillna(impute_missing_0, inplace=True)
    
                df_new['preschool_quota'].fillna(impute_missing_0, inplace=True)
                df_new['school_quota'].fillna(impute_missing_0, inplace=True)
                
                if keep_is_missing:
                    df_new['missing_build_info'] = df_new['build_count_block'].isnull().astype(int)
                df_new['build_count_block'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_after_1995'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_before_1920'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_wood'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_mix'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_brick'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_foam'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_frame'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_1921-1945'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_monolith'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_panel'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_slag'].fillna(impute_missing_0, inplace=True)
                df_new['raion_build_count_with_material_info'].fillna(impute_missing_1, inplace=True)
                df_new['raion_build_count_with_builddate_info'].fillna(impute_missing_1, inplace=True)
                df_new['build_count_1946-1970'].fillna(impute_missing_0, inplace=True)
                df_new['build_count_1971-1995'].fillna(impute_missing_0, inplace=True)
    
            #     df_new['prom_part_5000'].fillna(df_new['prom_part_5000'].median(), inplace=True)
            #     df_new['metro_km_walk'].fillna(df_new['metro_km_walk'].median(), inplace=True)
            #     df_new['metro_min_walk'].fillna(df_new['metro_min_walk'].median(), inplace=True)
            #     df_new['id_railroad_station_walk'].fillna(df_new['id_railroad_station_walk'].median(), inplace=True)
            #     df_new['railroad_station_walk_min'].fillna(df_new['railroad_station_walk_min'].median(), inplace=True)
            #     df_new['railroad_station_walk_km'].fillna(df_new['railroad_station_walk_km'].median(), inplace=True)
            #     df_new['green_part_2000'].fillna(df_new['green_part_2000'].median(), inplace=True)
                df_new['product_type'].fillna(df_new['product_type'].mode()[0], inplace=True)
    
                '''----------------------------------------------------------------------------------
                    fill missing values in full_sq, life_sq, kitch_sq, num_room
                    use apartment strategy to fill the most likely values
                ----------------------------------------------------------------------------------'''
                df_new.loc[df_new.full_sq == 5326.0, 'full_sq'] = 53
                df_new.loc[df_new.full_sq < 10, 'full_sq'] = np.nan
                df_new.loc[df_new.life_sq == 2, 'life_sq'] = np.nan
                df_new.loc[df_new.life_sq == 7478.0, 'life_sq'] = 48
                
                df_new['_missing_num_room'] = df_new['num_room'].isnull().astype(int)
                df_new['_missing_kitch_sq'] = df_new['kitch_sq'].isnull().astype(int)
                df_new['_missing_material'] = df_new['material'].isnull().astype(int)
                df_new['_missing_max_floor'] = df_new['max_floor'].isnull().astype(int)
                df_new['_apartment_name']=df_new['sub_area'] + df_new['metro_km_avto'].apply(lambda x: np.round(x)).astype(str)
                df_new['_apartment_name_drop']=df_new['sub_area'] + df_new['metro_km_avto'].apply(lambda x: np.round(x)).astype(str)
                df_new.life_sq.\
                fillna(df_new.groupby(['_apartment_name_drop'])['life_sq'].transform("median"), inplace=True)
                df_new.full_sq.\
                fillna(df_new.groupby(['_apartment_name_drop'])['full_sq'].transform("median"), inplace=True)
                df_new.kitch_sq.\
                fillna(df_new.groupby(['_apartment_name_drop'])['kitch_sq'].transform("median"), inplace=True)
                df_new.num_room.\
                fillna(df_new.groupby(['_apartment_name_drop'])['num_room'].transform("median"), inplace=True)
                df_new.life_sq.\
                fillna(df_new.groupby(['sub_area'])['life_sq'].transform("median"), inplace=True)
                df_new.full_sq.\
                fillna(df_new.groupby(['sub_area'])['full_sq'].transform("median"), inplace=True)
                df_new.kitch_sq.\
                fillna(df_new.groupby(['sub_area'])['kitch_sq'].transform("median"), inplace=True)
                df_new.num_room.\
                fillna(df_new.groupby(['sub_area'])['num_room'].transform("median"), inplace=True)
                
                '''----------------------------------------------------------------------------------
                    fix wrong values
                ----------------------------------------------------------------------------------'''
                wrong_kitch_sq_index = df_new['kitch_sq'] > df_new['life_sq']
                df_new.loc[wrong_kitch_sq_index, 'kitch_sq'] = df_new.loc[wrong_kitch_sq_index, 'life_sq'] * 1 / 3
    
                wrong_life_sq_index = df_new['life_sq'] > df_new['full_sq']
                df_new.loc[wrong_life_sq_index, 'life_sq'] = df_new.loc[wrong_life_sq_index, 'full_sq'] * 3 / 5
                df_new.loc[wrong_life_sq_index, 'kitch_sq'] = df_new.loc[wrong_life_sq_index, 'full_sq'] * 1 / 5
                df_new.loc[df_new.life_sq.isnull(), 'life_sq'] = df_new.loc[df_new.life_sq.isnull(), 'full_sq'] * 3 / 5
    
                '''----------------------------------------------------------------------------------
                    others
                ----------------------------------------------------------------------------------'''
                df_new['_rel_kitch_sq'] = df_new['kitch_sq'] / df_new['full_sq'].astype(float)
                
                df_new['_room_size'] = (df_new['life_sq'] - df_new['kitch_sq']) / df_new.num_room
                df_new['_room_size'] = df_new['_room_size'].apply(lambda x: 0 if x > 50 else x)
                df_new['_room_size'].fillna(0, inplace=True)
                
                df_new['_life_proportion'] = df_new['life_sq'] / df_new['full_sq']
                df_new['_kitch_proportion'] = df_new['kitch_sq'] / df_new['full_sq']
                
                df_new['_other_sq'] = df_new['full_sq'] - df_new['life_sq']
                df_new['_other_sq'] = df_new['_other_sq'].apply(lambda x: 0 if x <0 else x)
                
                df_new['max_floor'].fillna(1, inplace=True)
                df_new['floor'].fillna(1, inplace=True)
                
                wrong_max_floor_index = ((df_new['max_floor'] - df_new['floor']).fillna(-1)) < 0
                df_new['max_floor'][wrong_max_floor_index] = df_new['floor'][wrong_max_floor_index]
                df_new['max_floor'].fillna(1, inplace=True)
                
                df_new['_floor_from_top'] = df_new['max_floor'] - df_new['floor']
                df_new['_floor_by_top'] = df_new['floor'] / df_new['max_floor']
    
                # Year
                df_new.loc[df_new['build_year'] == 2, 'build_year'] = np.nan
                df_new.loc[df_new['build_year'] == 3, 'build_year'] = np.nan
                df_new.loc[df_new['build_year'] == 20, 'build_year'] = np.nan
                df_new.loc[df_new['build_year'] == 71, 'build_year'] = np.nan
                df_new.loc[df_new['build_year'] == 215, 'build_year'] = np.nan
                df_new.loc[df_new['build_year'] == 4965, 'build_year'] = 1956
                
                if len(df_new.loc[df_new['build_year'] == 20052009]['id'].values)>0:
                    df_new.loc[df_new['id'] == (df_new.loc[df_new['build_year'] == 20052009]['id'].values[0]+1), 'build_year'] = 2009
                
                df_new.loc[df_new['build_year'] == 20052009, 'build_year'] = 2005
                df_new['_build_year_missing'] = df_new['build_year'].isnull().astype(int)
                df_new['build_year'].fillna(df_new.groupby(['sub_area', 'max_floor'])['build_year'].
                                        transform('median'), inplace=True)
                df_new['build_year'].fillna(df_new.groupby(['sub_area'])['build_year'].
                                        transform('median'), inplace=True)
                df_new['_age_of_house_before_sale'] = np.abs(df_new["year"] - df_new.build_year)
                df_new['_sale_before_build'] = ((df_new["year"] - df_new.build_year) < 0).astype(int)
                
                # State
                df_new['_missing_state'] = df_new['state'].isnull().astype(int)
                state_missing_map = {33:3, None:0}
                df_new['state'] = df_new.state.replace(state_missing_map)
    
                
                df_new.material.fillna(0, inplace=True)
    
                df_new = get_stats_target(df_new, ['sub_area'], ['max_floor'])
                df_new = get_stats_target(df_new, ['sub_area'], ['num_room'])
                df_new = get_stats_target(df_new, ['sub_area'], ['full_sq'])
                df_new = get_stats_target(df_new, ['sub_area'], ['life_sq'])
                df_new = get_stats_target(df_new, ['sub_area'], ['kitch_sq'])
                
                # 1m 2m 3m part
                df_new['_particular_1m_2m_3m_missing'] = df_new['_missing_num_room'] + df_new['_missing_kitch_sq'] \
                                                        + df_new['_missing_max_floor'] + df_new['_missing_material']
                
                sub_area_donot_contain_1m2m3m = ['Arbat',
                                                'Molzhaninovskoe',
                                                'Poselenie Filimonkovskoe',
                                                'Poselenie Kievskij',
                                                'Poselenie Mihajlovo-Jarcevskoe',
                                                'Poselenie Rjazanovskoe',
                                                'Poselenie Rogovskoe',
                                                'Poselenie Voronovskoe',
                                                'Vostochnoe']
    
                df_new['_particular_1m_2m_3m_sub_area'] = 1 - df_new.sub_area.isin(sub_area_donot_contain_1m2m3m).astype(int)
                df_new['_particular_1m_2m_3m_magic'] = df_new['_particular_1m_2m_3m_missing']*10 + df_new['_particular_1m_2m_3m_sub_area']
                df_new.drop(['_missing_num_room', '_missing_kitch_sq', '_missing_max_floor', '_missing_material','_particular_1m_2m_3m_sub_area'], axis=1, inplace=True)
    
                # create new feature
                # district
                df_new['_pop_density'] = df_new.raion_popul / df_new.area_m
                df_new['_hospital_bed_density'] = df_new.hospital_beds_raion / df_new.raion_popul
                df_new['_healthcare_centers_density'] = df_new.healthcare_centers_raion / df_new.raion_popul
                df_new['_shopping_centers_density'] = df_new.shopping_centers_raion / df_new.raion_popul
                df_new['_university_top_20_density'] = df_new.university_top_20_raion / df_new.raion_popul
                df_new['_sport_objects_density'] = df_new.sport_objects_raion / df_new.raion_popul
                df_new['_best_university_ratio'] = df_new.university_top_20_raion / (df_new.sport_objects_raion + 1)
                df_new['_good_bad_propotion'] = (df_new.sport_objects_raion + 1) / (df_new.additional_education_raion + 1)
                df_new['_num_schools'] = df_new.sport_objects_raion + df_new.additional_education_raion
                df_new['_schools_density'] = df_new._num_schools + df_new.raion_popul
                df_new['_additional_education_density'] = df_new.additional_education_raion / df_new.raion_popul
                
                df_new['_ratio_preschool'] = df_new.preschool_quota / df_new.children_preschool
                df_new['_seat_per_preschool_center'] = df_new.preschool_quota / df_new.preschool_education_centers_raion
                df_new['_seat_per_preschool_center'] = df_new['_seat_per_preschool_center'].apply(lambda x: df_new['_seat_per_preschool_center'].median() if x > 1e8 else x)
                
                df_new['_ratio_school'] = df_new.school_quota / df_new.children_school
                df_new['_seat_per_school_center'] = df_new.school_quota / df_new.school_education_centers_raion
                df_new['_seat_per_school_center'] = df_new['_seat_per_school_center'].apply(lambda x: df_new['_seat_per_preschool_center'].median() if x > 1e8 else x)
                
                
                df_new['_raion_top_20_school'] = df_new['school_education_centers_top_20_raion'] / df_new['school_education_centers_raion']
                df_new['_raion_top_20_school'].fillna(0, inplace=True)
                df_new['_maybe_magic'] =  df_new.product_type.apply(str) + '_' + df_new.id_metro.apply(str)
                df_new['_id_metro_line'] = df_new.id_metro.apply(lambda x: str(x)[0])
                
                df_new['_female_ratio'] = df_new.female_f / df_new.full_all
                df_new['_male_ratio'] = df_new.male_f / df_new.full_all
                df_new['_male_female_ratio_area'] = df_new.male_f / df_new.female_f
                df_new['_male_female_ratio_district'] = (df_new.young_male + df_new.work_male + df_new.ekder_male) /\
                                                        (df_new.young_female + df_new.work_female + df_new.ekder_female)
                
                df_new['_young_ratio'] = df_new.young_all / df_new.raion_popul
                df_new['_young_female_ratio'] = df_new.young_female / df_new.raion_popul
                df_new['_young_male_ratio'] = df_new.young_male / df_new.raion_popul
                
                df_new['_work_ratio'] = df_new.work_all / df_new.raion_popul
                df_new['_work_female_ratio'] = df_new.work_female / df_new.raion_popul
                df_new['_work_male_ratio'] = df_new.work_male / df_new.raion_popul
                
                df_new['_children_burden'] = df_new.young_all / df_new.work_all
                df_new['_ekder_ratio'] = df_new.ekder_all / df_new.raion_popul
                df_new['_ekder_female_ratio'] = df_new.ekder_female / df_new.raion_popul
                df_new['_ekder_male_ratio'] = df_new.ekder_male / df_new.raion_popul
                
                sale_dict = dict(df_new[df_new.build_year > 3].groupby(['sub_area'])['timestamp'].count())
                df_new['_on_sale_known_build_year_ratio'] = df_new.sub_area.apply(lambda x: sale_dict[x]) / df_new.raion_build_count_with_builddate_info
                
                df_new['_congestion_metro'] = df_new.metro_km_avto / df_new.metro_min_avto
                df_new['_congestion_metro'].fillna(df_new['_congestion_metro'].mean(), inplace=True)
                df_new['_congestion_railroad'] = df_new.railroad_station_avto_km / df_new.railroad_station_avto_min
                
                df_new['_big_road1_importance'] = df_new.groupby(['id_big_road1'])['big_road1_km'].transform('mean')
                df_new['_big_road2_importance'] = df_new.groupby(['id_big_road2'])['big_road2_km'].transform('mean')
                df_new['_bus_terminal_importance'] = df_new.groupby(['id_bus_terminal'])['bus_terminal_avto_km'].transform('mean')
                
                df_new['_square_per_office_500'] = df_new.office_sqm_500 / df_new.office_count_500
                df_new['_square_per_trc_500'] = df_new.trc_sqm_500 / df_new.trc_count_500
                df_new['_square_per_office_1000'] = df_new.office_sqm_1000 / df_new.office_count_1000
                df_new['_square_per_trc_1000'] = df_new.trc_sqm_1000 / df_new.trc_count_1000
                df_new['_square_per_office_1500'] = df_new.office_sqm_1500 / df_new.office_count_1500
                df_new['_square_per_trc_1500'] = df_new.trc_sqm_1500 / df_new.trc_count_1500
                df_new['_square_per_office_2000'] = df_new.office_sqm_2000 / df_new.office_count_2000
                df_new['_square_per_trc_2000'] = df_new.trc_sqm_2000 / df_new.trc_count_2000    
                df_new['_square_per_office_3000'] = df_new.office_sqm_3000 / df_new.office_count_3000
                df_new['_square_per_trc_3000'] = df_new.trc_sqm_3000 / df_new.trc_count_3000 
                df_new['_square_per_office_5000'] = df_new.office_sqm_5000 / df_new.office_count_5000
                df_new['_square_per_trc_5000'] = df_new.trc_sqm_5000 / df_new.trc_count_5000 
                
                df_new['_square_per_trc_5000'].fillna(0, inplace=True)
                df_new['_square_per_office_5000'].fillna(0, inplace=True)
                df_new['_square_per_trc_3000'].fillna(0, inplace=True)
                df_new['_square_per_office_3000'].fillna(0, inplace=True)
                df_new['_square_per_trc_2000'].fillna(0, inplace=True)
                df_new['_square_per_office_2000'].fillna(0, inplace=True)
                df_new['_square_per_trc_1500'].fillna(0, inplace=True)
                df_new['_square_per_office_1500'].fillna(0, inplace=True)
                df_new['_square_per_trc_1000'].fillna(0, inplace=True)
                df_new['_square_per_office_1000'].fillna(0, inplace=True)
                df_new['_square_per_trc_500'].fillna(0, inplace=True)
                df_new['_square_per_office_500'].fillna(0, inplace=True)
                
                
                df_new['_cafe_sum_500_diff'] = df_new.cafe_sum_500_max_price_avg - df_new.cafe_sum_500_min_price_avg
                # replace it with ordering number
                ecology_map = {"satisfactory":5, "excellent":4, "poor":3, "good":2, "no data":1}
                df_new.ecology = df_new.ecology.map(ecology_map)
                
                return df_new 

            self.expert_postprocessing = True

            X_train.columns = map(str.lower, X_train.columns)
            X_test.columns = map(str.lower, X_test.columns)
            y_train = y_train.apply(lambda x: 11111112 if x>111111111 else x)

            # magic number for Investment and OwnerOcupier
            invest_idx = X_train[X_train["product_type"] == "Investment"].index
            owner_idx = X_train[X_train["product_type"] == "OwnerOccupier"].index
            y_train.loc[invest_idx] = y_train.loc[invest_idx] * 1.05
            y_train.loc[owner_idx] = y_train.loc[owner_idx] * 0.9

            if use_test:
                num_test_rows = len(X_test)
                X = pd.concat([X_train, X_test]).reset_index(drop=True)
                X_preprocessed = preprocess_data(X)
                X_train = X_preprocessed.iloc[:-num_test_rows]
                X_test = X_preprocessed.iloc[-num_test_rows:]
            else:
                X_train = preprocess_data(X_train)
                X_test = preprocess_data(X_test)
    
            # Encode binary cat features as numeric
            for col in X_train.columns[X_train.nunique()==2]:
                if X_train[col].dtype in [str, "O", "category", "object"]:
                    le = LabelEncoder()
                    mode = X_train[col].mode()[0]
                    X_train[col] = le.fit_transform(X_train[col])
    
                    if len(X_test[col].unique())==2:
                        X_test[col] = le.transform(X_test[col])
                    else:
                        X_test[col] = X_test[col].fillna(mode)
                        X_test[col] = le.transform(X_test[col])

            dtype = pd.CategoricalDtype(categories=list(X_train["timestamp"].astype(str).fillna("nan").unique()))
            X_train["timestamp"] = X_train["timestamp"].astype(str).fillna("nan").astype(dtype)
            X_test["timestamp"] = X_test["timestamp"].astype(str).fillna("nan").astype(dtype)       
            
            self.cat_indices = np.where(np.logical_and(X_train.dtypes.apply(lambda x: x in ["object", "category"]), X_train.nunique()>2))[0].tolist()
            X_train.loc[:,X_train.dtypes=="UInt32"] = X_train.loc[:,X_train.dtypes=="UInt32"].astype(int)
            X_test.loc[:,X_test.dtypes=="UInt32"] = X_test.loc[:,X_test.dtypes=="UInt32"].astype(int)

            self.X_train, self.X_test, self.y_train = X_train, X_test, y_train
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True) 
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))

    def get_cv_folds(self, X_train, y_train, seed=42):
        ss = KFold(n_splits=5, random_state=seed, shuffle=True)
        folds = []

        for num, (train_idx, test_idx) in enumerate(ss.split(X_train.copy(), y_train.copy())):
            # train = X_train.loc[(X_train.index.isin(train_idx)) & (X_train.product_type == 1)].index
            # test = X_train.loc[(X_train.index.isin(test_idx)) & (X_train.product_type == 1)].index
            # folds.append([train, test])
            folds.append([train_idx, test_idx])

        if self.expert_preprocessing:
            # one-hot-encoding of product_type column differ between minimalistic and expert preprocessing
            OwnerOccupier = X_train[X_train["id"] == 29]["product_type"].values[0]

            for num, (train_idx, test_idx) in enumerate(ss.split(X_train.copy(), y_train.copy())):
                train = X_train.loc[(X_train.index.isin(train_idx)) & (X_train.product_type == OwnerOccupier)].index
                test = X_train.loc[(X_train.index.isin(test_idx)) & (X_train.product_type == OwnerOccupier)].index
                folds.append([train, test])

            return folds
    
    # def get_cv_folds(self, X_train, y_train, seed=42):
    #     ss = KFold(n_splits=5, random_state=seed, shuffle=True)
    #     folds = []

    #     # Ensemble-1: Trend-adjust model to simulate the magic number
    #     # 5 folds for each of the 4 models
    #     for _ in range(4):
    #         for num, (train,test) in enumerate(ss.split(X_train.copy(), y_train.copy())):
    #             folds.append([train, test])

    #     # Ensemble-2: Remove bad points to adjust the former model
    #     model = xgb.XGBRegressor(enable_categorical=True)
    #     model.fit(X_train, y_train)
    #     y_hat = model.predict(X_train)
    #     train = pd.concat([X_train, y_train], axis=1)
    #     train["residual"] = (abs(y_train - y_hat) / y_train) * 100
    #     train = train[train["residual"] > 50]
    #     train = train.drop(columns=["residual"])

    #     for num, (train,test) in enumerate(ss.split(train.drop(columns=["price_doc"]).copy(), train["price_doc"].copy())):
    #         folds.append([train, test])

    #     return folds

################################################################
################################################################
################################################################

class WalmartRecruitingTripType(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "walmart-recruiting-trip-type-classification"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "classification"
        self.eval_metric_name = "mlogloss"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [1,2,4,5]            
        self.y_col = "TripType"
        self.large_dataset = False
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented ()

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            
            
            self.cat_indices = list(np.where(X_train.dtypes=="category")[0])#[np.where(X_train.columns==i)[0][0] for i in cat_vars]
    
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    def get_cv_folds(self, X_train, y_train, seed=42):
        ss = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        folds = []
        for num, (train,test) in enumerate(ss.split(X_train.copy(), y_train.copy())):
            folds.append([train, test])  

        return folds

    # def get_cv_folds(self, X_train, y_train, seed=42):
    #     # Note from experts:
    #     # We will predict test.csv using GroupKFold with months as groups. 
    #     # The training data are the months December 2017, January 2018, February 2018, March 2018, April 2018, and May 2018. 
    #     # We refer to these months as 12, 13, 14, 15, 16, 17. 
    #     # Fold one in GroupKFold will train on months 13 thru 17 and predict month 12. 
    #     # Note that the only purpose of month 12 is to tell XGB when to early_stop we don't actual care about the backwards time predictions. 
    #     # The model trained on months 13 thru 17 will also predict test.csv which is forward in time.        
        
    #     skf = StratifiedGroupKFold(n_splits=10)
    #     folds = []
    #     for num, (train,test) in enumerate(skf.split(X_train.copy(), y_train.copy(), groups=X_train.VisitNumber)):
    #         folds.append([train, test])    
            
    #     return folds        

    def pred_to_submission(self, y_pred):
        submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sample_submission.csv", engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        
        df_visit_pred = pd.concat([X_test["VisitNumber"],pd.DataFrame(y_pred,index=X_test.index,columns=submission.columns[1:])],axis=1)
        submission[submission.columns[1:]] = df_visit_pred.groupby("VisitNumber").mean().values
        

        return submission

##########################################################   
##########################################################   
##########################################################        
        
class AllstateClaimsSeverity(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "allstate-claims-severity"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "regression" # "binary", "classification"
        self.eval_metric_name = "mae"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [ 73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85, 86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]          
        # cat_cols = data.columns[np.logical_and(data.dtypes=="O",data.nunique()>2)]
        # np.where(np.logical_and(data.dtypes=="O",data.nunique()>2))[0].tolist()
        self.y_col = "loss"
        self.large_dataset = False
        self.heavy_tailed = True # (actually is true)
    
    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', nrows=188318)
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', nrows=125546)
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        y_train = data[self.y_col]
        X_train = data.drop(self.y_col,axis=1)    
        
        # self.cat_indices = [num for num, col in enumerate(X_train.columns) if col.startswith('cat')]

        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, neural_net=False, **kwargs):
        '''
        Solution implemented based on multiple solution descriptions as no single solution was reproducible well enough to achieve a high rank.  Utilized solutions:
        - https://www.kaggle.com/code/nitink12/prog-pyth2
        - https://www.kaggle.com/competitions/allstate-claims-severity/discussion/26414
        - 
        
        was sufficient enough in https://www.kaggle.com/competitions/allstate-claims-severity/discussion/26427 and the provided code in https://github.com/alno/kaggle-allstate-claims-severity

        '''
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
    
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            y_train = np.log1p(y_train)

            # 1. Binary encode categorical features
            binary_columns = [f'cat{i}' for i in range(1, 73)]
            for col in binary_columns:
                X_train[col] = (X_train[col] == 'B').astype(int)
                X_test[col] = (X_test[col] == 'B').astype(int)

            
            # prepare basic
            
            # prepare numeric-boxcox
            from scipy.stats import skew, boxcox, rankdata
            from scipy.special import erfinv
            
            cat_col_inds = X_train.columns[self.cat_indices]
            num_col_inds = [num for num, col in enumerate(X_train.columns) if col.startswith('cont')]


            multi_level_columns = [f'cat{i}' for i in range(73, 117)]
            # # 5. -8. Log(Loss) Statistics Per Level
            # log_loss = y_train
            
            # for col in multi_level_columns:
            #     group_stats = pd.concat([X_train,y_train]).groupby(col)['loss'].agg(['min', 'max', 'mean', 'std']).reset_index().rename(columns={'min': col + '_min_log_loss', 'max': col + '_max_log_loss', 'mean': col + '_mean_log_loss', 'std': col + '_std_log_loss'})
            #     X_train = X_train.merge(group_stats, on=col, how='left')
            #     X_test = X_test.merge(group_stats, on=col, how='left')

            # 10. Mean Log(Loss) Per Level Summary Stats
            # Aggregate summary statistics for mean log(loss) of all multi-level categorical columns
            # stats_features = [col + '_mean_log_loss' for col in multi_level_columns]
            # X_train['mean_log_loss_summary'] = X_train[stats_features].mean(axis=1)
            # X_test['mean_log_loss_summary'] = X_test[stats_features].mean(axis=1)

            if use_test:
                from sklearn import preprocessing
                train_test = pd.concat([X_train,X_test],axis=0)
                train_test["cont15"] = train_test["cont1"] + train_test["cont2"] + train_test["cont3"] + train_test["cont4"] + train_test["cont5"] + train_test["cont6"] + train_test["cont7"] + train_test["cont8"] + train_test["cont9"] + train_test["cont10"] + train_test["cont11"] + train_test["cont12"] + train_test["cont13"] + train_test["cont14"]
                train_test["cont15"] = train_test["cont15"]/14
                
                train_test["cont16"] = train_test["cont1"] * train_test["cont2"] * train_test["cont3"] * train_test["cont4"] * train_test["cont5"] * train_test["cont6"] * train_test["cont7"] * train_test["cont8"] * train_test["cont9"] * train_test["cont10"] * train_test["cont11"] * train_test["cont12"] * train_test["cont13"] * train_test["cont14"]
                train_test["cont16"] = train_test["cont15"]**(1/14)
                
                train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
                train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
                train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
                train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
                train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
                train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
                train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))
                
                train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"])+0000.1)
                train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"])+0000.1)
                train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"])+0000.1)
                train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"])+0000.1)
                train_test["cont14"]=(np.maximum(train_test["cont14"]-0.179722,0)/0.665122)**0.25

            X_train = train_test.iloc[:X_train.shape[0]]
            X_test = train_test.iloc[X_train.shape[0]:]

            # 12. Multi-Level Categorical Probabilities of Occurrence
            # Calculate the probability of occurrence for each level in multi-level categorical columns
            for col in multi_level_columns:
                prob_occ = X_train[col].value_counts(normalize=True)
                X_train[col + '_prob'] = X_train[col].map(prob_occ)
                X_test[col + '_prob'] = X_test[col].map(prob_occ)

            # 15. Base Numeric Features Summary Stats
            numeric_columns = X_train.columns[num_col_inds]
            X_train['numeric_mean'] = X_train[numeric_columns].mean(axis=1)
            X_train['numeric_std'] = X_train[numeric_columns].std(axis=1)
            X_train['numeric_min'] = X_train[numeric_columns].min(axis=1)
            X_train['numeric_max'] = X_train[numeric_columns].max(axis=1)

            X_test['numeric_mean'] = X_test[numeric_columns].mean(axis=1)
            X_test['numeric_std'] = X_test[numeric_columns].std(axis=1)
            X_test['numeric_min'] = X_test[numeric_columns].min(axis=1)
            X_test['numeric_max'] = X_test[numeric_columns].max(axis=1)

            # 16 & 17. Skew-Corrected, Numeric Columns
            # Correct skew in numeric columns
            for col in numeric_columns:
                if skew(X_train[col]) > 0.75:
                    X_train[col + '_skew_corrected'] = np.log1p(X_train[col])
                    X_test[col + '_skew_corrected'] = np.log1p(X_test[col])
                else:
                    X_train[col + '_skew_corrected'] = X_train[col]        
                    X_test[col + '_skew_corrected'] = X_test[col]

            # 18 & 19. Square Root and Log Transformed Numeric Columns
            # Square root transformation
            for col in numeric_columns:
                X_train[col + '_sqrt'] = np.sqrt(X_train[col])
                X_test[col + '_sqrt'] = np.sqrt(X_test[col])
            
            # Log transformation
            for col in numeric_columns:
                X_train[col + '_log'] = np.log1p(X_train[col]+0.1)
                X_test[col + '_log'] = np.log1p(X_test[col]+0.1)
               
            # for col in X_train.columns[num_col_inds]:
            #     # prepare numeric-boxcox
            #     if use_test:
            #         values = np.hstack((X_train.loc[:, col], X_test.loc[:, col]))
            #         sk = skew(values)
            #         if sk > 0.25:
            #             values_enc, lam = boxcox(values+1)
            #             X_train[col] = values_enc[:X_train.shape[0]]
            #             X_test[col] = values_enc[X_train.shape[0]:]
            #     else:
            #         values_train = X_train.loc[:, col].values
            #         values_test = X_test.loc[:, col].values
            #         sk = skew(values_train)
            #         if sk > 0.25:
            #             values_enc_train, lam = boxcox(values_train+1)
            #             X_train[col] = values_enc_train
                        
            #             values_enc_test, lam = boxcox(values_test+1, lmbda=lam)
            #             X_test[col] = values_enc_test
                    
                # prepare numeric-scaled
                # ss = StandardScaler()
                # if use_test:
                #     values = np.hstack((X_train.loc[:, col], X_test.loc[:, col]))
                #     scaled = ss.fit_transform(values.reshape(-1,1))
                #     X_train[col] = scaled[:X_train.shape[0]]
                #     X_test[col] = scaled[X_train.shape[0]:]
                # else:
                #     values_train = X_train.loc[:, col].values
                #     values_test = X_test.loc[:, col].values
                #     X_train[col] = ss.fit_transform(values_train.reshape(-1,1))
                #     X_test[col] = sss.transform(values_test.reshape(-1,1))
                
                # self.x_scaled = True
            
            
                        
            #     # prepare numeric-rank-norm
            #     if use_test:
            #         values = np.hstack((X_train.loc[:, col], X_test.loc[:, col]))
            #         scaled = ss.fit_transform(values.reshape(-1,1))
            #         X_train[col] = scaled[:X_train.shape[0]]
            #         X_test[col] = scaled[X_train.shape[0]:]
            #     else:
            #         values_train = X_train.loc[:, col].values
            #         values_test = X_test.loc[:, col].values
            #         X_train[col] = ss.fit_transform(values_train.reshape(-1,1))
            #         X_test[col] = ss.transform(values_test.reshape(-1,1))
            
            
            # for col in X_train.columns[cat_col_inds]:
            #     # prepare categorical-encoded
            #     values = np.hstack((X_train.loc[:, col], X_test.loc[:, col]))
            #     values = np.unique(values)
            #     values = sorted(values, key=lambda x: (len(x), x))
            
            #     encoding = dict(zip(values, range(len(values))))
            
            #     X_train[col+"_lexical"] = X_train[col].map(encoding)
            #     X_test[col+"_lexical"] = X_test[col].map(encoding)
                
            #     # prepare categorical-counts
            #     if use_test:
            #         counts = pd.concat((X_train[col], X_test[col])).value_counts()
            #     else:
            #         counts = X_train[col].value_counts()
            #     X_train[col+"_cnt"] = X_train[col].map(counts).values
            #     X_test[col+"_cnt"] = X_test[col].map(counts).values        
            # # prepare categorical-dummy
                        
            # prepare svd
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            # self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
            
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train  

    def expert_postprocessing(self, X_train, y, **kwargs):
        return np.expm1(y)

################################################################
################################################################
################################################################

class BNPParibasCardifClaimsManagement(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "bnp-paribas-cardif-claims-management"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "logloss"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [3,  22,  24,  30,  31,  47,  52,  56,  66,  71,  74,  75,  79, 91, 107, 110, 112, 113, 125]
        self.y_col = "target"
        self.large_dataset = False
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, target_encode_cat=False, **kwargs):
        '''
        Solution implemented: https://www.kaggle.com/code/confirm/xfeat-catboost-cpu-only

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")


            import xfeat
            from xfeat import SelectNumerical, SelectCategorical, LabelEncoder, Pipeline, ConcatCombination, ArithmeticCombinations, LambdaEncoder
            
            # import cudf
            
            # X_train = self.X_train.copy()
            # y_train = self.y_train.copy()
            # X_test = self.X_test.copy()
            
            out_path = f"./datasets/{self.dataset_name}/processed/"
            
            USECOLS = [
                "v10", "v12", "v14", "v21", "v22", "v24", "v30", "v31", "v34", "v38",
                "v40", "v47", "v50", "v52", "v56", "v62", "v66", "v72", "v75", "v79",
                "v91", "v112", "v113", "v114", "v129", "target"
            ]
            
            def preload():
                # Convert dataset into feather format.
                xfeat.utils.compress_df(pd.concat([
                    pd.concat([X_train,y_train],axis=1),
                    X_test,
                ], sort=False)).reset_index(drop=True)[USECOLS].to_feather(
                    out_path+f"train_test_{dataset_version}.ftr")
            
            
            preload()
            if not use_test:
                X_concat = xfeat.utils.compress_df(pd.concat([
                    pd.concat([X_train,y_train],axis=1),
                    X_test,
                ], sort=False)).reset_index(drop=True)[USECOLS]
            
            print("(1) Save numerical features")
            SelectNumerical().fit_transform(pd.read_feather(out_path+f"train_test_{dataset_version}.ftr")).reset_index(
                drop=True
            ).to_feather(out_path+f"feature_num_features_{dataset_version}.ftr")
            
            print("(2) Categorical encoding using label encoding: 13 features")
            if use_test:
                Pipeline([SelectCategorical(), LabelEncoder(output_suffix="")]).fit_transform(
                    pd.read_feather(out_path+f"train_test_{dataset_version}.ftr")
                ).reset_index(drop=True).to_feather(out_path+f"feature_1way_label_encoding_{dataset_version}.ftr")
            else:
                pipeline = Pipeline([SelectCategorical(), LabelEncoder(output_suffix="")])
                X_train_transformed = pipeline.fit_transform(
                    X_concat.iloc[:X_train.shape[0]]
                ).reset_index(drop=True)
                
                
                X_test_transformed = pipeline.transform(
                    X_concat.iloc[X_train.shape[0]:]
                ).reset_index(drop=True)

                pd.concat([X_train_transformed,X_test_transformed]).reset_index(drop=True).to_feather(out_path+f"feature_1way_label_encoding_{dataset_version}.ftr")
                
            
            print("(3) 2-order combination of categorical features: 78 features (13 * 12 / 2 = 78)")
            if use_test:
                Pipeline(
                    [
                        SelectCategorical(),
                        ConcatCombination(drop_origin=True, r=2),
                        LabelEncoder(output_suffix=""),
                    ]
                ).fit_transform(pd.read_feather(out_path+f"train_test_{dataset_version}.ftr")).reset_index(
                    drop=True
                ).to_feather(
                    out_path+f"feature_2way_label_encoding_{dataset_version}.ftr"
                )
            else:
                pipeline = Pipeline(
                        [
                            SelectCategorical(),
                            ConcatCombination(drop_origin=True, r=2),
                            LabelEncoder(output_suffix=""),
                        ]
                    )
                
                X_train_transformed = pipeline.fit_transform(
                    X_concat.iloc[:X_train.shape[0]]
                ).reset_index(drop=True)
                
                
                X_test_transformed = pipeline.transform(
                    X_concat.iloc[X_train.shape[0]:]
                ).reset_index(drop=True)
                
                pd.concat([X_train_transformed,X_test_transformed]).reset_index(drop=True).to_feather(
                    out_path+f"feature_2way_label_encoding_{dataset_version}.ftr"
                )
                
            
            print("(4) 3-order combination of categorical features")
            # Use `include_cols=` kwargs to reduce the total count of combinations.
            # 66 features (12 * 11 / 2 = 66)
            if use_test:
                Pipeline(
                    [
                        SelectCategorical(),
                        ConcatCombination(drop_origin=True, include_cols=["v22"], r=3),
                        LabelEncoder(output_suffix=""),
                    ]
                ).fit_transform(pd.read_feather(out_path+f"train_test_{dataset_version}.ftr")).reset_index(
                    drop=True
                ).to_feather(
                    out_path+f"feature_3way_including_v22_label_encoding_{dataset_version}.ftr"
                )
            else:
                pipeline = Pipeline(
                    [
                        SelectCategorical(),
                        ConcatCombination(drop_origin=True, include_cols=["v22"], r=3),
                        LabelEncoder(output_suffix=""),
                    ]
                )
                X_train_transformed = pipeline.fit_transform(
                    X_concat.iloc[:X_train.shape[0]]
                ).reset_index(drop=True)
                
                
                X_test_transformed = pipeline.transform(
                    X_concat.iloc[X_train.shape[0]:]
                ).reset_index(drop=True)
                
                pd.concat([X_train_transformed,X_test_transformed]).reset_index(drop=True).to_feather(
                    out_path+f"feature_3way_including_v22_label_encoding_{dataset_version}.ftr"
                )
                
            
            print("(5) Convert numerical to categorical using round: 12 features")
            if use_test:
                df_rnum = (
                    Pipeline(
                        [
                            SelectNumerical(),
                            LambdaEncoder(
                                lambda x: str(x)[:-2],
                                output_suffix="_rnum",
                                exclude_cols=["target"],
                            ),
                        ]
                    )
                    .fit_transform(pd.read_feather(out_path+f"train_test_{dataset_version}.ftr"))
                    .reset_index(drop=True)
                )
                df_rnum.to_feather(out_path+f"feature_round_num_{dataset_version}.ftr")
                rnum_cols = [col for col in df_rnum.columns if col.endswith("_rnum")]
                Pipeline([LabelEncoder(output_suffix="")]).fit_transform(
                    pd.read_feather(out_path+f"feature_round_num_{dataset_version}.ftr")[rnum_cols]
                ).reset_index(drop=True).to_feather(out_path+f"feature_round_num_label_encoding_{dataset_version}.ftr")
            else:
                pipeline = Pipeline(
                        [
                            SelectNumerical(),
                            LambdaEncoder(
                                lambda x: str(x)[:-2],
                                output_suffix="_rnum",
                                exclude_cols=["target"],
                            ),
                        ]
                    )
                df_rnum_train = (
                    pipeline.fit_transform(X_concat.iloc[:X_train.shape[0]])
                    .reset_index(drop=True)
                )
                
                df_rnum_test = (
                    pipeline.fit_transform(X_concat.iloc[X_train.shape[0]:])
                    .reset_index(drop=True)
                )
                
                pd.concat([df_rnum_train,df_rnum_test]).reset_index(drop=True).to_feather(out_path+f"feature_round_num_{dataset_version}.ftr")
                
                rnum_cols = [col for col in df_rnum_train.columns if col.endswith("_rnum")]
                pipeline = Pipeline([LabelEncoder(output_suffix="")])

                X_train_transformed = pipeline.fit_transform(
                    df_rnum_train[rnum_cols]
                ).reset_index(drop=True)
                
                
                X_test_transformed = pipeline.transform(
                    df_rnum_test[rnum_cols]
                ).reset_index(drop=True)
                
                pd.concat([X_train_transformed,X_test_transformed]).reset_index(drop=True).to_feather(
                    out_path+f"feature_round_num_label_encoding_{dataset_version}.ftr"
                )

                
            
            print("(6) 2-order Arithmetic combinations.")
            if use_test:
                Pipeline(
                    [
                        SelectNumerical(),
                        ArithmeticCombinations(
                            exclude_cols=["target"], drop_origin=True, operator="+", r=2,
                        ),
                    ]
                ).fit_transform(pd.read_feather(out_path+f"train_test_{dataset_version}.ftr")).reset_index(
                    drop=True
                ).to_feather(
                    out_path+f"feature_arithmetic_combi2_{dataset_version}.ftr"
                )
            else:
                pipeline = Pipeline(
                    [
                        SelectNumerical(),
                        ArithmeticCombinations(
                            exclude_cols=["target"], drop_origin=True, operator="+", r=2,
                        ),
                    ]
                )

                X_train_transformed = pipeline.fit_transform(
                    X_concat.iloc[:X_train.shape[0]]
                ).reset_index(drop=True)
                
                
                X_test_transformed = pipeline.transform(
                    X_concat.iloc[X_train.shape[0]:]
                ).reset_index(drop=True)
                
                pd.concat([X_train_transformed,X_test_transformed]).reset_index(drop=True).to_feather(
                    out_path+f"feature_arithmetic_combi2_{dataset_version}.ftr"
                )
            
            print("(7) Add more combinations: 11-order concat combinations.")
            if use_test:
                Pipeline(
                    [
                        SelectCategorical(),
                        ConcatCombination(drop_origin=True, include_cols=["v22"], r=11),
                        LabelEncoder(output_suffix=""),
                    ]
                ).fit_transform(pd.read_feather(out_path+f"train_test_{dataset_version}.ftr")).reset_index(
                    drop=True
                ).to_feather(
                    out_path+f"feature_11way_including_v22_label_encoding_{dataset_version}.ftr"
                )
            else:
                pipeline = Pipeline(
                    [
                        SelectCategorical(),
                        ConcatCombination(drop_origin=True, include_cols=["v22"], r=11),
                        LabelEncoder(output_suffix=""),
                    ]
                )
                
                X_train_transformed = pipeline.fit_transform(
                    X_concat.iloc[:X_train.shape[0]]
                ).reset_index(drop=True)
                
                
                X_test_transformed = pipeline.transform(
                    X_concat.iloc[X_train.shape[0]:]
                ).reset_index(drop=True)
                
                pd.concat([X_train_transformed,X_test_transformed]).reset_index(drop=True).to_feather(
                    out_path+f"feature_11way_including_v22_label_encoding_{dataset_version}.ftr"
                )                
                

            print("Load numerical features")
            df_num = pd.concat(
                [
                    pd.read_feather(out_path+f"feature_num_features_{dataset_version}.ftr"),
                    pd.read_feather(out_path+f"feature_arithmetic_combi2_{dataset_version}.ftr"),
                ],
                axis=1,
            )
            y_train = df_num["target"].dropna()
            df_num.drop(["target"], axis=1, inplace=True)
            
            print("Load categorical features")
            df = pd.concat(
                [
                    pd.read_feather(out_path+f"feature_1way_label_encoding_{dataset_version}.ftr"),
                    pd.read_feather(out_path+f"feature_2way_label_encoding_{dataset_version}.ftr"),
                    pd.read_feather(out_path+f"feature_3way_including_v22_label_encoding_{dataset_version}.ftr"),
                    pd.read_feather(out_path+f"feature_round_num_label_encoding_{dataset_version}.ftr"),
                    pd.read_feather(out_path+f"feature_11way_including_v22_label_encoding_{dataset_version}.ftr"),
                ],
                axis=1,
            )
            cat_cols = df.columns.tolist()
            df = pd.concat([df, df_num], axis=1)            

            X_train = df.iloc[:y_train.shape[0]]
            X_test = df.iloc[y_train.shape[0]:]

            self.cat_indices = [np.where(df.columns==i)[0][0] for i in cat_cols]
    
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))


            def oof_likelihood_encoding(X_train, y_train, folds, col, random_state=42):
                oof_likelihood_train = np.zeros(len(X_train))
            
                target = y_train.name
                Xy_train = pd.concat([X_train, y_train],axis=1)
                
                for fold_id, (train_idx, val_idx) in enumerate(folds):
                    train_fold, val_fold = Xy_train.iloc[train_idx], Xy_train.iloc[val_idx]
                    self.fold_cat_target_map[col][fold_id] = train_fold.groupby(col)[target].mean()
                    oof_likelihood_train[val_idx] = val_fold[col].map(self.fold_cat_target_map[col][fold_id])
                
                # For categories not present in training folds, we use the global mean
                global_mean = Xy_train[target].mean()
                oof_likelihood_train = np.where(pd.isnull(oof_likelihood_train), global_mean, oof_likelihood_train)
                
                return oof_likelihood_train
            
            self.original_cat_indices = self.cat_indices
            folds = self.get_cv_folds(X_train,y_train)
            self.fold_cat_target_map = {}
            Xy_train = pd.concat([X_train,y_train],axis=1)
            for col in X_train.columns[self.cat_indices]:
                self.fold_cat_target_map[col] = {}
                target_mean = Xy_train.groupby(col)[y_train.name].mean()
                X_test[col] = X_test[col].map(target_mean)
                X_test.loc[X_test[col].isna(),col] = Xy_train[y_train.name].mean()
                X_train[col] = oof_likelihood_encoding(X_train, y_train, folds, col) # Todo: Add seed 
            self.cat_indices = []
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    def get_cv_folds(self, X_train, y_train, seed=42):
        ss = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        folds = []
        for num, (train,test) in enumerate(ss.split(X_train.copy(), y_train.copy())):
            folds.append([train, test])  

        return folds

    def pred_to_submission(self, y_pred):
        submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sample_submission.csv", engine="pyarrow")
        if self.toy_example:
            submission = submission.iloc[:1000]
        submission["PredictedProb"] = y_pred

        return submission

##########################################################   
##########################################################   
##########################################################        
        
class  RestaurantRevenuePrediction(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "restaurant-revenue-prediction"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "regression" # "binary", "classification"
        self.eval_metric_name = "rmse"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [0,1,2]            
        self.y_col = "revenue"
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        y_train = data[self.y_col]

        date = pd.to_datetime(data["Open Date"])
        data["month"] = date.dt.month
        data["year"] = date.dt.year
        data["weekday"] = date.dt.weekday
        data["lasting_days"] = (pd.Timestamp.today()-date).dt.days
        data["lasting_years"] = ((pd.Timestamp.today()-date).dt.days/365).apply(np.floor)

        date = pd.to_datetime(X_test["Open Date"])
        X_test["month"] = date.dt.month
        X_test["year"] = date.dt.year
        X_test["weekday"] = date.dt.weekday
        X_test["lasting_days"] = (pd.Timestamp.today()-date).dt.days
        X_test["lasting_years"] = ((pd.Timestamp.today()-date).dt.days/365).apply(np.floor)

        X_train = data.drop([self.y_col, "Id", "Open Date"],axis=1)    
        X_test = X_test.drop(["Id", "Open Date"],axis=1)
        
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Solution implemented based on the descriptions in https://www.kaggle.com/competitions/restaurant-revenue-prediction/discussion/14066

        1. Square root transformation was applied to the obfuscated P variables with maximum value >= 10, to make them into the same scale, as well as the target variable “revenue”.
        2. Random assignments of uncommon city levels to the common city levels in both training and test set, which I believe, diversified the geo location information contained in the city variable and in some of the obfuscated P variables.
        3. Missing value indicator for multiple P variables, i.e. P14 to P18, P24 to P27, and P30 to P37 was created to help differentiate synthetic and real test data (based on info from test data so might be problematic)
        4. Type “MB”, which did not occur in training set, was changed to Type “DT” in test set.
        5. Time / Age related information was also extracted, including open day, week, month and lasting years and days.
        6. Zeroes were treated as missing values and mice imputation was applied on training and test set separately.

        Additional remarks: 
            - Used 10-fold cv repeated 10 times (default setting)
            - Used GBDTs
            - Selected best model not only based on CV score, but also with outlier removal

        '''
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            
            # 1. Square root transformation was applied to the obfuscated P variables with maximum value >= 10
            X_train[[f"P{i}" for i in range(1,38)]] = np.sqrt(X_train[[f"P{i}" for i in range(1,38)]])
            X_test[[f"P{i}" for i in range(1,38)]] = np.sqrt(X_test[[f"P{i}" for i in range(1,38)]])

            # 2. Original: Random assignments of uncommon city levels to the common city levels in both training and test set
            # We use an 'other' category as this is more common practice 
            use_cities = ["İstanbul","Ankara","İzmir","Bursa","Samsun","Sakarya","Antalya"]
            X_train.City = X_train.City.apply(lambda x: "Other" if x not in use_cities else x)
            X_test.City = X_test.City.apply(lambda x: "Other" if x not in use_cities else x)

            # 3. Missing value indicator for multiple P variables, i.e. P14 to P18, P24 to P27, and P30 to P37 was created to help differentiate synthetic and real test data (based on info from test data so might be problematic)
            X_train["missing_indicator"] = (X_train["P14"]==0)*1
            X_test["missing_indicator"] = (X_test["P14"]==0)*1

            # 4. Type “MB”, which did not occur in training set, was changed to Type “DT” in test set.
            X_test.Type = X_test.Type.apply(lambda x: "DT" if x=="MB" else x)
            
            # 5. PART OF LOAD FUNCTION Time / Age related information was also extracted, including open day, week, month and lasting years and days.
                        
            # 6. Zeroes were treated as missing values and mice imputation was applied on training and test set separately.
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            
            mice_imputer = IterativeImputer()
            missing = X_train[[f"P{i}" for i in range(1,38)]]
            missing[missing==0] = np.nan
            X_train[[f"P{i}" for i in range(1,38)]] = mice_imputer.fit_transform(missing)

            mice_imputer = IterativeImputer()
            missing = X_test[[f"P{i}" for i in range(1,38)]]
            missing[missing==0] = np.nan
            X_test[[f"P{i}" for i in range(1,38)]] = mice_imputer.fit_transform(missing)

            y_train = np.sqrt(y_train)
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))
                
        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))

        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train  
    
    def expert_postprocessing(self, X, y, **kwargs):
        return np.power(y,2)
    
    def get_cv_folds(self, X_train, y_train, seed=42):
        ### !! Currently not original implemented - original solution used 30-fold CV - but also dicusses 5-fold
        folds = []
        # for seed_num in range(seed, seed+10):
        ss = KFold(n_splits=10, random_state=seed, shuffle=True)
        for num, (train,test) in enumerate(ss.split(y_train.copy(), y_train.copy())):
            folds.append([train, test])
        return folds
        
    def pred_to_submission(self, y_pred):
        submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sampleSubmission.csv", engine="pyarrow")
        if self.toy_example:
            submission = submission.iloc[:1000]
        submission["Prediction"] = y_pred

        return submission

################################################################
################################################################
################################################################

# class MoAPrediction(BaseDataset):
#     def __init__(self, toy_example=False):
#         super().__init__(toy_example)
#         self.dataset_name = "lish-moa"
#         ############## 0. Define Data Parameters  ##############
#         self.task_type = "multilabel"
#         self.eval_metric_name = "multilogloss"
#         self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

#         self.cat_indices = [0,1,2,3]
#         self.y_col = "Class"
#         self.large_dataset = False


#     def load_data(self):
#         data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train_features.csv', engine="pyarrow")
#         X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test_features.csv', engine="pyarrow")
#         # y_train = 
        
#         drug = pd.read_csv('datasets/lish-moa/raw/train_drug.csv', engine="pyarrow")
#         data = pd.merge(drug,data,on="sig_id")
#         X_test = pd.merge(X_test,data,on="sig_id")

#         if self.toy_example:
#             data = data.iloc[:1000]
#             X_test = X_test.iloc[:1000]
#         y_train = data[self.y_col]

#         X_train = data.drop(["sig_id"],axis=1)    
#         X_test = X_test.drop(["sig_id"],axis=1)
        
        
#     #     self.X_train, self.X_test, self.y_train = X_train, X_test, y_train       
    
#     def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
#         '''
#         Summary of the solution implemented (https://www.kaggle.com/competitions/lish-moa/discussion/202256, https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution)

        

#         '''
        
#         if use_test and not self.toy_example:
#             dataset_version = "expert_test"
#         elif not use_test and not self.toy_example:
#             dataset_version = "expert_notest"
#         elif use_test and self.toy_example:
#             dataset_version = "expert_test_toy"
#         elif not use_test and not self.toy_example:
#             dataset_version = "expert_notest_toy"
        
#         if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
#             print(f"Apply expert preprocessing")


#             self.cat_indices = [np.where(df.columns==i)[0][0] for i in cat_cols]
    
#             if not self.toy_example:
#                 os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
#                 pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
#                 pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
#                 pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
#                 pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

#         else:
#             print(f"Load existing expert-preprocessed data")
#             X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
#             y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
#             X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
#             self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
#         self.preprocess_states.append("expert")        
#         self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
#     # def pred_to_submission(self, y_pred):
#     #     submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sample_submission.csv", engine="pyarrow")
#     #     if self.toy_example:
#     #         submission = submission.iloc[:1000]
#     #     submission["class_0"] = 1-y_pred
#     #     submission["class_1"] = y_pred

#     #     return submission        


################################################################
################################################################
################################################################

class ZillowPrice(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "zillow-prize-1"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "regression"
        self.eval_metric_name = "mae"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [25, 35, 37, 52, 58,  1,  2,  3,  4,  5,  9, 10, 12, 26, 35, 36, 39,
       40, 41, 42, 46, 59]
        self.y_col = "logerror"
        self.large_dataset = False


    def load_data(self):
        # From: https://www.kaggle.com/code/abdelwahedassklou/only-cat-boost-lb-0-0641-939

        if not os.path.exists("datasets/zillow-prize-1/processed/X_train_raw.csv"):
            # Loading Properties ...
            properties2016 = pd.read_csv('datasets/zillow-prize-1/raw/properties_2016.csv', engine="pyarrow")
            properties2017 = pd.read_csv('datasets/zillow-prize-1/raw/properties_2017.csv', engine="pyarrow")
            
            # Loading Train ...
            train2016 = pd.read_csv('datasets/zillow-prize-1/raw/train_2016_v2.csv', parse_dates=['transactiondate'], engine="pyarrow")
            train2017 = pd.read_csv('datasets/zillow-prize-1/raw/train_2017.csv', parse_dates=['transactiondate'], engine="pyarrow")
            
            def add_date_features(df):
                df["transaction_year"] = df["transactiondate"].dt.year
                df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
                df["transaction_day"] = df["transactiondate"].dt.day
                df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
                df.drop(["transactiondate"], inplace=True, axis=1)
                return df
            
            train2016 = add_date_features(train2016)
            train2017 = add_date_features(train2017)
            
            # Loading Sample ...
            sample_submission = pd.read_csv('datasets/zillow-prize-1/raw/sample_submission.csv', engine="pyarrow")
            
            # Merge Train with Properties ...
            train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
            train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')
            
            # Concat Train 2016 & 2017 ...
            train_df = pd.concat([train2016, train2017], axis = 0)
            
            y_train = train_df["logerror"]
            X_train = train_df.drop(["logerror"],axis=1)
            
            test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')
            
            test_df['transactiondate'] = pd.Timestamp('2016-12-01') 
            test_df = add_date_features(test_df)
            X_test = test_df
            X_test = X_test.rename({"ParcelId": "parcelid"},axis=1)
            X_test = X_test[X_train.columns] 

            X_train.to_csv("datasets/zillow-prize-1/processed/X_train_raw.csv",index=False)
            y_train.to_csv("datasets/zillow-prize-1/processed/y_train_raw.csv",index=False)
            X_test.to_csv("datasets/zillow-prize-1/processed/X_test_raw.csv",index=False)

            # Might decide to move either to minimalistic or expert
            for col in X_train.columns[np.logical_and(X_train.nunique()==1,X_train.isna().sum()>0)]:
                X_train[col].loc[~X_train[col].isna()] = 1
                X_train[col].loc[X_train[col].isna()] = 0
                X_test[col].loc[~X_test[col].isna()] = 1
                X_test[col].loc[X_test[col].isna()] = 0
        
        else:
            X_train = pd.read_csv("datasets/zillow-prize-1/processed/X_train_raw.csv", engine="pyarrow")
            y_train = pd.read_csv("datasets/zillow-prize-1/processed/y_train_raw.csv", engine="pyarrow")[self.y_col]
            X_test = pd.read_csv("datasets/zillow-prize-1/processed/X_test_raw.csv", engine="pyarrow")
            
            # Might decide to move either to minimalistic or expert
            for col in X_train.columns[np.logical_and(X_train.nunique()==1,X_train.isna().sum()>0)]:
                X_train[col].loc[~X_train[col].isna()] = 1
                X_train[col].loc[X_train[col].isna()] = 0
                X_test[col].loc[~X_test[col].isna()] = 1
                X_test[col].loc[X_test[col].isna()] = 0

        X_train = X_train.drop('parcelid',axis=1)
        X_test = X_test.drop('parcelid',axis=1)
        
        if self.toy_example:
            X_train = X_train.iloc[:1000]
            y_train = y_train.iloc[:1000]
            X_test = X_test.iloc[:1000]
        
        cat_cols = ["hashottuborspa", "propertycountylandusecode","propertyzoningdesc","fireplaceflag","taxdelinquencyflag","transaction_month","transaction_day", "transaction_quarter","airconditioningtypeid","architecturalstyletypeid","buildingclasstypeid","buildingqualitytypeid","decktypeid", "heatingorsystemtypeid","propertycountylandusecode","propertylandusetypeid","regionidcity","regionidcounty","regionidneighborhood", "regionidzip","typeconstructiontypeid","taxdelinquencyyear"]
        self.cat_indices = np.array([np.where(X_train.columns==i)[0][0]-1 for i in cat_cols]).tolist()
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train       
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented (https://www.kaggle.com/competitions/lish-moa/discussion/202256, https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution)

        

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            # print('Tax Features 2017  ...')
            # X_train.iloc[:, X_train.columns.str.startswith('tax')] = np.nan
            # # X_test.iloc[:, X_train.columns.str.startswith('tax')] = np.nan

            # print('Remove missing data fields ...')
            # missing_perc_thresh = 0.98
            # exclude_missing = []
            # num_rows = X_train.shape[0]
            # for c in X_train.columns:
            #     num_missing = X_train[c].isnull().sum()
            #     if num_missing == 0:
            #         continue
            #     missing_frac = num_missing / float(num_rows)
            #     if missing_frac > missing_perc_thresh:
            #         exclude_missing.append(c)
            # print("We exclude: %s" % len(exclude_missing))


            # print ("Remove features with one unique value !!")
            # exclude_unique = []
            # for c in X_train.columns:
            #     num_uniques = len(X_train[c].unique())
            #     if X_train[c].isnull().sum() != 0:
            #         num_uniques -= 1
            #     if num_uniques == 1:
            #         exclude_unique.append(c)
            # print("We exclude: %s" % len(exclude_unique))

            
            # print ("Define training features !!")
            # exclude_other = ['propertyzoningdesc']
            # train_features = []
            # for c in X_train.columns:
            #     if c not in exclude_missing \
            #        and c not in exclude_other and c not in exclude_unique:
            #         train_features.append(c)
            # print("We use these for training: %s" % len(train_features))


            # print ("Define categorial features !!")
            # cat_feature_inds = []
            # cat_unique_thresh = 1000
            # for i, c in enumerate(train_features):
            #     num_uniques = len(X_train[c].unique())
            #     if num_uniques < cat_unique_thresh \
            #        and not 'sqft' in c \
            #        and not 'cnt' in c \
            #        and not 'nbr' in c \
            #        and not 'number' in c:
            #         cat_feature_inds.append(i)

            # print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

            # X_train = X_train[train_features]
            # X_test = X_test[train_features]

            from tqdm import tqdm
            import gc
            import datetime as dt
            
            print('Loading Properties ...')
            properties2016 = pd.read_csv('datasets/zillow-prize-1/raw/properties_2016.csv', low_memory = False)
            properties2017 = pd.read_csv('datasets/zillow-prize-1/raw/properties_2017.csv', low_memory = False)
            
            print('Loading Train ...')
            train2016 = pd.read_csv('datasets/zillow-prize-1/raw/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
            train2017 = pd.read_csv('datasets/zillow-prize-1/raw/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
            
            def add_date_features(df):
                df["transaction_year"] = df["transactiondate"].dt.year
                df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
                df["transaction_day"] = df["transactiondate"].dt.day
                df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
                df.drop(["transactiondate"], inplace=True, axis=1)
                return df
            
            train2016 = add_date_features(train2016)
            train2017 = add_date_features(train2017)
            
            print('Loading Sample ...')
            sample_submission = pd.read_csv('datasets/zillow-prize-1/raw/sample_submission.csv', low_memory = False)
            
            print('Merge Train with Properties ...')
            train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
            train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')
            
            print('Tax Features 2017  ...')
            train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan
            
            print('Concat Train 2016 & 2017 ...')
            train_df = pd.concat([train2016, train2017], axis = 0)
            test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')
            
            del properties2016, properties2017, train2016, train2017
            gc.collect();
            
            print('Remove missing data fields ...')
            
            missing_perc_thresh = 0.98
            exclude_missing = []
            num_rows = train_df.shape[0]
            for c in train_df.columns:
                num_missing = train_df[c].isnull().sum()
                if num_missing == 0:
                    continue
                missing_frac = num_missing / float(num_rows)
                if missing_frac > missing_perc_thresh:
                    exclude_missing.append(c)
            print("We exclude: %s" % len(exclude_missing))
            
            del num_rows, missing_perc_thresh
            gc.collect();
            
            print ("Remove features with one unique value !!")
            exclude_unique = []
            for c in train_df.columns:
                num_uniques = len(train_df[c].unique())
                if train_df[c].isnull().sum() != 0:
                    num_uniques -= 1
                if num_uniques == 1:
                    exclude_unique.append(c)
            print("We exclude: %s" % len(exclude_unique))
            
            print ("Define training features !!")
            exclude_other = ['parcelid', 'logerror','propertyzoningdesc']
            train_features = []
            for c in train_df.columns:
                if c not in exclude_missing \
                   and c not in exclude_other and c not in exclude_unique:
                    train_features.append(c)
            print("We use these for training: %s" % len(train_features))
            
            print ("Define categorial features !!")
            cat_feature_inds = []
            cat_unique_thresh = 1000
            for i, c in enumerate(train_features):
                num_uniques = len(train_df[c].unique())
                if num_uniques < cat_unique_thresh \
                   and not 'sqft' in c \
                   and not 'cnt' in c \
                   and not 'nbr' in c \
                   and not 'number' in c:
                    cat_feature_inds.append(i)
                    
            print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])
            
            print ("Replacing NaN values by -999 !!")
            train_df.fillna(-999, inplace=True)
            test_df.fillna(-999, inplace=True)
            
            print ("Training time !!")
            X_train = train_df[train_features]
            y_train = train_df.logerror
            print(X_train.shape, y_train.shape)
            
            test_df['transactiondate'] = pd.Timestamp('2016-12-01') 
            test_df = add_date_features(test_df)
            X_test = test_df[train_features]
            print(X_test.shape)
            
            self.cat_indices = cat_feature_inds # [np.where(df.columns==i)[0][0] for i in cat_cols]
    
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
    def pred_to_submission(self, y_pred):
        submission = pd.read_csv('datasets/zillow-prize-1/raw/sample_submission.csv', engine="pyarrow")
        if self.toy_example:
            submission = submission.iloc[:1000]
        
        test_dates = {
            '201610': pd.Timestamp('2016-09-30'),
            '201611': pd.Timestamp('2016-10-31'),
            '201612': pd.Timestamp('2016-11-30'),
            '201710': pd.Timestamp('2017-09-30'),
            '201711': pd.Timestamp('2017-10-31'),
            '201712': pd.Timestamp('2017-11-30')
        }
        for label, test_date in test_dates.items():
            submission[label] = y_pred
            
        return submission        

################################################################
################################################################
################################################################

class OttoGroupProductClassification(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "otto-group-product-classification-challenge"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "classification"
        self.eval_metric_name = "mlogloss"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []
        self.y_col = "target"
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        y_train = data[self.y_col]
        X_train = data.drop([self.y_col, "id"],axis=1)    
        X_test = X_test.drop("id",axis=1)    

        if self.task_type== "classification":
            self.target_label_enc = LabelEncoder()
            y_train = pd.Series(self.target_label_enc.fit_transform(y_train),index=y_train.index, name=y_train.name)
            self.num_classes = y_train.nunique()
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
    
    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented (https://www.kaggle.com/competitions/otto-group-product-classification-challenge/discussion/14295)

        1. Standardize data with x = 1/(1+exp(-sqrt(x)))
        2. Compute tsne features and append them to the data

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            X_train = 1/(1+np.exp(-np.sqrt(X_train)))
            X_test = 1/(1+np.exp(-np.sqrt(X_test)))
    
            if use_test:
                X_concat = pd.concat([X_train,X_test])
                tsne = TSNE(n_components=2, verbose=0, perplexity=30, angle=0.5)
                X_2d = tsne.fit_transform(X_concat)
                X_concat["tsne_1"] = X_2d[:,0]
                X_concat["tsne_2"] = X_2d[:,1]
                X_train = X_concat.iloc[:X_train.shape[0]]
                X_test = X_concat.iloc[X_train.shape[0]:]
            else: 

                pca = PCA(n_components=2)
                
                X_2d = pca.fit_transform(X_train)
                X_train["tsne_1"] = X_2d[:,0]
                X_train["tsne_2"] = X_2d[:,1]
                
                X_2d_test = pca.transform(X_test)
                X_test["tsne_1"] = X_2d_test[:,0]
                X_test["tsne_2"] = X_2d_test[:,1]
            
                for ncl in range(2, 11):
                    cls = KMeans(n_clusters=ncl)
                    cls.fit(X_train.values)
                    key = "kmeans_cluster" + str(ncl)
                    X_train[key] = cls.predict(X_train.values)
                    X_test[key] = cls.predict(X_test.values)

            X_train = np.round(X_train,3)
            X_test = np.round(X_test,3)
            
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            # self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
    # def pred_to_submission(self, y_pred):
    #     submission = pd.read_csv(f'datasets/zillow-prize-1/raw/sample_submission.csv', engine="pyarrow")
    #     if self.toy_example:
    #         submission = submission.iloc[:1000]
        
    #     test_dates = {
    #         '201610': pd.Timestamp('2016-09-30'),
    #         '201611': pd.Timestamp('2016-10-31'),
    #         '201612': pd.Timestamp('2016-11-30'),
    #         '201710': pd.Timestamp('2017-09-30'),
    #         '201711': pd.Timestamp('2017-10-31'),
    #         '201712': pd.Timestamp('2017-11-30')
    #     }
    #     for label, test_date in test_dates.items():
    #         submission[label] = y_pred
            
    #     return submission        

    def pred_to_submission(self, y_pred):
        submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sampleSubmission.csv", engine="pyarrow")
        
        submission[submission.columns[1:]] = y_pred
        

        return submission

################################################################
################################################################
################################################################

class SpringleafMarketingResponse(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "springleaf-marketing-response"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [   0,    4,   72,   74,  155,  156,  157,  158,  165,  166,  167,
        168,  175,  176,  177,  178,  199,  203,  213,  216,  235,  271,
        280,  302,  322,  339,  349,  350,  351,  401,  464,  490, 1931]
        self.y_col = "target"
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        
        binary_value_nan = data.columns[np.logical_and(data.dtypes=="object",data.nunique()<=2)]
        for col in binary_value_nan:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            X_test[col] = le.transform(X_test[col])
        
        y_train = data[self.y_col]
        X_train = data.drop([self.y_col, "ID"],axis=1)    
        X_test = X_test.drop("ID",axis=1)    

        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented 

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            # self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
    # def pred_to_submission(self, y_pred):
    #     submission = pd.read_csv(f'datasets/zillow-prize-1/raw/sample_submission.csv', engine="pyarrow")
    #     if self.toy_example:
    #         submission = submission.iloc[:1000]
        
    #     test_dates = {
    #         '201610': pd.Timestamp('2016-09-30'),
    #         '201611': pd.Timestamp('2016-10-31'),
    #         '201612': pd.Timestamp('2016-11-30'),
    #         '201710': pd.Timestamp('2017-09-30'),
    #         '201711': pd.Timestamp('2017-10-31'),
    #         '201712': pd.Timestamp('2017-11-30')
    #     }
    #     for label, test_date in test_dates.items():
    #         submission[label] = y_pred
            
    #     return submission        

    # def pred_to_submission(self, y_pred):
    #     submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sampleSubmission.csv", engine="pyarrow")
        
    #     submission[submission.columns[1:]] = y_pred
        
################################################################
################################################################
################################################################

class PrudentialLifeInsuranceAssessment(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "prudential-life-insurance-assessment"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "cohens_kappa"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []
        self.y_col = ""
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        
        # binary_value_nan = data.columns[np.logical_and(data.dtypes=="object",data.nunique()<=2)]
        # for col in binary_value_nan:
        #     le = LabelEncoder()
        #     data[col] = le.fit_transform(data[col])
        #     X_test[col] = le.transform(X_test[col])
        
        y_train = data[self.y_col]
        X_train = data.drop([self.y_col, "ID"],axis=1)    
        X_test = X_test.drop("ID",axis=1)    

        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented (https://www.kaggle.com/competitions/otto-group-product-classification-challenge/discussion/14295)

        1. Standardize data with x = 1/(1+exp(-sqrt(x)))
        2. Compute tsne features and append them to the data

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            # self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
    # def pred_to_submission(self, y_pred):
    #     submission = pd.read_csv(f'datasets/zillow-prize-1/raw/sample_submission.csv', engine="pyarrow")
    #     if self.toy_example:
    #         submission = submission.iloc[:1000]
        
    #     test_dates = {
    #         '201610': pd.Timestamp('2016-09-30'),
    #         '201611': pd.Timestamp('2016-10-31'),
    #         '201612': pd.Timestamp('2016-11-30'),
    #         '201710': pd.Timestamp('2017-09-30'),
    #         '201711': pd.Timestamp('2017-10-31'),
    #         '201712': pd.Timestamp('2017-11-30')
    #     }
    #     for label, test_date in test_dates.items():
    #         submission[label] = y_pred
            
    #     return submission        

    # def pred_to_submission(self, y_pred):
    #     submission = pd.read_csv(f"datasets/{self.dataset_name}/raw/sampleSubmission.csv", engine="pyarrow")
        
    #     submission[submission.columns[1:]] = y_pred


################################################################
################################################################
################################################################

class MicrosoftMalwarePrediction(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "microsoft-malware-prediction"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = [0,  1,  2,  3, 17, 18, 19, 22, 23, 24, 27, 30, 33, 34, 40, 42, 46, 50, 51, 53, 54, 55, 58, 59, 60, 63, 65, 66, 69]
        self.y_col = "HasDetections"
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        
        # binary_value_nan = data.columns[np.logical_and(data.dtypes=="object",data.nunique()<=2)]
        # for col in binary_value_nan:
        #     le = LabelEncoder()
        #     data[col] = le.fit_transform(data[col])
        #     X_test[col] = le.transform(X_test[col])
        
        y_train = data[self.y_col]
        X_train = data.drop([self.y_col, "MachineIdentifier"],axis=1)    
        X_test = X_test.drop("MachineIdentifier",axis=1)    

        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented 

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            # self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        


################################################################
################################################################
################################################################

class HomesiteQuoteConversion(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "homesite-quote-conversion"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        #self.cat_indices = [  1,   3,   7,   9,  22,  23,  34,  49,  59,  60,  61,  62, 130, 131, 132, 134, 142, 160, 162, 163, 164, 165, 166, 168, 169, 170, 297, 298]
        # update for after removal of target col
        self.cat_indices = [1, 2, 6, 8, 21, 22, 33, 48, 58, 59, 60, 61, 129, 130, 131, 133, 141, 159, 161, 162, 163, 164, 165, 167, 168, 169, 296, 297]
        self.y_col = "QuoteConversion_Flag"
        self.large_dataset = False

    def load_data(self):
        data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
        X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
        if self.toy_example:
            data = data.iloc[:1000]
            X_test = X_test.iloc[:1000]
        
        # binary_value_nan = data.columns[np.logical_and(data.dtypes=="object",data.nunique()<=2)]
        # for col in binary_value_nan:
        #     le = LabelEncoder()
        #     data[col] = le.fit_transform(data[col])
        #     X_test[col] = le.transform(X_test[col])
        
        y_train = data[self.y_col]
        X_train = data.drop([self.y_col],axis=1)    

        X_test["PropertyField37"] = X_test["PropertyField37"].replace(" ", None)

        # add weekday cat feature
        def extract_weekday(date_str):
            date_str = str(date_str)
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.weekday()

        X_train["weekday"] = X_train["Original_Quote_Date"].apply(extract_weekday)
        X_test["weekday"] = X_test["Original_Quote_Date"].apply(extract_weekday)

        self.cat_indices.append(X_train.columns.get_loc("weekday"))
        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Solution implemented:            
            https://www.kaggle.com/competitions/homesite-quote-conversion/discussion/18225  
            Feature described by user Julien in https://www.kaggle.com/competitions/homesite-quote-conversion/discussion/18831; 
            

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")
            
            #print(X_train.columns[X_train.isna().any()].tolist())
            # for col in X_train.columns[X_train.isna().any()].tolist():
            #     X_train[col] = X_train[col].replace(np.nan, -1)

            # for col in X_test.columns[X_test.isna().any()].tolist():
            #     X_test[col] = X_test[col].replace(np.nan, -1)

            X_train["sum_of_nas"] = X_train.isna().sum(axis=1)
            X_test["sum_of_nas"] = X_test.isna().sum(axis=1)
            
            X_train["sum_of_zeros"] = (X_train==0).sum(axis=1)
            X_test["sum_of_zeros"] = (X_test==0).sum(axis=1)

            def interaction_weekday_SalesField7(dataframe):
                dataframe["weekday_salesField7"] = dataframe["SalesField7"] + dataframe["weekday"].map(str)
                return dataframe
            
            X_train = interaction_weekday_SalesField7(X_train)
            X_test = interaction_weekday_SalesField7(X_test)
            self.cat_indices.append(X_train.columns.get_loc("weekday_salesField7"))


            # Encode binary cat features as numeric
            for col in X_train.columns[X_train.nunique()==2]:
                if X_train[col].dtype in [str, "O", "category", "object"]:
                    le = LabelEncoder()
                    mode = X_train[col].mode()[0]
                    X_train[col] = le.fit_transform(X_train[col])
    
                    if len(X_test[col].unique())==2:
                        X_test[col] = le.transform(X_test[col])
                    else:
                        X_test[col] = X_test[col].fillna(mode)
                        X_test[col] = le.transform(X_test[col])
                    
            
            # Define categorical feature types
            self.cat_indices += list(np.where(X_train.dtypes=="O")[0]) 
            self.cat_indices += list(np.where(X_train.dtypes=="object")[0]) 
            self.cat_indices += list(np.where(X_train.dtypes=="category")[0]) 
            self.cat_indices = np.unique(self.cat_indices).tolist()
            
            for num, col in list(zip(self.cat_indices,X_train.columns[self.cat_indices])):
                # Encode binary categorical features
                if X_train[col].nunique()==2:
                    value_1 = X_train[col].dropna().unique()[0]
                    X_train[col] = (X_train[col]==value_1).astype(float)
                    X_test[col] = (X_test[col]==value_1).astype(float)
                    self.cat_indices.remove(num)
                else:
                    # Note: The category dtype needs to entail all train categories when applying .astype("category") on test data
                    dtype = pd.CategoricalDtype(categories=list(X_train[col].astype(str).fillna("nan").unique()))
                    X_train[col] = X_train[col].astype(str).fillna("nan").astype(dtype)
                    X_test[col] = X_test[col].astype(str).fillna("nan").astype(dtype)       
                    
            
            # Drop constant columns
            # drop_cols = X_train.columns[X_train.nunique()==X_train.shape[0]].values.tolist()
            drop_cols = X_train.columns[X_train.nunique()==1].values.tolist()
            if len(drop_cols)>0:
                print(f"Drop {len(drop_cols)} constant/unique features")
                original_categorical_names =  X_train.columns[self.cat_indices]
                X_train.drop(drop_cols,axis=1,inplace=True)
                X_test.drop(drop_cols,axis=1,inplace=True)
                self.cat_indices = [np.where(X_train.columns==i)[0][0] for i in original_categorical_names if i in X_train.columns]
            
            os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
            pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
            pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
            pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))

        
        self.preprocess_states.append("expert")   
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
    


################################################################
################################################################
################################################################

class PredictingRedHatBusinessValue(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "predicting-red-hat-business-value"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "binary"
        self.eval_metric_name = "auc"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []
        self.y_col = "outcome"
        self.large_dataset = False

    # def load_data(self):
    #     data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
    #     X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
    #     if self.toy_example:
    #         data = data.iloc[:1000]
    #         X_test = X_test.iloc[:1000]
        
        # binary_value_nan = data.columns[np.logical_and(data.dtypes=="object",data.nunique()<=2)]
        # for col in binary_value_nan:
        #     le = LabelEncoder()
        #     data[col] = le.fit_transform(data[col])
        #     X_test[col] = le.transform(X_test[col])
        
        # y_train = data[self.y_col]
        # X_train = data.drop([self.y_col],axis=1)    
        
        # self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented 

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            # self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
    
################################################################
################################################################
################################################################

class TalkingdataMobileUserDemographics(BaseDataset):
    def __init__(self, toy_example=False):
        super().__init__(toy_example)
        self.dataset_name = "talkingdata-mobile-user-demographics"
        ############## 0. Define Data Parameters  ##############
        self.task_type = "classification"
        self.eval_metric_name = "mlogloss"
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        self.cat_indices = []
        self.y_col = "?"
        self.large_dataset = False

    # def load_data(self):
    #     data = pd.read_csv(f'./datasets/{self.dataset_name}/raw/train.csv', engine="pyarrow")
    #     X_test = pd.read_csv(f'./datasets/{self.dataset_name}/raw/test.csv', engine="pyarrow")
    #     if self.toy_example:
    #         data = data.iloc[:1000]
    #         X_test = X_test.iloc[:1000]
        
        # binary_value_nan = data.columns[np.logical_and(data.dtypes=="object",data.nunique()<=2)]
        # for col in binary_value_nan:
        #     le = LabelEncoder()
        #     data[col] = le.fit_transform(data[col])
        #     X_test[col] = le.transform(X_test[col])
        
        # y_train = data[self.y_col]
        # X_train = data.drop([self.y_col],axis=1)    
        
        # self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     

    
    def expert_preprocessing(self, X_train, X_test, y_train, overwrite_existing=False, use_test=True, **kwargs):
        '''
        Summary of the solution implemented 

        '''
        
        if use_test and not self.toy_example:
            dataset_version = "expert_test"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest"
        elif use_test and self.toy_example:
            dataset_version = "expert_test_toy"
        elif not use_test and not self.toy_example:
            dataset_version = "expert_notest_toy"
        
        if not os.path.exists(f"./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle") or overwrite_existing:
            print(f"Apply expert preprocessing")

            
            if not self.toy_example:
                os.makedirs(f'./datasets/{self.dataset_name}/processed/', exist_ok=True)
                pickle.dump(X_train, open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(y_train, open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'wb'))            
                pickle.dump(X_test, open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'wb'))            
                # pickle.dump(self.cat_indices, open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'wb'))

        else:
            print(f"Load existing expert-preprocessed data")
            X_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_train_{dataset_version}.pickle', 'rb'))            
            y_train = pickle.load(open(f'./datasets/{self.dataset_name}/processed/y_train_{dataset_version}.pickle', 'rb'))            
            X_test = pickle.load(open(f'./datasets/{self.dataset_name}/processed/X_test_{dataset_version}.pickle', 'rb'))
            # self.cat_indices = pickle.load(open(f'./datasets/{self.dataset_name}/processed/cat_indices_{dataset_version}.pickle', 'rb'))
        
        self.preprocess_states.append("expert")        
        self.X_train, self.X_test, self.y_train = X_train, X_test, y_train     
        
