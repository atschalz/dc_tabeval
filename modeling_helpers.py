import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
warnings.filterwarnings("ignore", message="Failed to load image Python extension:*")
import logging

import tensorflow as tf
# Set TensorFlow logging to ERROR only
tf.get_logger().setLevel(logging.ERROR)

import random
import numpy as np
import pickle
import time

import pandas as pd
import numpy as np

from models import get_model
from utils import set_seed, get_metric
from datasets import get_dataset

import optuna
import joblib
import ray
import gc


def get_submission_multiseed(configs, seeds=[42,2024]):

    exp_name = configs["exp_name"]
    model_name = configs["model"]["model_name"]
    preprocess_type = configs["dataset"]["preprocess_type"]

    
    # if configs["hpo"]["n_trials"] is None:
    results = {}
    dataset = get_dataset(configs["dataset"]["dataset_name"])
    dataset.load_data()
    n_folds = len(dataset.get_cv_folds(dataset.X_train,dataset.y_train))

    configs["model"]["save_path"] = f'results/{dataset.dataset_name}/{preprocess_type}/{model_name}/{exp_name}/'
    if not os.path.exists(configs["model"]["save_path"]):
        os.makedirs(configs["model"]["save_path"])
    configs["model"]["exp_name"] = configs["exp_name"]
    
    for num, seed in enumerate(seeds):
        print(f"Start training for seed={seed}") 
        seed_configs = configs.copy()
        seed_configs["seed"] = seed
        configs["split_seed"] = seed
        seed_configs["exp_name"] += f"seed{seed_configs['seed']}"
        
        results[seed] = get_submission(seed_configs)
        print(f"Performance with seed={seed}:", results[seed]["performance_ens"]["Test"])
    
    results["seed_ensemble"] = {}
    results["seed_ensemble"]["test_predictions"] = np.mean([np.array([results[seed]["predictions"][f"fold_{i}"][2] for i in range(n_folds)]).mean(axis=0) for seed in seeds],axis=0) 
    # results["seed_ensemble"]["seed_performances"] = [results[seed]["performance"]["Test"] for seed in seeds]
    
    submission = dataset.pred_to_submission(results["seed_ensemble"]["test_predictions"])
    submission.to_csv(configs["model"]["save_path"]+f"{model_name}_{preprocess_type}_{exp_name}_seedensemble.csv",index=False)
    
    
    if configs["direct_submit"]:
        public_score, private_score, public_rank, public_percentile, private_rank, private_percentile = dataset.submit_data(configs["model"]["save_path"]+f"{model_name}_{preprocess_type}_{exp_name}_seedensemble.csv")

        results["seed_ensemble"]["test_performance"] = {"public_score": public_score, 
                                          "private_score": private_score, 
                                          "public_rank": public_rank, 
                                          "public_percentile": public_percentile, 
                                          "private_rank": private_rank, 
                                          "private_percentile": private_percentile
                                         }
    # else:
    #     print(f"Warning: When n_trials is not None, ensembling over seeds is not implemented. Instead, ensembling is performed over the best performing hyperparameter configurations found using the first seed provided: {seeds[0]}")
    #     results = get_submission(configs,
    #                            n_trials=configs["hpo"]["n_trials"],
    #                            seed=seeds[0],
    #                            exp_name=exp_name,
    #                            direct_submit=configs["direct_submit"]
    #                           )
    return results


def get_submission(configs):
    os.environ["CUDA_VISIBLE_DEVICES"] = configs["model"]["gpus"]
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore")

    exp_name = configs["exp_name"]
    seed = configs["seed"]
    model_name = configs["model"]["model_name"]
    preprocess_type = configs["dataset"]["preprocess_type"]
    n_trials = configs["hpo"]["n_trials"]

    set_seed(seed)
    dataset = get_dataset(configs["dataset"]["dataset_name"], configs["dataset"]["toy_example"])
    model_class = get_model(model_name)
    
    if configs["hpo"]["n_trials"] is not None:
        exp_name += "_tuned"
    
    if not os.path.exists(f'results/{dataset.dataset_name}/{preprocess_type}/{model_name}/{exp_name}/{exp_name}_seed{seed}.pickle'):
        if not os.path.exists(f'results/{dataset.dataset_name}/{preprocess_type}/{model_name}/{exp_name}/'):
            os.makedirs(f'results/{dataset.dataset_name}/{preprocess_type}/{model_name}/{exp_name}/')
        # if not os.path.exists(f'results/{dataset.dataset_name}/submissions'):
        #     os.makedirs(f'results/{dataset.dataset_name}/submissions')

        if preprocess_type=="expert":
            if "cat_method" in configs["dataset"]:
                cat_method = configs["dataset"]["cat_method"]
            else:
                cat_method = None
            
            dataset.load_data()
            if model_class.model_class == "neural_net":
                neural_net = True
            else:
                neural_net = False
            dataset.expert_preprocessing(dataset.X_train, dataset.X_test, dataset.y_train, 
                                         overwrite_existing=configs["dataset"]["overwrite_existing"], 
                                         use_test=configs["dataset"]["use_test"], 
                                         neural_net=neural_net, 
                                         cat_method=cat_method) 
        elif preprocess_type=="standardized":
            dataset.load_data()
            dataset.standardized_preprocessing(dataset.X_train, dataset.X_test, dataset.y_train)
        
        elif preprocess_type=="minimalistic":
            dataset.load_data()
            dataset.minimalistic_preprocessing(dataset.X_train, dataset.X_test, dataset.y_train)

        elif preprocess_type=="openfe":
            dataset.load_data()
            dataset.minimalistic_preprocessing(dataset.X_train, dataset.X_test, dataset.y_train)
            dataset.openfe_preprocessing(dataset.X_train, dataset.X_test, dataset.y_train, overwrite_existing=configs["dataset"]["overwrite_existing"])
        
        else:
            print(f"No preprocessing applied (either because none is selected or because preprocess_type={preprocess_type} is not implemented)")
            dataset.load_data()
        
        # Apply model-specific preprocessing
        if model_class.model_class == "neural_net":
            dataset.neuralnet_preprocessing(dataset.X_train, dataset.X_test, dataset.y_train)
            cat_cardinalities = (np.array([dataset.X_train.iloc[:,dataset.cat_indices].max(),
                                           dataset.X_test.iloc[:,dataset.cat_indices].max()]).max(axis=0)+1).tolist()
        else:
            cat_cardinalities = list(dataset.X_train.iloc[:,dataset.cat_indices].nunique())
            
        print(f"Train dataset has {dataset.X_train.shape[0]} samples and {dataset.X_train.shape[1]} features of which {len(dataset.cat_indices)} are categorical")
        # Update dataset-specific parameters
        configs["model"].update({
            # Dataset-specific Parameters
            "dataset_name": dataset.dataset_name,
            "task_type": dataset.task_type,
            "cont_indices": [i for i in range(dataset.X_train.shape[1]) if i not in dataset.cat_indices],
            "cat_indices": dataset.cat_indices,
            "cat_cardinalities": cat_cardinalities,
            "d_out": 1 if dataset.task_type in ["regression", "binary"] else dataset.num_classes,
            "sample_size": dataset.X_train.shape[0],
            "large_dataset": dataset.large_dataset,
            "eval_metric": dataset.eval_metric_name if dataset.dataset_name!="santander-value-prediction-challenge" else "rmse"
        })   
        if dataset.task_type=="classification":
            configs["model"].update({
                "num_classes": dataset.num_classes
            })   
            
        results = {}
        
        print(f"Train model {model_name}")
        results["performance"] = {}
        results["performance"]["Train"] = {}
        results["performance"]["Val"] = {}
        results["performance"]["Test"] = {}
        results["predictions"] = {}
        results["times"] = {}
        if configs["hpo"]["ensemble_best_trials"] == "auto" or type(configs["hpo"]["ensemble_best_trials"])==int:
            results["performance_ens"] = {}
            results["performance_ens"]["Train"] = {}
            results["performance_ens"]["Val"] = {}
            results["performance_ens"]["Test"] = {}
            results["predictions_ens"] = {}

        configs["model"]["save_path"] = f'results/{dataset.dataset_name}/{preprocess_type}/{model_name}/{exp_name}/'
        configs["model"]["exp_name"] = configs["exp_name"]
        configs["model"]["seed"] = configs["seed"]

        
        if model_name=="AutoGluon":
            configs["hpo"]["ensemble"] = False
            
            model_class = get_model(model_name)
            model = model_class(configs["model"])

            model.fit(dataset.X_train,dataset.y_train)

            y_train_pred = model.predict(dataset.X_train)
            y_test_pred = model.predict(dataset.X_test)

            # Apply dataset-specific preprocessing
            if "minimalistic" in dataset.preprocess_states:
                y_train_eval = dataset.minimalistic_postprocessing(dataset.X_train, dataset.y_train, test=False)
                y_train_pred = dataset.minimalistic_postprocessing(dataset.X_train, y_train_pred, test=False)
                y_test_pred = dataset.minimalistic_postprocessing(dataset.X_test, y_test_pred, test=True)
            # Apply expert-specific preprocessing
            elif "expert" in dataset.preprocess_states:
                y_train_eval = dataset.expert_postprocessing(dataset.X_train, dataset.y_train, test=False)
                y_train_pred = dataset.expert_postprocessing(dataset.X_train, y_train_pred, test=False)
                y_test_pred = dataset.expert_postprocessing(dataset.X_test, y_test_pred, test=True)
            else:
                y_train_eval = dataset.y_train.copy()
            
            if dataset.eval_metric_name=="ams":
                results["performance"]["Train"] = dataset.eval_metric(y_train_eval,y_train_pred,dataset)
            else:
                results["performance"]["Train"] = dataset.eval_metric(y_train_eval,y_train_pred)
            results["performance"]["Val"] = model.model.leaderboard()["score_val"][0]
            results["predictions"] = [y_train_pred, None, y_test_pred]
            results["model_specific_outputs"] = {"leaderboard": model.model.leaderboard()}

            submission = dataset.pred_to_submission(y_test_pred)
            submission.to_csv(configs["model"]["save_path"]+f"{model_name}_{preprocess_type}_{exp_name}.csv",index=False)        
        else:
            folds = dataset.get_cv_folds(dataset.X_train, dataset.y_train, seed=configs["split_seed"])
            
            if configs["dataset"]["toy_example"]:
                folds = folds[:2]

            if configs["model"]["device"] in ["gpu", "cuda"]:
                parallel_tasks = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
                    
                print(f"Use {parallel_tasks} GPUs and parallelize {configs['model']['folds_parallel']} folds on each GPU")
                ray.init(num_cpus=(configs["model"]["folds_parallel"]*parallel_tasks), # Each parallel fold uses own CPU-core 
                         num_gpus=parallel_tasks) # Use all available GPUs as previously specified
            
                if configs["hpo"]["n_trials"] is not None and configs["model"]["model_name"] == "GRANDE":
                    run_fold_parallel = run_fold.options(num_cpus=1, # Each GPU uses one CPU
                                                         num_gpus=0) # Each GPU trains folds_parallel folds
                
                else:
                    run_fold_parallel = run_fold.options(num_cpus=1, # Each GPU uses one CPU
                                                         num_gpus=1/configs["model"]["folds_parallel"]) # Each GPU trains folds_parallel folds
            else:
                # ray.init(num_cpus=(configs["model"]["folds_parallel"]), # Each parallel fold uses own CPU-core 
                #          num_gpus=parallel_tasks) # Use all available GPUs as previously specified
                parallel_tasks = 0
                run_fold_parallel = run_fold.options(num_cpus=np.trunc(configs["model"]["num_threads"]/configs["model"]["folds_parallel"]), # Each fold uses X CPUs
                                                     num_gpus=0) # Each GPU trains folds_parallel folds


            configs["hpo"]["ensemble"] = configs["hpo"]["n_trials"] is not None and configs["hpo"]["ensemble_best_trials"] is not None

            result_by_trial = [run_fold_parallel.remote(
                                               dataset=dataset,
                                               num_fold=num_fold, 
                                                 train=train, 
                                                 val=val,
                                                 fold_configs=configs) for num_fold, (train, val) in enumerate(folds)]
            result_by_trial = ray.get(result_by_trial)
            for num_fold, result_by_fold in enumerate(result_by_trial):
                results[f"fold_{num_fold}"] = result_by_fold
                results["performance"]["Train"][f"fold_{num_fold}"] = result_by_fold["performance"]["Train"]
                results["performance"]["Val"][f"fold_{num_fold}"] = result_by_fold["performance"]["Val"]
                results["predictions"][f"fold_{num_fold}"] = result_by_fold["predictions"]
                results["times"][f"fold_{num_fold}"] = result_by_fold["times"]
                if configs["hpo"]["ensemble"]:
                    results["performance_ens"]["Train"][f"fold_{num_fold}"] = result_by_fold["performance_ens"]["Train"]
                    results["performance_ens"]["Val"][f"fold_{num_fold}"] = result_by_fold["performance_ens"]["Val"]
                    results["predictions_ens"][f"fold_{num_fold}"] = result_by_fold["predictions_ens"]

            ray.shutdown()
            # Todo: Either fix issues with sberbank dataset or remove it entirely
            if dataset.dataset_name == "sberbank-russian-housing-market":
                if dataset.expert_postprocessing:
                    investment = dataset.X_test[dataset.X_test["id"] == 30474]["product_type"].values[0]
                    owner = dataset.X_test[dataset.X_test["id"] == 30475]["product_type"].values[0]
                    invest_rows = dataset.X_test[dataset.X_test["product_type"]==investment].index - 30474
                    owner_rows = dataset.X_test[dataset.X_test["product_type"]==owner].index - 30474
                    y_test_pred_invest = np.array([results["predictions"][f"fold_{i}"][2][invest_rows] for i in range(0, 5)]).mean(axis=0)
                    y_test_pred_owner = np.array([results["predictions"][f"fold_{i}"][2][owner_rows] for i in range(5, 10)]).mean(axis=0)
                    submission = pd.DataFrame(np.hstack((y_test_pred_invest, y_test_pred_owner)), columns=["price_doc"])
                    submission["index"] = np.hstack((invest_rows, owner_rows))
                    submission = submission.sort_values(by="index")
                    submission = submission.drop(columns=["index"])
                    submission["id"] = dataset.X_test["id"]
                    submission.to_csv("submission.csv", index=False)

                #     X_test = dataset.X_test
                #     X_test = pd.merge(X_test, dataset.macro, on=["year", "month", "day"], how="left")
                    
                #     # Ensemble-1: Trend-adjust model to simulate the magic number
                #     y_test_pred = np.zeros(X_test.shape[0])
                #     macro_variables = ["micex_rgbi_tr", "gdp_quart_growth", "oil_urals*gdp_quart_growth"]
                #     for idx, col in enumerate(macro_variables):
                #         macro_var = X_test[col]
                #         y_test_pred += np.array([results["predictions"][f"fold_{i}"][2] * (1 + macro_var) for i in range((idx)*5, (idx+1)*5)]).mean(axis=0)
                #     y_test_pred += np.array([results["predictions"][f"fold_{i}"][2] for i in range(15, 20)]).mean(axis=0)

                #     # Ensemble-2: Remove bad points to adjust the former model
                #     y_test_pred += np.array([results["predictions"][f"fold_{i}"][2] for i in range(20, 25)]).mean(axis=0)
                    
                #     y_test_pred = y_test_pred / 5
            else:
            
                y_test_pred = np.array([results["predictions"][f"fold_{i}"][2] for i in range(len(folds))]).mean(axis=0)
                submission = dataset.pred_to_submission(y_test_pred)
                submission.to_csv(configs["model"]["save_path"]+f"{model_name}_{preprocess_type}_{exp_name}.csv",index=False)        
            if configs["hpo"]["ensemble"]:
                y_test_pred_ens = np.array([results["predictions_ens"][f"fold_{i}"][2] for i in range(len(folds))]).mean(axis=0)
                submission = dataset.pred_to_submission(y_test_pred_ens)
                submission.to_csv(configs["model"]["save_path"] + f"{model_name}_{preprocess_type}_{exp_name}_seed{seed}_hpoensemble.csv",index=False)        


        if configs["direct_submit"]:
            public_score, private_score, public_rank, public_percentile, private_rank, private_percentile = dataset.submit_data(configs["model"]["save_path"]+f"{model_name}_{preprocess_type}_{exp_name}.csv")
            results["performance"]["Test"] = {"public_score": public_score, 
                                              "private_score": private_score, 
                                              "public_rank": public_rank, 
                                              "public_percentile": public_percentile, 
                                              "private_rank": private_rank, 
                                              "private_percentile": private_percentile
                                             }
            if configs["hpo"]["ensemble"] and model_name!="AutoGluon":
                public_score, private_score, public_rank, public_percentile, private_rank, private_percentile = dataset.submit_data(configs["model"]["save_path"] + f"{model_name}_{preprocess_type}_{exp_name}_seed{seed}_hpoensemble.csv")
                results["performance_ens"]["Test"] = {"public_score": public_score, 
                                                  "private_score": private_score, 
                                                  "public_rank": public_rank, 
                                                  "public_percentile": public_percentile, 
                                                  "private_rank": private_rank, 
                                                  "private_percentile": private_percentile
                                                 }
        
        with open(configs["model"]["save_path"]+f'{exp_name}_seed{seed}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        
    else:
        print(f'Results at "results/{dataset.dataset_name}/{preprocess_type}/{model_name}/{exp_name}/{exp_name}_seed{seed}.pickle" already exist and are loaded')
        with open(f'results/{dataset.dataset_name}/{preprocess_type}/{model_name}/{exp_name}/{exp_name}_seed{seed}.pickle', 'rb') as handle:
            results = pickle.load(handle)

    return results




@ray.remote(num_cpus=1, num_gpus=1)
def run_fold(dataset,
             num_fold, 
             train, 
             val,
             fold_configs):
    if fold_configs["model"]["model_name"]=="GRANDE":
        from GRANDE import GRANDE

    print(f"Start Training for fold {num_fold}")
        
    exp_name = fold_configs["exp_name"]+f"_{num_fold}"
    seed = fold_configs["seed"]
    
    set_seed(seed)
    
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore")
    
    res = {}
    res["performance"] = {}
    res["times"] = {}
    
    # if dataset.dataset_name == "sberbank-russian-housing-market":
    #     if num_fold in range(0, 15):
    #         X_train_macro = pd.merge(dataset.X_train, dataset.macro, on=["year", "month", "day"], how="left")
    #         X_train_fold = dataset.X_train.iloc[train]
    #         X_val_fold = dataset.X_train.iloc[val]
    #         if num_fold in range(0, 5):
    #             y_train_fold = dataset.y_train.iloc[train] / (1 + X_train_macro["micex_rgbi_tr"].iloc[train])
    #             y_val_fold = dataset.y_train.iloc[val] / (1 + X_train_macro["micex_rgbi_tr"].iloc[val])
    #         elif num_fold in range(5, 10):
    #             y_train_fold = dataset.y_train.iloc[train] / (1 + X_train_macro["gdp_quart_growth"].iloc[train])
    #             y_val_fold = dataset.y_train.iloc[val] / (1 + X_train_macro["gdp_quart_growth"].iloc[val])
    #         else:
    #             y_train_fold = dataset.y_train.iloc[train] / (1 + X_train_macro["oil_urals*gdp_quart_growth"].iloc[train])
    #             y_val_fold = dataset.y_train.iloc[val] / (1 + X_train_macro["oil_urals*gdp_quart_growth"].iloc[val])
    #     else:
    #         X_train_fold = dataset.X_train.iloc[train]
    #         y_train_fold = dataset.y_train.iloc[train]
    #         X_val_fold = dataset.X_train.iloc[val]
    #         y_val_fold = dataset.y_train.iloc[val]
    # else:
    #     X_train_fold = dataset.X_train.iloc[train]
    #     y_train_fold = dataset.y_train.iloc[train]
    #     X_val_fold = dataset.X_train.iloc[val]
    #     y_val_fold = dataset.y_train.iloc[val]

    X_train_fold = dataset.X_train.iloc[train]
    y_train_fold = dataset.y_train.iloc[train]
    X_val_fold = dataset.X_train.iloc[val]
    y_val_fold = dataset.y_train.iloc[val]

    fold_configs["model"]["save_path"] += f"/fold_{num_fold}"
    if not os.path.exists(fold_configs["model"]["save_path"]):
        os.makedirs(fold_configs["model"]["save_path"])

    n_trials = fold_configs["hpo"]["n_trials"]
    
    if n_trials is not None:
        start = time.time()
        print(f"Run HPO for {n_trials} trials")
        study = tune_hyperparameters(
            X_train_fold, y_train_fold, 
            eval_set = [(X_val_fold, y_val_fold)], 
            X_test=dataset.X_test,
            dataset=dataset,
            configs=fold_configs
        )                    
        
        fold_configs["model"]["hyperparameters"] = study.best_params
        
        end = time.time()
        res["times"]["mean_trial_time"] = (end-start)/60/n_trials

        print(f'Mean time per trial: {res["times"]["mean_trial_time"]}')

        y_train_pred_fold, y_val_pred_fold, y_test_pred_fold = study.best_trial.user_attrs["predictions"]
        if "neuralnet" in dataset.preprocess_states:
            y_train_fold = dataset.neuralnet_postprocessing(X_train_fold, y_train_fold.values.reshape(-1,1))
            y_val_fold = dataset.neuralnet_postprocessing(X_val_fold, y_val_fold.values.reshape(-1,1))
        # Apply expert-specific preprocessing
        if "expert" in dataset.preprocess_states:
            y_train_fold = dataset.expert_postprocessing(X_train_fold, y_train_fold, test=False)
            y_val_fold = dataset.expert_postprocessing(X_val_fold, y_val_fold, test=False)           
        if "minimalistic" in dataset.preprocess_states:
            y_train_fold = dataset.minimalistic_postprocessing(X_train_fold, y_train_fold)
            y_val_fold = dataset.minimalistic_postprocessing(X_val_fold, y_val_fold)

        if fold_configs["hpo"]["ensemble"]:
            if fold_configs["hpo"]["ensemble_best_trials"] == "auto" and n_trials>=3: 
                # Obtain best val performances and predictions from trials
                trials = study.get_trials()
                trial_performances = [trials[i].values for i in range(n_trials)]
                if dataset.eval_metric_direction=="maximize": 
                    trial_performances = np.array([i[0] if i is not None else -np.inf  for i in trial_performances])
                if dataset.eval_metric_direction=="minimize": 
                    trial_performances = np.array([i[0] if i is not None else np.inf  for i in trial_performances])
                
                best_trial_performances_idx = np.argsort(trial_performances)
                if dataset.eval_metric_direction=="maximize": 
                    best_trial_performances_idx = np.argsort(trial_performances)[::-1]
                
                val_trial_predictions = np.array([trials[i].user_attrs["predictions"][1] for i in range(n_trials)])
    
                curr_best_trial_perf = np.round(trial_performances[best_trial_performances_idx[0]],4)
                print(f"Best performance prior ensembling: {curr_best_trial_perf}")

                # Limit max. no. of trials for ensembling to 10 as deploying too many models would be impractical in real applications
                max_trials = np.min([11,n_trials])
                
                # Get ensemble performances
                if dataset.eval_metric_name=="ams":
                    hpo_ensembles = [dataset.eval_metric(pd.Series(y_val_fold.ravel(),index=dataset.y_train.iloc[val].index),val_trial_predictions[best_trial_performances_idx[:used_trials]].mean(axis=0),dataset) for used_trials in range(2,max_trials)]
                else:
                    hpo_ensembles = [dataset.eval_metric(y_val_fold,val_trial_predictions[best_trial_performances_idx[:used_trials]].mean(axis=0)) for used_trials in range(2,max_trials)]
                
                if dataset.eval_metric_direction=="maximize": 
                    best_ensemble = np.argmax(hpo_ensembles)
                else:
                    best_ensemble = np.argmin(hpo_ensembles)
                
                y_val_pred_fold_ens = val_trial_predictions[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                if dataset.eval_metric_name=="ams":
                    ensemble_perf = dataset.eval_metric(pd.Series(y_val_fold.ravel(),index=dataset.y_train.iloc[val].index),y_val_pred_fold_ens, dataset)
                else:
                    ensemble_perf = dataset.eval_metric(y_val_fold,y_val_pred_fold_ens)
                
                if dataset.eval_metric_direction=="maximize": 
                    condition = np.round(ensemble_perf,4)>curr_best_trial_perf
                else:
                    condition = np.round(ensemble_perf,4)<curr_best_trial_perf
                
                if condition:
                
                    for used_trials in range(2,2+best_ensemble+1):
                        y_val_pred_fold_ens = val_trial_predictions[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                        if dataset.eval_metric_name=="ams":
                            ensemble_perf = dataset.eval_metric(pd.Series(y_val_fold.ravel(),index=dataset.y_train.iloc[val].index),y_val_pred_fold_ens, dataset)
                        else:
                            ensemble_perf = dataset.eval_metric(y_val_fold,y_val_pred_fold_ens)
                        
                        if used_trials==2+best_ensemble:
                            print(f"Final Ensemble using top {used_trials} HP settings: {ensemble_perf}")
                        else:
                            print(f"Ensemble using top {used_trials} HP settings: {ensemble_perf}")
                    
                    y_train_pred_fold_ens = np.array([trials[i].user_attrs["predictions"][0] for i in range(n_trials)])[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                    y_val_pred_fold_ens = np.array([trials[i].user_attrs["predictions"][1] for i in range(n_trials)])[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                    y_test_pred_fold_ens = np.array([trials[i].user_attrs["predictions"][2] for i in range(n_trials)])[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                else:
                    print(f"Using top HP settings does not improve the ensemble")
                    y_train_pred_fold_ens, y_val_pred_fold_ens, y_test_pred_fold_ens = study.best_trial.user_attrs["predictions"]
                    
                print("--------------------")            
                
            elif type(fold_configs["hpo"]["ensemble_best_trials"])==2: 
                trials = study.get_trials()

                trial_performances = [trials[i].values for i in range(n_trials)]
                if dataset.eval_metric_direction=="maximize": 
                    trial_performances = np.array([i[0] if i is not None else -np.inf  for i in trial_performances])
                    best_trial_performances = np.argsort(trial_performances)[::-1]

                if dataset.eval_metric_direction=="minimize": 
                    trial_performances = np.array([i[0] if i is not None else np.inf  for i in trial_performances])
                    best_trial_performances = np.argsort(trial_performances)
                
                y_train_pred_fold_ens = np.array([trials[i].user_attrs["predictions"][0] for i in range(n_trials)])[best_trial_performances].mean(axis=0)
                y_val_pred_fold_ens = np.array([trials[i].user_attrs["predictions"][1] for i in range(n_trials)])[best_trial_performances].mean(axis=0)
                y_test_pred_fold_ens = np.array([trials[i].user_attrs["predictions"][2] for i in range(n_trials)])[best_trial_performances].mean(axis=0)
            else: 
                print("Not enough trials for ensembling - disable ensembling over trials.")
                fold_configs["hpo"]["ensemble"] = False

            res["performance_ens"] = {}
            if dataset.eval_metric_name=="ams":
                res["performance_ens"]["Train"] = dataset.eval_metric(pd.Series(y_train_fold.ravel(),index=dataset.y_train.iloc[train].index),y_train_pred_fold_ens,dataset)
                res["performance_ens"]["Val"] = dataset.eval_metric(pd.Series(y_val_fold.ravel(),index=dataset.y_train.iloc[val].index),y_val_pred_fold_ens,dataset)
            else:
                res["performance_ens"]["Train"] = dataset.eval_metric(y_train_fold,y_train_pred_fold_ens)
                res["performance_ens"]["Val"] = dataset.eval_metric(y_val_fold,y_val_pred_fold_ens)
            res["predictions_ens"] = [y_train_pred_fold, y_val_pred_fold, y_test_pred_fold_ens]    
            
    
    else:
        # if "hyperparameters" in fold_configs["model"]:
        #     if isinstance(fold_configs["model"]["hyperparameters"], list):
        #         fold_configs["model"]["hyperparameters"] = fold_configs["model"]["hyperparameters"][num_fold]
        #         print(f'Hyperparameters: {fold_configs["model"]["hyperparameters"]}')
        
        # Train model
        model_class = get_model(fold_configs["model"]["model_name"])
        
        start = time.time()
        model = model_class(params=fold_configs["model"])

        model.fit(X_train_fold,y_train_fold,
                  [(X_val_fold, y_val_fold)],
                 )
        
        end = time.time()
        res["times"]["train_time"] = (end-start)/60

        start = time.time()
        y_train_pred_fold = model.predict(X_train_fold)
        y_val_pred_fold = model.predict(X_val_fold)
        y_test_pred_fold = model.predict(dataset.X_test)
        
        end = time.time()
        res["times"]["test_time"] = (end-start)/60
        
        print(f'Fit+Predict Time: {res["times"]["train_time"]+res["times"]["test_time"]}')
 
        # Apply model-specific postprocessing
        # Apply model-specific postprocessing
        if "neuralnet" in dataset.preprocess_states:
            y_train_fold = dataset.neuralnet_postprocessing(X_train_fold, y_train_fold)
            y_val_fold = dataset.neuralnet_postprocessing(X_val_fold, y_val_fold)
            y_train_pred_fold = dataset.neuralnet_postprocessing(X_train_fold, y_train_pred_fold)
            y_val_pred_fold = dataset.neuralnet_postprocessing(X_val_fold, y_val_pred_fold)
            y_test_pred_fold = dataset.neuralnet_postprocessing(dataset.X_test, y_test_pred_fold)
        
        # Apply expert-specific preprocessing
        if "expert" in dataset.preprocess_states:
            y_train_fold = dataset.expert_postprocessing(X_train_fold, y_train_fold, test=False)
            y_val_fold = dataset.expert_postprocessing(X_val_fold, y_val_fold, test=False)
            y_train_pred_fold = dataset.expert_postprocessing(X_train_fold, y_train_pred_fold, test=False)
            y_val_pred_fold = dataset.expert_postprocessing(X_val_fold, y_val_pred_fold, test=False)
            y_test_pred_fold = dataset.expert_postprocessing(dataset.X_test, y_test_pred_fold, test=True)
        elif "minimalistic" in dataset.preprocess_states:
            y_train_fold = dataset.minimalistic_postprocessing(X_train_fold, y_train_fold)
            y_val_fold = dataset.minimalistic_postprocessing(X_val_fold, y_val_fold)
            y_train_pred_fold = dataset.minimalistic_postprocessing(X_train_fold, y_train_pred_fold)
            y_val_pred_fold = dataset.minimalistic_postprocessing(X_val_fold, y_val_pred_fold)
            y_test_pred_fold = dataset.minimalistic_postprocessing(dataset.X_test, y_test_pred_fold)
    
    # Specific implementation for the Higgs-Boson dataset
    if dataset.eval_metric_name=="ams":
        res["performance"]["Train"] = dataset.eval_metric(pd.Series(y_train_fold.ravel(),index=dataset.y_train.iloc[train].index),y_train_pred_fold,dataset)
        res["performance"]["Val"] = dataset.eval_metric(pd.Series(y_val_fold.ravel(),index=dataset.y_train.iloc[val].index),y_val_pred_fold,dataset)
    else:
        res["performance"]["Train"] = dataset.eval_metric(y_train_fold,y_train_pred_fold)
        res["performance"]["Val"] = dataset.eval_metric(y_val_fold,y_val_pred_fold)
    res["predictions"] = [y_train_pred_fold, y_val_pred_fold, y_test_pred_fold]

    print(f'Val Performance fold {num_fold}: {res["performance"]["Val"]}') 
    if fold_configs["hpo"]["ensemble"]:
        print(f'Val Ensemble Performance fold {num_fold}: {res["performance_ens"]["Val"]}')
    
    return res



def tune_hyperparameters(X_train_tune, y_train_tune, # Dataset
                         eval_set, # Dataset
                         X_test,
                         dataset,
                         configs): # External

    exp_name = configs["exp_name"]
    seed = configs["seed"]
    
    if not os.path.exists(configs["model"]["save_path"]):
        os.makedirs(configs["model"]["save_path"])
    
    X_val_tune, y_val_tune = eval_set[0]
    
    set_seed(seed)

    eval_metric, eval_metric_direction = get_metric(dataset.eval_metric_name)

    def objective(trial, study):
        model_class = get_model(configs["model"]["model_name"])
        
        configs["model"]["hyperparameters"] = model_class.get_optuna_hyperparameters(trial, 
                                                                                     n_features = X_train_tune.shape[1], 
                                                                                     large_dataset = configs["model"]["large_dataset"],
                                                                                     dataset_name = configs["model"]["dataset_name"],
                                                                                     sample_size = configs["model"]["sample_size"],
                                                                                    )
        
        print(configs["model"]["hyperparameters"])
        # try:
        if configs["model"]["model_name"]=="GRANDE":
            @ray.remote(num_cpus=1, num_gpus=1/configs["model"]["folds_parallel"])
            def run_trial(configs, X_train_tune,y_train_tune,X_val_tune, y_val_tune, X_test):
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                start = time.time()
                model = model_class(params=configs["model"])
        
                model.fit(X_train_tune.copy(),y_train_tune.copy(),
                          [(X_val_tune.copy(), y_val_tune.copy())],
                         )
                
                end = time.time()
                train_time = (end-start)/60
        
                start = time.time()
                y_train_pred = model.predict(X_train_tune.copy())
                y_val_pred = model.predict(X_val_tune.copy())
                y_test_pred = model.predict(X_test.copy())
                end = time.time()
                test_time = (end-start)/60
    
                return y_train_pred, y_val_pred, y_test_pred, train_time, test_time
            
            X_train_tune_ray = ray.put(X_train_tune)
            y_train_tune_ray = ray.put(y_train_tune)
            X_val_tune_ray = ray.put(X_val_tune) 
            y_val_tune_ray = ray.put(y_val_tune) 
            X_test_ray = ray.put(dataset.X_test)
            result_by_trial = run_trial.remote(configs, X_train_tune_ray,y_train_tune_ray,X_val_tune_ray, y_val_tune_ray, X_test_ray)
            y_train_pred, y_val_pred, y_test_pred, train_time, test_time = ray.get(result_by_trial)
        else:
            start = time.time()
            model = model_class(params=configs["model"])

            model.fit(X_train_tune,y_train_tune,
                      [(X_val_tune, y_val_tune)],
                      )
            
            end = time.time()
            train_time = (end-start)/60
    
            start = time.time()
            y_train_pred = model.predict(X_train_tune)
            y_val_pred = model.predict(X_val_tune)
            y_test_pred = model.predict(dataset.X_test)
            end = time.time()
            test_time = (end-start)/60
        
        # Apply model-specific postprocessing
        if "neuralnet" in dataset.preprocess_states:
            y_train_eval = dataset.neuralnet_postprocessing(X_train_tune, y_train_tune.values.reshape(-1,1))
            y_val_eval = dataset.neuralnet_postprocessing(X_val_tune, y_val_tune.values.reshape(-1,1))
            y_train_pred = dataset.neuralnet_postprocessing(X_train_tune, y_train_pred)
            y_val_pred = dataset.neuralnet_postprocessing(X_val_tune, y_val_pred)
            y_test_pred = dataset.neuralnet_postprocessing(dataset.X_test, y_test_pred)
        else:
            y_train_eval = y_train_tune.copy()
            y_val_eval = y_val_tune.copy()
        # Apply expert-specific preprocessing
        if "expert" in dataset.preprocess_states:
            y_train_eval = dataset.expert_postprocessing(X_train_tune, y_train_eval, test=False)
            y_val_eval = dataset.expert_postprocessing(X_val_tune, y_val_eval, test=False)           
            y_train_pred = dataset.expert_postprocessing(X_train_tune, y_train_pred, test=False)
            y_val_pred = dataset.expert_postprocessing(X_val_tune, y_val_pred, test=False)
            y_test_pred = dataset.expert_postprocessing(dataset.X_test, y_test_pred, test=True)
        if "minimalistic" in dataset.preprocess_states:
            y_train_eval = dataset.minimalistic_postprocessing(X_train_tune, y_train_eval)
            y_val_eval = dataset.minimalistic_postprocessing(X_val_tune, y_val_eval)
            y_train_pred = dataset.minimalistic_postprocessing(X_train_tune, y_train_pred)
            y_val_pred = dataset.minimalistic_postprocessing(X_val_tune, y_val_pred)
            y_test_pred = dataset.minimalistic_postprocessing(dataset.X_test, y_test_pred)
        
        if dataset.eval_metric_name=="ams":
            train_score = eval_metric(pd.Series(y_train_eval.ravel(),index=y_train_tune.index),y_train_pred, dataset)
            val_score = eval_metric(pd.Series(y_val_eval.ravel(),index=y_val_tune.index),y_val_pred, dataset)
        else:
            train_score = eval_metric(y_train_eval,y_train_pred)
            val_score = eval_metric(y_val_eval,y_val_pred)
        # except:
        #     print(f"An exception occurred in Trial {trial.number}")
        #     if eval_metric_direction == "maximize":
        #         train_score = -np.inf
        #         val_score = -np.inf
        #     else:
        #         train_score = np.inf
        #         val_score = np.inf
        #     y_train_pred = np.zeros(X_train_tune.shape[0])
        #     y_val_pred = np.zeros(X_val_tune.shape[0])
        #     y_test_pred = np.zeros(dataset.X_test.shape[0])
        #     train_time = -1
        #     test_time = -1

        trial.set_user_attr("predictions", [y_train_pred,y_val_pred,y_test_pred])
        trial.set_user_attr("train_performance", train_score)
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("test_time", test_time)

        if (trial.number % study.user_attrs["save_interval"])==0:
            joblib.dump(study, study.user_attrs["save_path"])

        return val_score

    def wrapped_objective(trial):
        return objective(trial, study)

    if not os.path.exists(f'{configs["model"]["save_path"]}/{exp_name}_study.pkl'):

        # Create a study object and optimize the objective function
        sampler = optuna.samplers.TPESampler(seed=seed,
                                             n_startup_trials=configs["hpo"]["n_startup_trials"],
                                             multivariate=True,
                                             warn_independent_sampling=False
                                            ) 

        study = optuna.create_study(direction=eval_metric_direction,
                                    sampler=sampler,
                                   )
        study.set_user_attr("save_path", f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
        study.set_user_attr("save_interval", configs["hpo"]["save_interval"])
        
        study.optimize(wrapped_objective, 
                       n_trials=configs["hpo"]["n_trials"], 
                       gc_after_trial=True)

        joblib.dump(study, study.user_attrs["save_path"])

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

    else:
        print(f"Results '{configs['model']['save_path']}/{exp_name}_study.pkl' already exist and will be loaded.")  
        
        study = joblib.load(f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
        study.set_user_attr("save_path", f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
        study.set_user_attr("save_interval", configs["hpo"]["save_interval"])
        try:
            print(f"Best trial until now: {study.best_trial.value} with parameters: {study.best_trial.params}")
        except:
            print("No trials finished yet")
        if configs["hpo"]["n_trials"]>len(study.trials):
            study.optimize(wrapped_objective, 
                           n_trials=configs["hpo"]["n_trials"]-len(study.trials), 
                           gc_after_trial=True)
            joblib.dump(study, f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
    
            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)            
            
    
    return study
