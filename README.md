# A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data

This repository accompanies the paper "A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data" [https://arxiv.org/abs/2407.02112](https://arxiv.org/abs/2407.02112). It contains the proposed evaluation framework, consisting of datasets from Machine Learning Competitions and expert-level solutions for each task. This evaluation framework enables researchers to evaluate machine learning models with realistic preprocessing pipelines beyond overly standardized evaluation setups typically used in academia.

![Figure 1: Overview of our results.](figures/all_results.pdf)


If you use our evaluation framework, please cite the following bib entry:

```bibtex
@article{tschalzev2024data,
  title={A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data},
  author={Tschalzev, Andrej and Marton, Sascha and L{\"u}dtke, Stefan and Bartelt, Christian and Stuckenschmidt, Heiner},
  journal={arXiv preprint arXiv:2407.02112},
  year={2024}
}
```

## Quick start (Linux)
1. Create a new Python 3.11.7 environment and install 'requirements.txt'. (Currently, only Linux systems are supported.)
2. A Kaggle account and the KAggle API are required to use our framwork. If necessary, create a Kaggle account. Then follow https://www.kaggle.com/docs/api to create an API token from Kaggle, download it and place the 'kaggle.json' file containing your API token in the directory '~/.kaggle/'.
3. Run download_datasets.py. If necessary, visit the Kaggle competition websites, accept the competition rules, and rerun the script. More details below.
4. Run run_experiment.py.

  
### Datasets
To download the datasets for the evaluation framework, run download_datasets.py. Before downloading, the Kaggle API needs to be correctly configured (see https://www.kaggle.com/docs/api). Furthermore, for each competition, the competition rules need to be accepted.

The following competitions are included:

- https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing
- https://www.kaggle.com/competitions/santander-value-prediction-challenge
- https://www.kaggle.com/competitions/amazon-employee-access-challenge
- https://www.kaggle.com/competitions/otto-group-product-classification-challenge
- https://www.kaggle.com/competitions/santander-customer-satisfaction
- https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management
- https://www.kaggle.com/competitions/santander-customer-transaction-prediction
- https://www.kaggle.com/competitions/homesite-quote-conversion
- https://www.kaggle.com/competitions/ieee-fraud-detection
- https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction


## Reproducing the Results
To run an experiment, 'run_experiment.py' can be used and adapted. The script loads a predefined configuration file and runs the modeling pipeline for the specified model, hyperparameter optimization (HPO) regime, dataset, and preprocessing configuration. The configs directory contains configuration files for each model currently implemented in the framework and one example configuration. The example configuration runs a CatBoost model with default hyperparameters on the mercedes-benz-greener-manufacturing (MBGM) dataset with standardized (minimalistic) preprocessing. The model config files include the configuration for the model with extensive HPO on the MBGM dataset with standardized preprocessing and can be adapted to reproduce all reported settings in the paper. To achieve this, we provide an overview of the most important configuration choices:

- [dataset][dataset_name]: One of {'mercedes-benz-greener-manufacturing', 'santander-value-prediction-challenge', 'amazon-employee-access-challenge', 'otto-group-product-classification-challenge', 'santander-customer-satisfaction', 'bnp-paribas-cardif-claims-management', 'santander-customer-transaction-prediction', 'homesite-quote-conversion', 'ieee-fraud-detection', 'porto-seguro-safe-driver-prediction'}
- [dataset][preprocess_type]: One of {expert, minimalistic, null} - 'minimalistic' corresponds to the standardized preprocessing pipeline, 'expert' to the feature engineering pipeline
- [dataset][use_test]: One of {false, true} - If true, the test-time adaptation pipeline is executed. If false, the feature engineering pipeline is executed. The parameter only takes effect whenever the preprocess_type is 'expert'
- [model][model_name]: One of {XGBoost, CatBoost, LightGBM, ResNet, FTTRansformer, MLP-PLR, GRANDE, AutoGluon}
- [model][gpus]: Which GPUs in cluster to use
- [model][folds_parallel]: How many CV folds to train in parallel on one GPU. For small datasets, running cross-validation folds in parallel greatly reduces the overall training time. Consider increasing the single-gpu parallelization depending on your hardware.
- [hpo][n_trials]: No. of HPO trials. 100 for the extensive HPO setting, 20 for light HPO, and null for default.
- [hpo][n_startup_trials]: No. random search warmup iterations. We set this parameter to 20 whenever we use HPO.

The notebooks Final_Evaluation.ipynb and Final_Evaluation_orig_metric.ipynb gather all our results and execute the evaluation reported in the paper.

## Contributing
### Datasets
To add new datasets, the following steps are required:
- A new class in the datasets.py file following the instructions in the BaseDataset class header
- Add the class to the get_datasets function in datasets.py 
- Addition of the dataset to download_datasets.py

Currently, the only supported platform is Kaggle as it offers an API that allows for easy post-competition submissions. We appreciate any contributions allowing us to integrate datasets from other platforms. Contributions of datasets without expert preprocessing are also appreciated.

### Models
To add new models, the following steps are required:
- A new class in the models.py file that can be initialised with a params dictionary and includes the following functions: fit, predict, get_default_hyperparameters, get_optuna_hyperparameters
- Add the class to the get_models function in models.py 
- Add a configs/{new_model}.yaml file including information about configurable, but not tuned hyperparameters


## Citation

```bibtex
@article{tschalzev2024data,
  title={A Data-Centric Perspective on Evaluating Machine Learning Models for Tabular Data},
  author={Tschalzev, Andrej and Marton, Sascha and L{\"u}dtke, Stefan and Bartelt, Christian and Stuckenschmidt, Heiner},
  journal={arXiv preprint arXiv:2407.02112},
  year={2024}
}
```
