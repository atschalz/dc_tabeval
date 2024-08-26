from kaggle.api.kaggle_api_extended import KaggleApi
import argparse
import os
import zipfile

dataset_list = [
    "mercedes-benz-greener-manufacturing", 
    "amazon-employee-access-challenge",
    "santander-value-prediction-challenge", 
    "santander-customer-transaction-prediction", 
    "ieee-fraud-detection",
    "santander-customer-satisfaction",
    "porto-seguro-safe-driver-prediction",
    # "sberbank-russian-housing-market",
    # "higgs-boson",
    # "walmart-recruiting-trip-type-classification",
    # "allstate-claims-severity",
    "bnp-paribas-cardif-claims-management",
    # "talkingdata-mobile-user-demographics",
    # "predicting-red-hat-business-value",
    # "restaurant-revenue-prediction",
    # "zillow-prize-1",
    "otto-group-product-classification-challenge",
    # "springleaf-marketing-response",
    # "prudential-life-insurance-assessment",
    # "microsoft-malware-prediction",
    "homesite-quote-conversion",
                ]

def extract_zip(zip_path, extract_to):
    """Extracts a zip file to a specified location and then deletes the zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)

def find_and_extract_all_zips(start_path):
    """Finds all zip files in a directory and its subdirectories, extracts them, and removes the zip files."""
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                try:
                    extract_zip(zip_path, root)
                except:
                    print(f"Not able to extract {zip_path}")

if __name__ == "__main__":
    
    for dataset_name in dataset_list:
        # Specify download path and create it if necessary
        download_path = f"./datasets/{dataset_name}/raw/"
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        if not os.path.exists(f"./datasets/{dataset_name}/processed/"):
            os.makedirs(f"./datasets/{dataset_name}/processed/")
        if len(os.listdir(download_path)) == 0:
            # Download data
            os.system(f"kaggle competitions download -c {dataset_name} -p {download_path}")
    
            if os.path.exists(download_path+f"{dataset_name}.zip"):
                if dataset_name == "walmart-recruiting-trip-type-classification":
                    # Extract the main zip file
                    extract_zip(download_path+f"{dataset_name}.zip", download_path)
                    
                    # Now, find and extract all nested zip files and remove them
                    find_and_extract_all_zips(download_path)        

                    print("Download test data separately")
                    os.system(f"kaggle datasets download -d thitchen/walmart-trip-type-test -p {download_path}")
                    extract_zip(download_path+"walmart-trip-type-test.zip", download_path)
                    os.system(f"rm {download_path}test.csv.zip")
                    
                    print(f"Dataset '{dataset_name}' downloaded successfully to '{download_path}'.")                    
                else:
                    # Extract the main zip file
                    extract_zip(download_path+f"{dataset_name}.zip", download_path)
                    
                    # Now, find and extract all nested zip files and remove them
                    find_and_extract_all_zips(download_path)        
                    
                    print(f"Dataset '{dataset_name}' downloaded successfully to '{download_path}'.")
            else:
                print(f"Accept competition rules at: https://www.kaggle.com/competitions/{dataset_name}/rules")
        else:
            print(f"{download_path} is not empty - dataset might already be downloaded. If not, clean the directory and rerun this function.")


            