import warnings
import pickle
import os
import argparse
import json
from functools import reduce
from itertools import product
import time
from datetime import datetime
import numpy as np
import copy

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from experiment_utils_mc import experiment, product_dict, extract_best_results, set_seed, mislabel_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run CNN experiments with various hyperparameters")
    
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON format)")
    parser.add_argument("--batchsize", type=int, nargs='+', help="Batch sizes for training")
    parser.add_argument("--lr", type=float, nargs='+', help="Learning rates to try")
    parser.add_argument("--n_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--l2_sum_lambda", type=float, nargs='+', help="L2 summation regularization lambda values")
    parser.add_argument("--l2_mul_lambda", type=float, nargs='+', help="L2 multiplication regularization lambda values")
    parser.add_argument("--wn", type=str, nargs='+', help="Weight normalization (e.g., 'None', 'default', '1.75')")
    parser.add_argument("--depth_normalization", type=str, nargs='+', help="Normalize the penalty term by the number of layers")
    parser.add_argument("--features_normalization", type=str, nargs='+', help="Normalize the penalty term by the number of features")
    parser.add_argument("--batch_norm", type=str, nargs='+', help="Use batch normalization (e.g., 'True', 'False')")
    parser.add_argument("--bias", type=str, nargs='+', help="Use bias (e.g., 'True', 'False')")
    parser.add_argument("--opt_name", type=str, nargs='+', help="Optimizer name (e.g., 'adam', 'sgd')")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps", "cuda:0", "cuda:1", "cuda:2", "cuda:3"], help="Device to run the training on ('auto', 'cpu', 'cuda', 'mps')")
    parser.add_argument("--seed", type=int, nargs='+', help="Random seed values to try")
    parser.add_argument("--init_scaling", type=float, nargs='+', help="Scale factor(s) of weights after init (between 0 and 1, etc.). If not provided, no scaling.")
    parser.add_argument("--mislabel_percentage", type=float, nargs='+', help="Percentage(s) of labels to mislabel (between 0 and 1). If not provided, no mislabeling.")
    
    # ------------------------------------------------------------------------------------------

    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Update args with config file values
        for key, value in config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
            else:
                # If the argument is a list and the command-line argument is None, use the config value
                if isinstance(value, list) and getattr(args, key) == []:
                    setattr(args, key, value)
                # If the argument is not set via command-line, use the config value
                elif getattr(args, key) is None:
                    setattr(args, key, value)

    args.wn = [None if wn == "None" else float(wn) if wn.replace('.', '', 1).isdigit() else wn for wn in args.wn]
        
    args.features_normalization = [None if f_n == "None" else f_n for f_n in args.features_normalization]
    
    # Ensure 'seed' is a list of integers
    if isinstance(args.seed, list):
        args.seed = list(map(int, args.seed))
    else:
        args.seed = [int(args.seed)] if args.seed else [1]

    # Ensure 'batchsize' is a list of integers
    if isinstance(args.batchsize, list):
        args.batchsize = list(map(int, args.batchsize))
    else:
        args.batchsize = [int(args.batchsize)]

    # If user didn't pass init_scaling or mislabel_percentage, default to empty lists
    if not args.init_scaling:
        args.init_scaling = []
    if not args.mislabel_percentage:
        args.mislabel_percentage = []

    return args

def main():
    args = parse_args()
    warnings.simplefilter(action="ignore", category=FutureWarning)

    print("Available CUDA devices: ", torch.cuda.device_count())
    print("Is CUDA available? ", torch.cuda.is_available())

    # Set device based on command line argument
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Running on {device.type.upper()}")

    # Configuration dictionary
    cfg = {
        'batchsize': args.batchsize,
        'lr': args.lr,
        'n_epochs': args.n_epochs,
        'l2_sum_lambda': args.l2_sum_lambda,
        'l2_mul_lambda': args.l2_mul_lambda,
        'wn': args.wn,
        'depth_normalization': args.depth_normalization,
        'features_normalization': args.features_normalization,
        'batch_norm': args.batch_norm,
        'bias': args.bias,
        'device': device,
        'opt_name': args.opt_name,
        'seed': args.seed,
        'init_scaling': args.init_scaling,
        'mislabel_percentage': args.mislabel_percentage,
    }

    print("Parameters:", cfg)

    # Define a transformation for the dataset (normalization, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Modify the transformation and dataset loading to filter only the first two classes (0 and 1)
    def filter_cifar10_classes(dataset, classes):
        class_mapping = {classes[0]: -1, classes[1]: 1}  # Map first class to -1, second class to 1
        idx = [i for i, label in enumerate(dataset.targets) if label in classes]
        dataset.targets = [class_mapping[dataset.targets[i]] for i in idx]
        dataset.data = dataset.data[idx]
        return dataset

    # Load the CIFAR-10 dataset and filter to use only classes 3 and 4 (binary classification)
    trainset_original = datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform)

    # Filter for only classes 3 and 4
    trainset_original = filter_cifar10_classes(trainset_original, classes=[3, 4])
    testset = filter_cifar10_classes(testset, classes=[3, 4])

    print("Unique labels in the dataset:", set(trainset_original.targets))

    # Limit to 10,000 training images and 2,000 testing images
    trainset_original.data = trainset_original.data[:10000]
    trainset_original.targets = trainset_original.targets[:10000]
    testset.data = testset.data[:2000]
    testset.targets = testset.targets[:2000]

    # Define parameter grids for different regularization settings
    param_grid = {
        "batchsize": cfg['batchsize'],  # Include batchsize in param_grid
        "in_channels": [3],  # CIFAR-10 has 3 color channels (RGB)
        "bias": [cfg['bias']],
        "n_epochs": [cfg['n_epochs']],
        "opt_name": [cfg['opt_name']],
        "lr": cfg['lr'],
        "batch_norm": cfg['batch_norm'],
        "seed": cfg['seed'],  # Include seed in the parameter grid
    }

    # Handle multiple 'wn' values directly
    wn_values = cfg['wn'] if cfg['wn'] else [None]

    # Convert lambda values to float to ensure consistency
    l2_sum_lambda = list(map(float, cfg['l2_sum_lambda']))
    l2_mul_lambda = list(map(float, cfg['l2_mul_lambda']))
    
    def strtobool(v):
        return v.lower() in ["yes", "true", "t", "1"]
    
    # Added Batch Normalization
    args.batch_norm = [strtobool(bn) for bn in args.batch_norm]
    
    # Convert depth normalization to boolean
    depth_normalization = list(map(strtobool, cfg['depth_normalization']))

    # Initialize parameter grids list
    param_grids = []
    experiment_types = []

    # Only add grids that are necessary
    if any(l != 0 for l in l2_sum_lambda):
        sum_param_grid = param_grid.copy()
        sum_param_grid["l2_sum_lambda"] = l2_sum_lambda
        sum_param_grid["l2_mul_lambda"] = [0.0]
        sum_param_grid["wn"] = wn_values
        sum_param_grid["depth_normalization"] = depth_normalization
        sum_param_grid["features_normalization"] = cfg['features_normalization']
        sum_param_grid["batch_norm"] = cfg['batch_norm']
        param_grids.append(sum_param_grid)
        experiment_types.append("Summation")

    if any(l != 0 for l in l2_mul_lambda):
        mul_param_grid = param_grid.copy()
        mul_param_grid["l2_sum_lambda"] = [0.0]
        mul_param_grid["l2_mul_lambda"] = l2_mul_lambda
        mul_param_grid["wn"] = wn_values
        mul_param_grid["depth_normalization"] = depth_normalization
        mul_param_grid["features_normalization"] = cfg['features_normalization']
        mul_param_grid["batch_norm"] = cfg['batch_norm']
        param_grids.append(mul_param_grid)
        experiment_types.append("Multiplication")

    if all(l == 0 for l in l2_sum_lambda + l2_mul_lambda):
        noreg_param_grid = param_grid.copy()
        noreg_param_grid["l2_sum_lambda"] = [0.0]
        noreg_param_grid["l2_mul_lambda"] = [0.0]
        noreg_param_grid["wn"] = wn_values
        noreg_param_grid["depth_normalization"] = depth_normalization
        noreg_param_grid["batch_norm"] = cfg['batch_norm']
        param_grids.append(noreg_param_grid)
        experiment_types.append("No_Regularization")

    init_scalings = cfg['init_scaling'] if len(cfg['init_scaling']) > 0 else [1.0]
    mislabels = cfg['mislabel_percentage'] if len(cfg['mislabel_percentage']) > 0 else [0.0]

    for pg in param_grids:
        pg["init_scaling"] = init_scalings
        pg["mislabel_percentage"] = mislabels

    # Initialize the results DataFrame with necessary columns, including 'experiment_type' and 'seed'
    results = pd.DataFrame(columns=[
        "experiment_type", "seed", "batchsize", "in_channels", "hidden_channels", "act_name", "bias",
        "opt_name", "lr", "n_epochs", "l2_sum_lambda", "l2_mul_lambda", "train_epoch_losses",
        "train_epoch_l2_sum_losses", "train_epoch_l2_mul_losses", "train_batch_losses",
        "train_batch_l2_sum_losses", "train_batch_l2_mul_losses", "test_losses",
        "test_accuracies", "norms", "wn", "train_losses", "train_accuracies", 
        "train_f1_scores", "train_confusion_matrices",
        "test_f1_scores", "test_confusion_matrices",
        "depth_normalization", "features_normalization", "margins", "batch_norm",
        "epoch_times", "margins_per_sample", "misclassified_indices", "weight_ranks",
        "rho_values", "learning_rates"
    ])

    # Calculate total number of experiments
    total_experiments = 0
    for param_grid in param_grids:
        # Count number of parameter combinations for this grid
        combinations = 1
        for param_values in param_grid.values():
            combinations *= len(param_values)
        total_experiments += combinations
    
    print(f"\n===== Running a total of {total_experiments} experiments =====\n")
    
    # Initialize experiment counter
    current_experiment = 0

    # Iterate over experiment types and parameter grids
    for experiment_type, param_grid in zip(experiment_types, param_grids):
        for params in product_dict(**param_grid):
            current_experiment += 1
            
            # Print progress information
            print(f"\n===== Running experiment {current_experiment}/{total_experiments} ({experiment_type}) =====")
            print(f"Parameters: LR={params['lr']}, Batch Size={params['batchsize']}, Seed={params['seed']}")
            if params['mislabel_percentage'] > 0:
                print(f"Mislabel: {params['mislabel_percentage']*100}%")
            if params['init_scaling'] != 1.0:
                print(f"Init Scaling: {params['init_scaling']}")
                
            experiment_cfg = cfg.copy()
            experiment_cfg.update(params)  # Update the cfg dictionary with current params

            # Set the seed for reproducibility
            set_seed(experiment_cfg['seed'])

            # Create a fresh copy of the training dataset for this experiment
            trainset = copy.deepcopy(trainset_original)

            # Mislabel the dataset if mislabel_percentage > 0
            if experiment_cfg['mislabel_percentage'] > 0:
                print(f"Mislabeling {experiment_cfg['mislabel_percentage']*100}% of training data with seed {experiment_cfg['seed']}")
                mislabel_dataset(trainset, experiment_cfg['mislabel_percentage'], seed=experiment_cfg['seed'])

            # Create the data loader for this experiment with the correct batch size
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=experiment_cfg['batchsize'], shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=experiment_cfg['batchsize'], shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

            # Run the experiment
            experiment_results = experiment(cfg=experiment_cfg, train_loader=train_loader, test_loader=test_loader)

            # Construct the row and append to the results DataFrame
            results = results._append({
                "experiment_type": experiment_type,
                **experiment_results["hp"],
                **experiment_results
            }, ignore_index=True)

            # Save per-experiment results
            experiment_folder = os.path.join('results', experiment_type)
            os.makedirs(experiment_folder, exist_ok=True)
            
            # Generate filename with date at the beginning and including the experiment name and seed
            experiment_filename = f"{datetime.now():%Y%m%d-%H%M%S}_{experiment_type}_lr_{params['lr']}_epochs_{params['n_epochs']}_seed_{params['seed']}_batchsize_{params['batchsize']}_per_experiment.pkl"
            experiment_filepath = os.path.join(experiment_folder, experiment_filename)

            # Save each experiment result to a separate pickle file
            with open(experiment_filepath, "wb") as f:
                pickle.dump(experiment_results, f)
            print(f"Results for {experiment_type} with seed {params['seed']} saved to {experiment_filepath}")
            
            # Report completion and remaining experiments
            remaining = total_experiments - current_experiment
            print(f"===== Experiment {current_experiment}/{total_experiments} completed. {remaining} remaining. =====")

    results.fillna(value='None', inplace=True)  # Fill None values

    # Print all results
    print("\nResults:")
    print(results)
    print("\n")

    # Extract and print best results
    best_results = []

    for experiment_type in results['experiment_type'].unique():
        subset = results[results['experiment_type'] == experiment_type]

        # Find the row with the best final test accuracy
        best_idx = subset['test_accuracies'].apply(lambda x: max(x) if len(x) > 0 else 0).idxmax()
        best_row = subset.loc[best_idx]

        # Extract the required columns, including F1 score and confusion matrix
        best_result = {
            "experiment_type": experiment_type,
            "seed": best_row["seed"],
            "batchsize": best_row["batchsize"],
            "in_channels": best_row["in_channels"],
            "act_name": best_row.get("act_name", None),
            "bias": best_row["bias"],
            "opt_name": best_row["opt_name"],
            "lr": best_row["lr"],
            "n_epochs": best_row["n_epochs"],
            "l2_sum_lambda": best_row["l2_sum_lambda"],
            "l2_mul_lambda": best_row["l2_mul_lambda"],
            "wn": best_row["wn"],
            "depth_normalization": best_row["depth_normalization"],
            "batch_norm": best_row["batch_norm"], 
            "final_test_accuracy": max(best_row["test_accuracies"] if len(best_row["test_accuracies"]) > 0 else [0]),
            "train_f1_score": max(best_row["train_f1_scores"] if len(best_row["train_f1_scores"]) > 0 else [0]),
            "test_f1_score": max(best_row["test_f1_scores"] if len(best_row["test_f1_scores"]) > 0 else [0]),
        }

        best_results.append(best_result)

    best_results_df = pd.DataFrame(best_results, columns=["experiment_type"] + [
        "seed",
        "batchsize",
        "in_channels",
        "act_name",
        "bias",
        "opt_name",
        "lr",
        "n_epochs",
        "l2_sum_lambda",
        "l2_mul_lambda",
        "wn",
        "depth_normalization",
        "batch_norm",
        "final_test_accuracy",
        "train_f1_score",
        "test_f1_score",
    ])
    best_results_df.fillna(value='None', inplace=True)  # Fill None values

    print("\nBest Results:")
    print(best_results_df)

    # Ensure the results folder exists
    os.makedirs('results', exist_ok=True)

    # Generate filename based on date, experiment types, and include optimizer and parameters in brackets
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_types_str = '_'.join(experiment_types)

    # Collect key parameters to add to the filename
    lr_str = f"[{args.lr[0]}]" if len(args.lr) == 1 else f"[{args.lr}]"
    l2_sum_str = f"[{args.l2_sum_lambda[0]}]" if len(args.l2_sum_lambda) == 1 else f"[{args.l2_sum_lambda}]"
    l2_mul_str = f"[{args.l2_mul_lambda[0]}]" if len(args.l2_mul_lambda) == 1 else f"[{args.l2_mul_lambda}]"
    wn_str = f"[{args.wn[0]}]" if len(args.wn) == 1 else f"[{args.wn}]"
    batch_norm_str = f"[{args.batch_norm[0]}]" if len(args.batch_norm) == 1 else f"[{args.batch_norm}]"
    seed_str = f"[{args.seed}]"
    batchsize_str = f"[{args.batchsize}]"
    mislabel_str = f"[{args.mislabel_percentage}]"

    # Determine normalization type for the filename
    normalization_type = "Weight_Normalization" if args.wn and any(args.wn) else "Batch_Normalization"

    # Create the overall filename
    filename = f"Results/{date_str}_[{normalization_type}]_[{experiment_types_str}]_opt_[{args.opt_name}]_lr_{lr_str}_batchsize{batchsize_str}_l2sum{l2_sum_str}_l2mul{l2_mul_str}_wn{wn_str}_bn{batch_norm_str}_seed{seed_str}_mislabel{mislabel_str}.pkl"
    os.makedirs('Results', exist_ok=True)

    # Save the pickle file in the 'results' folder
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Overall results saved to {filename}")

if __name__ == "__main__":
    main()