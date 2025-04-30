# Import necessary libraries
from functools import reduce
from itertools import product
import os
import random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from model import RegNet, PaperWNInitModel, PaperWNModel, OverparameterizedCNN, RhoWNModel, UnderparameterizedCNN, UnderparameterizedCNN_AvgPool, UnderparameterizedCNN_LinearBias, SuperUnderparameterizedCNN, set_model_norm_to_one
import gc
import time
from datetime import datetime
import copy

# Function to create a product of dictionaries
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))

# Extract and print best results
def extract_best_results(results):
    best_results = (
        results.assign(final_test_accuracy=results['test_accuracies'].apply(lambda x: max(x) if len(x) > 0 else 0))
        .loc[results.groupby('experiment_type')['final_test_accuracy'].idxmax()]
    )
    
    columns = ["experiment_type", "batchsize", "in_channels", "act_name", "bias", "opt_name", "lr",
               "n_epochs", "l2_sum_lambda", "l2_mul_lambda", "wn", "depth_normalization", 
               "final_test_accuracy"]
    
    best_results_df = best_results[columns].reset_index(drop=True)
    best_results_df.fillna(value='None', inplace=True)  # Added line to replace None values
    return best_results_df

def mislabel_dataset(dataset, mislabel_percentage, seed=None):
    if not 0 <= mislabel_percentage <= 1:
        raise ValueError("mislabel_percentage must be between 0 and 1")
    if seed is not None:
        np.random.seed(seed)
    num_samples = len(dataset.targets)
    num_to_mislabel = int(mislabel_percentage * num_samples)
    indices = np.random.choice(num_samples, num_to_mislabel, replace=False)
    for idx in indices:
        dataset.targets[idx] = -dataset.targets[idx]  # Flip the label
    return dataset

def compute_weight_ranks(model):
    layer_ranks = {}
    for name, param in model.named_parameters():
        # Skip non-weight parameters
        if 'weight' not in name:
            continue

        # Handle parametrized weights (for models using PaperWeightNorm, PaperWeightNormWithInit, etc.)
        if 'parametrizations' in name:
            # If 'original' in name, this is the actual weight after parametrization
            if 'original' in name:
                weight_name = name.replace('.parametrizations.weight.original', '.parametrizations.weight.0.g')
                g_param = dict(model.named_parameters()).get(weight_name)
                g_value = g_param.item() if g_param is not None else 1.0
                # Apply scaling to the original weight
                weight = param.detach().cpu().numpy() * g_value
            else:
                # Skip auxiliary parameters like .0.g which have already been used
                continue
        else:
            # Direct weight without parametrization
            weight = param.detach().cpu().numpy()

        # Handle different dimensions of weights
        if weight.ndim == 4:
            # Convolutional layer: reshape for rank calculation
            out_channels, in_channels, kh, kw = weight.shape
            weight_matrix = weight.reshape(out_channels, -1)
        elif weight.ndim == 2:
            # Fully connected layer: use as is
            weight_matrix = weight
        elif weight.ndim == 1:
            # Likely a bias or batch norm scale parameter, which should be skipped
            continue
        else:
            # Unexpected dimension: skip
            continue

        # Compute the singular values of the weight matrix
        try:
            singular_values = np.linalg.svd(weight_matrix, compute_uv=False)
            singular_values_normalized = singular_values / singular_values.max()
            threshold = 0.01
            rank = np.sum(singular_values_normalized > threshold)
            layer_ranks[name] = rank
        except np.linalg.LinAlgError:
            continue

    # Incorporate global scaling factors like rho from RhoWNModel
    if isinstance(model, RhoWNModel):
        rho_value = model.net.rho.item() ** 2  # Assuming rho is used in squared form
        for key in layer_ranks:
            # Update rank values by incorporating rho scaling
            layer_ranks[key] *= rho_value

    return layer_ranks

def train_epoch(model, cfg, train_loader, optimizer, print_every_batch=100):
    model.train()
    assert isinstance(model, RegNet)

    loss_out = 0
    l2_sum_loss_out = 0
    l2_mul_loss_out = 0
    loss_vec = []
    l2_sum_loss_vec = []
    l2_mul_loss_vec = []
    norms_vec = []  
    total_num_samples = 0
    all_preds = []
    all_targets = []
    output_confidences = []

    for batch_idx, (data, target) in enumerate(train_loader):
        num_samples = target.size(0)
        data, target = data.to(cfg['device']), target.to(cfg['device']).float()  # Convert targets to float
        optimizer.zero_grad()
        output = model(data).squeeze()

        # Use MSE Loss for binary classification
        loss = F.mse_loss(output, target)
        
        l2_sum_loss = 0
        l2_mul_loss = 0

        if cfg['l2_sum_lambda'] != 0:
            l2_sum_loss = model.compute_l2_sum(
                depth_normalization=cfg['depth_normalization'], 
                features_normalization=cfg['features_normalization']
            )
        elif cfg['l2_mul_lambda'] != 0:
            l2_mul_loss = model.compute_l2_mul(
                depth_normalization=cfg['depth_normalization'], 
                features_normalization=cfg['features_normalization']
            )

        reg_loss = loss + cfg['l2_sum_lambda'] * l2_sum_loss + cfg['l2_mul_lambda'] * l2_mul_loss
        reg_loss.backward()
        optimizer.step()

        loss_out += loss.item() * num_samples
        total_num_samples += num_samples
        l2_sum_loss_out += l2_sum_loss.item() if cfg['l2_sum_lambda'] != 0 else 0
        l2_mul_loss_out += l2_mul_loss.item() if cfg['l2_mul_lambda'] != 0 else 0
        loss_vec.append(loss.item())
        l2_sum_loss_vec.append(l2_sum_loss.item() if cfg['l2_sum_lambda'] != 0 else 0)
        l2_mul_loss_vec.append(l2_mul_loss.item() if cfg['l2_mul_lambda'] != 0 else 0)
        
        preds = torch.sign(output)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        # --- Confidence Calculation ---
        confidence = torch.abs(output).detach().cpu().numpy()
        output_confidences.append(confidence)

        # --- Compute Norms ---
        norms = torch.sqrt(torch.stack(model._compute_norms(False)))  # Compute norms
        norms_vec.append(norms.detach().cpu().numpy())  # Store the computed norms

        if batch_idx % print_every_batch == 0:
            mean_confidence = np.mean(confidence)
            print(
                f"Train Epoch: {cfg['epoch']} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
        
        if torch.isnan(loss):
            print("Loss is NaN, breaking epoch...")
            break

    loss_out /= total_num_samples
    mean_epoch_confidence = np.mean(np.concatenate(output_confidences))

    preds_binary = (np.array(all_preds) > 0).astype(int)
    targets_binary = (np.array(all_targets) > 0).astype(int)

    train_accuracy = accuracy_score(targets_binary, preds_binary)
    train_f1 = f1_score(targets_binary, preds_binary, average='binary')
    train_cm = confusion_matrix(targets_binary, preds_binary)
    print(f"\nEpoch {cfg['epoch']} Mean Confidence: {mean_epoch_confidence:.6f}")

    return {
        "loss_out": loss_out,
        "l2_sum_loss_out": l2_sum_loss_out,
        "l2_mul_loss_out": l2_mul_loss_out,
        "loss_vec": loss_vec,
        "l2_sum_loss_vec": l2_sum_loss_vec,
        "l2_mul_loss_vec": l2_mul_loss_vec,
        "norms_vec": norms_vec,  # Now norms are computed and stored
        "accuracy": train_accuracy,
        "f1_score": train_f1,
        "confusion_matrix": train_cm,
        "confidence_out": mean_epoch_confidence,
    }

def test(model, cfg, loader, loader_name="Test"):
    model.eval()
    loss = 0
    correct = 0
    total_num_samples = 0

    all_preds = []
    all_targets = []
    misclassified_indices = []  # To store indices of misclassified samples

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(cfg['device']), target.to(cfg['device']).float()  # Convert targets to float
            num_samples = target.size(0)
            output = model(data).squeeze()

            loss += F.mse_loss(output, target, reduction="sum").item()

            preds = torch.sign(output)
            all_preds_batch = preds.cpu().numpy()
            all_targets_batch = target.cpu().numpy()
            all_preds.extend(all_preds_batch)
            all_targets.extend(all_targets_batch)
            correct += preds.eq(target).sum().item()
            total_num_samples += num_samples
            misclassified = (all_preds_batch != all_targets_batch)
            indices = np.arange(batch_idx * loader.batch_size, batch_idx * loader.batch_size + num_samples)
            misclassified_indices.extend(indices[misclassified])

    loss /= total_num_samples
    accuracy = 100.0 * correct / total_num_samples

    # Convert preds and targets to 0/1
    preds_binary = (np.array(all_preds) > 0).astype(int)
    targets_binary = (np.array(all_targets) > 0).astype(int)

    test_f1 = f1_score(targets_binary, preds_binary, average='binary')  
    test_cm = confusion_matrix(targets_binary, preds_binary)  

    print(
        f"\n{loader_name} set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n"
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "f1_score": test_f1,
        "confusion_matrix": test_cm,
        "misclassified_indices": misclassified_indices,  # Return indices of misclassified samples
    }

# Margin: Product of the true label (target) and the model's raw prediction (output)
def compute_margins(model, cfg, data_loader):
    model.eval()
    margins_per_sample = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(cfg['device']), target.to(cfg['device']).float()
            output = model(data).squeeze()
            margins = target * output
            margins_per_sample.extend(margins.cpu().numpy())

    return margins_per_sample

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  

    # For CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure reproducibility for determinism if using cudnn backend (only relevant for CUDA)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

# Function to run an experiment with given hyperparameters
def experiment(cfg, train_loader, test_loader, print_every_batch=100):
    for key in ['wn', 'batch_norm', 'depth_normalization', 'features_normalization']:
        if isinstance(cfg.get(key), list):
            cfg[key] = cfg[key][0]
    if isinstance(cfg.get('batch_norm'), str):
        cfg['batch_norm'] = cfg['batch_norm'].lower() == "true"
    if isinstance(cfg.get('wn'), str) and cfg['wn'].lower() == "none":
        cfg['wn'] = None
    

    set_seed(cfg['seed'])  # Use the seed from cfg

    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg['device'] = device  

    # Log device info
    print(f"Running on device: {device}")
    if device.type == 'cuda':
        print(f"Available CUDA devices: {torch.cuda.device_count()}")

    # Adjust 'batch_norm' based on 'wn' and the logic
    if cfg['wn']:
        cfg['batch_norm'] = False  
    else:
        cfg['batch_norm'] = cfg['batch_norm']  

    model_name = "UnderparameterizedCNN_LinearBias" 
    model = UnderparameterizedCNN_LinearBias(
        in_channels=cfg['in_channels'],
        num_classes=1,  
        act="relu",  
        bias=cfg['bias'],
        batch_norm=cfg['batch_norm'], 
    ).to(cfg['device'])

    
    if 'init_scaling' in cfg and cfg['init_scaling'] is not None and cfg['init_scaling'] != 1.0:
        print(f"Scaling model weights by {cfg['init_scaling']}")
        for _, param in model.named_parameters():
            param.data.mul_(cfg['init_scaling'])
    

    # Apply weight normalization if specified in 'wn'
    if cfg['wn']:
        #print("Applying weight normalization...")
        wn_values = cfg['wn'] if isinstance(cfg['wn'], list) else [cfg['wn']]
        for wn_value in wn_values:
            if isinstance(wn_value, float):
                model = PaperWNInitModel(model, wn_value)
                model_name = "PaperWNInitModel"
            elif wn_value == 'default':
                model = PaperWNModel(model)
                model_name = "PaperWNModel"

    # Choose optimizer
    if cfg['opt_name'] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        print("Using Adam optimizer")
    elif cfg['opt_name'] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0)
        print("Using AdamW optimizer")
    elif cfg['opt_name'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg['opt_name']}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'], eta_min=0.003)

    # Initialize results dictionary
    results = {
        "hp": cfg,  # Include hyperparameters in results
        "train_epoch_losses": [],
        "train_epoch_l2_sum_losses": [],
        "train_epoch_l2_mul_losses": [],
        "train_batch_losses": [],
        "train_batch_l2_sum_losses": [],
        "train_batch_l2_mul_losses": [],
        "train_losses": [],
        "train_accuracies": [],
        "train_f1_scores": [], 
        "train_confusion_matrices": [], 
        "test_losses": [],
        "test_accuracies": [],
        "test_f1_scores": [], 
        "test_confusion_matrices": [], 
        "norms": [],
        "epoch_times": [],  
        "mean_margin": None, 
        "margins_per_sample": None,  
        "misclassified_indices": [],  
        "weight_ranks": None,  
        "rho_values": [],  
        "learning_rates": [], 
        "model_state_name": None, 

    }

    print("Parameters:", cfg)

    # --- Start the training loop ---
    for epoch in range(1, cfg['n_epochs'] + 1):
        cfg['epoch'] = epoch

        start_time = time.time()  

        train_results = train_epoch(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            optimizer=optimizer,
            print_every_batch=print_every_batch
        )

        if np.isnan(train_results['loss_out']):
            print("Loss is NaN, breaking training...")
            break

        train_loss = test(model, cfg, train_loader, loader_name='Train')
        test_loss = test(model, cfg, test_loader, loader_name='Test')

        # Update learning rate using scheduler
        scheduler.step()

        end_time = time.time()  # End timing the epoch
        epoch_time = end_time - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")
        results['learning_rates'].append(current_lr)

        # Store the time taken for this epoch
        results['epoch_times'].append(epoch_time)

        # Store results
        results['train_epoch_losses'].append(train_results['loss_out'])
        results['train_epoch_l2_sum_losses'].append(train_results['l2_sum_loss_out'])
        results['train_epoch_l2_mul_losses'].append(train_results['l2_mul_loss_out'])
        results['train_batch_losses'] += train_results['loss_vec']
        results['train_batch_l2_sum_losses'] += train_results['l2_sum_loss_vec']
        results['train_batch_l2_mul_losses'] += train_results['l2_mul_loss_vec']
        results['train_losses'].append(train_loss['loss'])
        results['train_accuracies'].append(train_loss['accuracy'])
        results['train_f1_scores'].append(train_loss['f1_score'])
        results['train_confusion_matrices'].append(train_loss['confusion_matrix'])
        results['test_losses'].append(test_loss['loss'])
        results['test_accuracies'].append(test_loss['accuracy'])
        results['test_f1_scores'].append(test_loss['f1_score'])
        results['test_confusion_matrices'].append(test_loss['confusion_matrix'])
        results['norms'] += train_results['norms_vec']

        # --- Compute and store rho ---
        if cfg['l2_sum_lambda'] == 0 and cfg['l2_mul_lambda'] == 0:
            prod_rho = 0  # No regularization
        else:
            prod_rho = torch.sqrt(model.compute_l2_mul(False, False)).item()  # Only compute if regularization is applied
        results['rho_values'].append(prod_rho)
        print(f"rho at epoch {epoch}: {prod_rho}")

        # Early stopping implementation
        if np.isnan(test_loss['loss']) or np.isnan(test_loss['accuracy']):
            print(f"Detected NaN values in test loss or accuracy at epoch {epoch}, stopping training.")
            break
            
        if test_loss['accuracy'] == 0:
            print(f"Test accuracy is 0% at epoch {epoch}, model is not learning, stopping training.")
            break
            
        if epoch >= 50 and test_loss['accuracy'] < 60:
            print(f"Model did not exceed 60% accuracy after {epoch} epochs, stopping training.")
            break

        # Free up GPU memory after 30 epochs
        if epoch % 30 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print()
    # Handle norms based on whether batch normalization was used
    if not cfg['batch_norm']:
        results['norms'] = np.array(results['norms']).transpose()
        print(f"Norms shape: {results['norms'].shape}")
        
    else:
        print("Batch normalization was used, norms are not computed.")
        results['norms'] = []  # Norms are not computed when batch_norm is True

    # --- Compute weight ranks at convergence ---
    # This computes the ranks after the training loop has completed, i.e., at convergence
    weight_ranks = compute_weight_ranks(model)
    results['weight_ranks'] = weight_ranks  # Store the ranks in the results dictionary

    # After train compute margins on training data
    margin_model = copy.deepcopy(model)
    set_model_norm_to_one(margin_model)

    margins_per_sample = compute_margins(margin_model, cfg, train_loader)
    mean_margin = np.mean(margins_per_sample)
    print(f"Mean Margin at Convergence: {mean_margin:.6f}")

    results['mean_margin'] = mean_margin
    results['margins_per_sample'] = margins_per_sample

    if cfg['epoch'] == cfg['n_epochs']:
        results['misclassified_indices'] = test_loss['misclassified_indices']


    # Save the model state
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")

    # Parameters to include in the filename
    params_to_include = [
        'batchsize',  # Added line to include batchsize in the filename
        'lr', 'n_epochs', 'l2_sum_lambda', 'l2_mul_lambda', 
        'wn', 'depth_normalization', 'features_normalization',
        'batch_norm', 'seed'  # Include seed in filename
    ]

    # Build the parameter string for the filename
    param_strs = [f"{param}[{str(cfg[param]).replace('/', '-').replace(' ', '_')}]" for param in params_to_include]

    # Build the model save path and determine regularization folder
    model_state_name = date_str + '-' + '-'.join(param_strs) + '.pt'

    if cfg['l2_sum_lambda'] != 0 and cfg['l2_mul_lambda'] == 0:
        regularization_folder = 'Summation'
    elif cfg['l2_sum_lambda'] == 0 and cfg['l2_mul_lambda'] != 0:
        regularization_folder = 'Multiplication'
    elif cfg['l2_sum_lambda'] == 0 and cfg['l2_mul_lambda'] == 0:
        regularization_folder = 'No_Regularization'
    else:
        regularization_folder = 'Both'

    # Create necessary directories early
    model_dir = os.path.join('Models', regularization_folder)
    os.makedirs(model_dir, exist_ok=True)

    model_state_path = os.path.join(model_dir, model_state_name)
    try:
        torch.save(model.state_dict(), model_state_path)
        print(f"Model state_dict saved to {model_state_path}")
    except Exception as e:
        print(f"Failed to save model state_dict to {model_state_path}. Error: {e}")

    results['model_state_name'] = model_state_name

    return results