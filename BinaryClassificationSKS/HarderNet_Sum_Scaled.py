import os
import sys
import torch
import datetime
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from losscape.create_landscape import create_3D_losscape
import pickle

################## Configuration ##################
batch_size = 16
epochs = 2000
learning_rate = 0.03
features_normalization = False
resolution_points = 1
x_min = -5
x_max = 5
y_min = -5
y_max = 5
experiment_type = 'Additive_Misclass_'
optimizer_choice = 'sgd'  
scheduler_choice = None
bias = False  
interval = 200
init_method = 'scaled'  
early_stopping = True
patience = 200  
min_delta = 0.0001
###################################################

# Read λ_sum from command line (or default to 0.05 if not provided)
if len(sys.argv) > 1:
    try:
        λ_sum = float(sys.argv[1])
        print(f"Using provided λ_sum={λ_sum} from command line")
    except ValueError:
        print(f"ERROR: Invalid λ_sum value '{sys.argv[1]}'. Using default λ_sum=0.05")
        λ_sum = 0.05
else:
    λ_sum = 0.05

# Read init_scaling from command line (or default to 0.2 if not provided)
if len(sys.argv) > 2:
    try:
        init_scaling = float(sys.argv[2])
        print(f"Using provided init_scaling={init_scaling} from command line")
    except ValueError:
        print(f"ERROR: Invalid init_scaling value '{sys.argv[2]}'. Using default init_scaling=0.2")
        init_scaling = 0.2
else:
    init_scaling = 0.2

# Read seed from command line (or default to 1 if not provided)
if len(sys.argv) > 3:
    try:
        seed = int(sys.argv[3])
        if seed <= 0:
            print(f"WARNING: Invalid seed value {seed}. Must be positive. Setting seed=1")
            seed = 1
        print(f"Using provided seed={seed} from command line")
    except ValueError:
        print(f"ERROR: Invalid seed value '{sys.argv[3]}'. Using default seed=1")
        seed = 1
else:
    seed = 1

# Read misclassification rate from command line
if len(sys.argv) > 4:
    try:
        misclassification_rate = float(sys.argv[4])
        if misclassification_rate < 0 or misclassification_rate > 1:
            print(f"WARNING: Invalid misclassification_rate value {misclassification_rate}. Must be between 0 and 1. Setting misclassification_rate=0.0")
            misclassification_rate = 0.0
        print(f"Using provided misclassification_rate={misclassification_rate} from command line")
    except ValueError:
        print(f"ERROR: Invalid misclassification_rate value '{sys.argv[4]}'. Using default misclassification_rate=0.0")
        misclassification_rate = 0.0
else:
    misclassification_rate = 0.0

# Read initialization method from command line
if len(sys.argv) > 5:
    if sys.argv[5] in ['scaled', 'kaiming', 'normalized']:
        init_method = sys.argv[5]
        print(f"Using provided init_method={init_method} from command line")
    else:
        print(f"ERROR: Invalid init_method value '{sys.argv[5]}'. Must be one of 'scaled', 'kaiming', 'normalized'. Using default init_method='scaled'")
        init_method = 'scaled'

# For additive script: λ_mul is fixed to 0
λ_mul = 0

# Update output path based on parameters
out_path = f'Results/Visualization/{experiment_type}_{init_scaling}_seed_{seed}_noise_{misclassification_rate}'
print(f"Output path: {out_path}")
print(f"Using λ_sum={λ_sum}, λ_mul={λ_mul}, misclassification_rate={misclassification_rate}")
print(f"Using initialization method: {init_method}")

# Set global seeds
torch.manual_seed(seed)
np.random.seed(seed)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
start_time = time.time()

############################
# Load and preprocess CIFAR-10 data
############################

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# Load full CIFAR-10 dataset
cifar_trainval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Extract cat (class 3) and deer (class 4) from CIFAR-10
def get_class_indices(dataset, class_labels):
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in class_labels:
            indices.append(i)
    return indices

cat_deer_indices_train = get_class_indices(cifar_trainval, [3, 4])  # 3: cat, 4: deer
cat_deer_indices_test = get_class_indices(cifar_test, [3, 4])

# Create binary subsets - convert labels from 3, 4 to 0, 1
cat_deer_trainval = Subset(cifar_trainval, cat_deer_indices_train)
cat_deer_test = Subset(cifar_test, cat_deer_indices_test)

# Create binary datasets with labels mapped to -1 (cat) and 1 (deer)
def map_labels(dataset, class_map):
    images = []
    labels = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        images.append(img)
        labels.append(class_map[label])
    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.float))

# Map cat (3) to -1 and deer (4) to 1 for MSE loss
class_map = {3: -1.0, 4: 1.0}  
binary_trainval = map_labels(cat_deer_trainval, class_map)
binary_test = map_labels(cat_deer_test, class_map)

# Apply artificial label noise if specified
if misclassification_rate > 0:
    print(f"Applying artificial label noise: {misclassification_rate * 100}% of training labels will be flipped")
    
    # Create noisy training dataset
    num_samples = len(binary_trainval)
    num_to_flip = int(num_samples * misclassification_rate)
    indices_to_flip = np.random.choice(num_samples, num_to_flip, replace=False)
    
    noisy_images = []
    noisy_labels = []
    for i in range(num_samples):
        img, label = binary_trainval[i]
        if i in indices_to_flip:
            # Flip the label (-1 to 1 or 1 to -1)
            label = -label
        noisy_images.append(img)
        noisy_labels.append(label)
    
    noisy_binary_trainval = TensorDataset(torch.stack(noisy_images), torch.tensor(noisy_labels, dtype=torch.float))
    print(f"Created noisy training dataset with {num_to_flip} flipped labels out of {num_samples} total samples")
    
    # Use noisy dataset instead of original
    binary_trainval = noisy_binary_trainval

# Split the training set 80/20
train_size = int(0.8 * len(binary_trainval))
val_size = len(binary_trainval) - train_size

train_dataset, val_dataset = random_split(binary_trainval, [train_size, val_size])
test_dataset = binary_test

# Prepare DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader_fixed = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Fixed loader for visualization 

print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Define CNN model
class CIFAR10CNN(nn.Module):
    def __init__(self, bias=False):
        super(CIFAR10CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias)
        
        # Average pooling instead of max pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # For binary classification via regression (MSE):
        self.fc1 = nn.Linear(128 * 2 * 2, 256, bias=True)
        self.fc2 = nn.Linear(256, 1, bias=True)  # single output neuron for MSE
        
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Conv block 3
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Conv block 4
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 2 * 2)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # shape: [batch_size, 1]
        
        # Squeeze the output to shape [batch_size] for compatibility with compute_loss
        x = x.squeeze(1)
        
        return x

model = CIFAR10CNN(bias=bias).to(device)

def apply_kaiming_initialization(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

def apply_init_scaling(model, scaling_factor):
    if scaling_factor != 1.0:
        print(f"Scaling model weights by {scaling_factor}")
        for _, param in model.named_parameters():
            param.data.mul_(scaling_factor)

def compute_layer_norms(model):
    """Compute the norm of each layer's weights."""
    norms = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            norm = torch.norm(m.weight, 2)
            norms.append(norm)
    return norms

def apply_normalized_scaling(model, scaling_factor):
    """Normalize each layer then scale. """
    print(f"Applying normalized scaling with factor {scaling_factor}")
    # First compute norms of each layer
    norms = compute_layer_norms(model)
    
    # Then normalize each layer and apply scaling
    i = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                # Normalize by dividing by the current norm, then multiply by scaling factor
                m.weight.data.mul_(scaling_factor / norms[i])
                i += 1
    
    # Verify the scaling worked as expected
    new_norms = compute_layer_norms(model)
    print(f"After normalization, layer norms: {[norm.item() for norm in new_norms]}")

# Early stopping
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_dict = None

    def __call__(self, val_loss, model):
        # Check for NaN in validation loss and stop immediately if found
        if torch.isnan(torch.tensor(val_loss)):
            if self.verbose:
                print(f'NaN detected in validation loss. Early stopping triggered.')
            self.early_stop = True
            return
            
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model_dict = model.state_dict().copy()
    
    def restore_best_model(self, model):
        if self.best_model_dict is not None:
            model.load_state_dict(self.best_model_dict)
            if self.verbose:
                print('Restored best model')

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # Already shape [batch_size] due to model's forward method
            
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)

            # Compute predictions: use sign for -1/1 classification
            predicted = torch.sign(outputs)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# Calculate margins - with MSE and -1/1 labels
def calculate_margins(model, data_loader):
    model.eval()
    all_margins = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # Already shape [batch_size]
            
            # For MSE with -1/1 labels, margin is just output * label
            # since positive output with positive label (or negative with negative) indicates correct prediction
            margins = outputs * y_batch
            all_margins.extend(margins.cpu().numpy())
    
    return np.array(all_margins)

# Calculate both raw and normalized margins - for MSE with -1/1 labels
def calculate_both_margins(model, data_loader):
    """Calculate both raw and normalized margins."""
    model.eval()
    raw_margins = []
    
    # For normalization, we'll compute the standard deviation of raw outputs
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # Already shape [batch_size]
            
            # Store raw outputs and labels for normalization later
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            # Calculate margins: output * label
            margins = outputs * y_batch
            raw_margins.extend(margins.cpu().numpy())
    
    raw_margins = np.array(raw_margins)
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # Compute standard deviation of outputs for normalization
    output_std = np.std(all_outputs)
    
    # Normalize margins by the standard deviation of outputs
    normalized_margins = raw_margins / output_std if output_std > 0 else raw_margins
    
    return raw_margins, normalized_margins

# Initialization
if init_method == 'kaiming':
    apply_kaiming_initialization(model)
    if init_scaling != 1.0:
        apply_init_scaling(model, init_scaling)
elif init_method == 'scaled':
    apply_kaiming_initialization(model)
    apply_init_scaling(model, init_scaling)
elif init_method == 'normalized':
    # First apply Kaiming init (or you could use default PyTorch init)
    apply_kaiming_initialization(model)
    # Then apply normalized scaling
    apply_normalized_scaling(model, init_scaling)
else:
    raise ValueError(f"Unknown initialization method: {init_method}")

# Use MSE for regression-based binary classification
criterion = nn.MSELoss()

if optimizer_choice == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_choice == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f"Unknown optimizer choice: {optimizer_choice}")

if scheduler_choice == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1)
elif scheduler_choice == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0001)
elif scheduler_choice is None:
    scheduler = None
else:
    raise ValueError(f"Unknown scheduler choice: {scheduler_choice}")

def compute_l2_sum(model, features_normalization):
    norms = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            p_norm = torch.norm(m.weight, 2) ** 2
            if features_normalization:
                p_norm /= m.weight.size(0)
            norms.append(p_norm)
    return torch.sum(torch.stack(norms))

def compute_l2_mul(model, features_normalization):
    norms = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            p_norm = torch.norm(m.weight, 2) ** 2
            if features_normalization:
                p_norm /= m.weight.size(0)
            norms.append(p_norm)
    return torch.prod(torch.stack(norms))

def compute_rank_per_layer(model):
    layer_ranks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            W = module.weight.data
            # For Conv2d, reshape to 2D matrix
            if isinstance(module, nn.Conv2d):
                W = W.reshape(W.size(0), -1)
            rank = torch.linalg.matrix_rank(W).item()
            layer_ranks[name] = rank
    return layer_ranks

# Tracking
ranks_per_layer_list = []
training_losses = []
validation_losses = []
validation_accuracies = []
l2_sum_values = []  # Track l2_sum over epochs
l2_mul_values = []  # Track l2_mul over epochs

early_stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

# Training
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    penalty = 0.0
    nan_detected = False  # Track if NaN was detected in this epoch
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)  # Already shape [batch_size]
        
        # Compute MSE loss
        loss = criterion(outputs, y_batch)
        
        # Check for NaN in the loss and break training if found
        if torch.isnan(loss):
            print(f"NaN detected in batch loss at epoch {epoch+1}. Stopping training.")
            nan_detected = True
            break

        # Always compute both regularization terms for monitoring/saving
        l2_sum_loss = compute_l2_sum(model, features_normalization)
        l2_mul_loss = compute_l2_mul(model, features_normalization)
        
        # Only apply the regularization that's active for this script
        if λ_sum != 0:
            penalty += l2_sum_loss.item()
            total_loss = loss + λ_sum * l2_sum_loss
        elif λ_mul != 0:
            penalty += l2_mul_loss.item()
            total_loss = loss + λ_mul * l2_mul_loss
        else:
            total_loss = loss
            
        total_loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    # Break from epoch loop if NaN was detected
    if nan_detected:
        break
        
    if scheduler is not None:
        scheduler.step()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    training_losses.append(epoch_loss)
    
    # Check for NaN in the epoch loss
    if torch.isnan(torch.tensor(epoch_loss)):
        print(f"NaN detected in epoch loss at epoch {epoch+1}. Stopping training.")
        break
    
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)
    
    # Check for NaN in validation loss
    if torch.isnan(torch.tensor(val_loss)):
        print(f"NaN detected in validation loss at epoch {epoch+1}. Stopping training.")
        break
    
    if early_stopping:
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            early_stopper.restore_best_model(model)
            break
    
    # Calculate regularization values for this epoch
    model.eval()
    current_l2_sum = compute_l2_sum(model, features_normalization).item()
    current_l2_mul = compute_l2_mul(model, features_normalization).item()
    l2_sum_values.append(current_l2_sum)
    l2_mul_values.append(current_l2_mul)
    
    if (epoch + 1) % interval == 0:
        model.eval()
        ranks = compute_rank_per_layer(model)
        ranks_per_layer_list.append((epoch+1, ranks))
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Penalty: {penalty:.4f}")
        print(f"L2 Sum: {current_l2_sum:.6f}, L2 Mul: {current_l2_mul:.6f}")

actual_epochs = epoch + 1

# Test
model.eval()
correct = 0
total = 0
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)  # Already shape [batch_size]
        
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        
        # Predictions using sign function for MSE with -1/1 labels
        predicted = torch.sign(outputs)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_loss = test_loss / len(test_loader.dataset)
final_accuracy = f'{100 * correct / total:.2f}%'
print(f'Test Loss: {test_loss:.4f}')
print(f'Accuracy on test set: {final_accuracy}')

# Calculate both raw and normalized margins for each dataset
train_raw_margins, train_normalized_margins = calculate_both_margins(model, train_loader_fixed)
val_raw_margins, val_normalized_margins = calculate_both_margins(model, val_loader)
test_raw_margins, test_normalized_margins = calculate_both_margins(model, test_loader)

# Compute margin statistics for raw margins
train_raw_margin_stats = {
    'min': np.min(train_raw_margins),
    'max': np.max(train_raw_margins),
    'mean': np.mean(train_raw_margins),
    'median': np.median(train_raw_margins),
    'std': np.std(train_raw_margins)
}

val_raw_margin_stats = {
    'min': np.min(val_raw_margins),
    'max': np.max(val_raw_margins),
    'mean': np.mean(val_raw_margins),
    'median': np.median(val_raw_margins),
    'std': np.std(val_raw_margins)
}

test_raw_margin_stats = {
    'min': np.min(test_raw_margins),
    'max': np.max(test_raw_margins),
    'mean': np.mean(test_raw_margins),
    'median': np.median(test_raw_margins),
    'std': np.std(test_raw_margins)
}

# Compute margin statistics for normalized margins
train_normalized_margin_stats = {
    'min': np.min(train_normalized_margins),
    'max': np.max(train_normalized_margins),
    'mean': np.mean(train_normalized_margins),
    'median': np.median(train_normalized_margins),
    'std': np.std(train_normalized_margins)
}

val_normalized_margin_stats = {
    'min': np.min(val_normalized_margins),
    'max': np.max(val_normalized_margins),
    'mean': np.mean(val_normalized_margins),
    'median': np.median(val_normalized_margins),
    'std': np.std(val_normalized_margins)
}

test_normalized_margin_stats = {
    'min': np.min(test_normalized_margins),
    'max': np.max(test_normalized_margins),
    'mean': np.mean(test_normalized_margins),
    'median': np.median(test_normalized_margins),
    'std': np.std(test_normalized_margins)
}

print(f'Raw Train Margin Statistics: {train_raw_margin_stats}')
print(f'Raw Validation Margin Statistics: {val_raw_margin_stats}')
print(f'Raw Test Margin Statistics: {test_raw_margin_stats}')

print(f'Normalized Train Margin Statistics: {train_normalized_margin_stats}')
print(f'Normalized Validation Margin Statistics: {val_normalized_margin_stats}')
print(f'Normalized Test Margin Statistics: {test_normalized_margin_stats}')

rank_per_layer = compute_rank_per_layer(model)
print(f'Rank per layer: {rank_per_layer}')

# Calculate final regularization values
final_l2_sum = compute_l2_sum(model, features_normalization).item()
final_l2_mul = compute_l2_mul(model, features_normalization).item()
print(f'Final l2_sum: {final_l2_sum}')
print(f'Final l2_mul: {final_l2_mul}')

config_params = {
    'batch_size': batch_size,
    'max_epochs': epochs,
    'actual_epochs': actual_epochs,
    'learning_rate': learning_rate,
    'lambda_sum': λ_sum,
    'lambda_mul': λ_mul,
    'misclassification_rate': misclassification_rate,
    'features_normalization': features_normalization,
    'resolution_points': resolution_points,
    'seed': seed,
    'out_path': out_path,
    'x_min': x_min,
    'x_max': x_max,
    'y_min': y_min,
    'y_max': y_max,
    'experiment_type': experiment_type,
    'optimizer_choice': optimizer_choice,
    'scheduler_choice': scheduler_choice,
    'bias': bias,
    'init_method': init_method,
    'init_scaling': init_scaling,
    'early_stopping': early_stopping,
    'patience': patience if early_stopping else None,
    'min_delta': min_delta if early_stopping else None,
    'final_accuracy': final_accuracy,
    'test_loss': test_loss,
    'loss_function': 'MSE'  # Added to identify the loss function used
}

timestamp = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")

prefix = ''
if λ_sum > 0:
    prefix = 'S_'
elif λ_mul > 0:
    prefix = 'M_'
if init_scaling != 1.0:
    prefix += f'Scale_[{init_scaling}]_'
if early_stopping:
    prefix += 'ES_'
if misclassification_rate > 0:
    prefix += f'Noise_[{misclassification_rate}]_'

foldername = (
    f'{prefix}_{experiment_type}_MSE_λ_sum[{λ_sum}]_λ_mul[{λ_mul}]_'
    f'[{timestamp}]_bs[{batch_size}]_res[{resolution_points}]_sd[{seed}]_fn[{features_normalization}]_'
    f'x[{x_min},{x_max}]_y[{y_min},{y_max}]_opt[{optimizer_choice}]_init[{init_method}]_scale[{init_scaling}]_'
    f'noise[{misclassification_rate}]_acc[{final_accuracy}]'
)

pickle_filename = f'{prefix}_flatness_measures_MSE_λ_sum[{λ_sum}]_λ_mul[{λ_mul}]_seed[{seed}]_init[{init_method}]_scale[{init_scaling}]_noise[{misclassification_rate}]_{timestamp}.pkl'
full_path = os.path.join(out_path, foldername)
os.makedirs(full_path, exist_ok=True)

results_dict = {
    'rank_per_layer': rank_per_layer,
    'config_params': config_params,
    'ranks_per_layer_over_training': ranks_per_layer_list,
    'training_losses': training_losses,
    'validation_losses': validation_losses,
    'validation_accuracies': validation_accuracies,
    'margins': {
        'train_raw': train_raw_margins,
        'val_raw': val_raw_margins,
        'test_raw': test_raw_margins,
        'train_normalized': train_normalized_margins,
        'val_normalized': val_normalized_margins,
        'test_normalized': test_normalized_margins
    },
    'margin_stats': {
        'train_raw': train_raw_margin_stats,
        'val_raw': val_raw_margin_stats,
        'test_raw': test_raw_margin_stats,
        'train_normalized': train_normalized_margin_stats,
        'val_normalized': val_normalized_margin_stats,
        'test_normalized': test_normalized_margin_stats
    },
    'regularization_values': {
        'final_l2_sum': final_l2_sum,
        'final_l2_mul': final_l2_mul,
        'l2_sum_per_epoch': l2_sum_values,
        'l2_mul_per_epoch': l2_mul_values
    }
}

with open(os.path.join(full_path, pickle_filename), 'wb') as f:
    pickle.dump(results_dict, f)

try:
    print("Calling create_3D_losscape...")
    create_3D_losscape(
        model,
        train_loader_fixed,
        criterion=criterion,  # Pass criterion directly - no wrapper needed
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        num_points=resolution_points,
        output_vtp=True,
        output_path=full_path,
        num_batches=len(train_loader),
    )
    print("create_3D_losscape finished successfully.")
except Exception as e:
    print(f"Error in create_3D_losscape: {e}")


end_time = time.time()
total_runtime_seconds = end_time - start_time
hours = int(total_runtime_seconds // 3600)
minutes = int((total_runtime_seconds % 3600) // 60)
print(f"Total runtime: {end_time - start_time:.2f} seconds")
print(f"Total runtime: {hours} hours {minutes} minutes")