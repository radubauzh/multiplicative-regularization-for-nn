import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torchvision.ops import MLP
import warnings

class RegNet:
    def _compute_norms(self, features_normalization):
        norms = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                p_norm = torch.norm(m.weight, 2) ** 2
                if features_normalization == 'f_out':
                    p_norm /= m.weight.size(0)
                elif features_normalization == 'f_in':
                    p_norm /= m.weight.size(1)
                norms.append(p_norm)
        return norms


    def compute_l2_sum(self, depth_normalization, features_normalization, return_norms=False):
        if features_normalization not in [None, 'f_out', 'f_in']:
            raise ValueError(f'Unknown features normalization: {features_normalization}')
        norms = self._compute_norms(features_normalization)
        l2_sum = torch.sum(torch.stack(norms))
        if depth_normalization and len(norms) > 0:
            l2_sum = l2_sum / len(norms)
        if return_norms:
            return l2_sum, [norm.detach().cpu().numpy() for norm in norms]
        return l2_sum

    def compute_l2_mul(self, depth_normalization, features_normalization, return_norms=False):
        if features_normalization not in [None, 'f_out', False, 'f_in']:
            raise ValueError(f'Unknown features normalization: {features_normalization}')
        norms = self._compute_norms(features_normalization)
        l2_mul = torch.prod(torch.stack(norms))

        if depth_normalization and len(norms) > 0:
            l2_mul = torch.pow(l2_mul, 1 / len(norms))
        if return_norms:
            return l2_mul, [norm.detach().cpu().numpy() for norm in norms]
        return l2_mul

class MyMLP(MLP, RegNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.view(x.size(0), -1))


class OverparameterizedCNN(nn.Module, RegNet):
    def __init__(self, in_channels=3, num_classes=1, act="relu", bias='True', batch_norm='False'):
        super().__init__()
        self.batch_norm = batch_norm == 'True'
        self.bias = bias == 'True'
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {act}")

        # Define network layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=self.bias)
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn_fc1 = nn.BatchNorm1d(2048)

        self.fc1 = nn.Linear(512, 2048, bias=self.bias)
        self.fc2 = nn.Linear(2048, num_classes, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_fc1(x)
        x = self.act(x)
        x = self.fc2(x)  # Output a single scalar value for binary classification
        return x

class UnderparameterizedCNN(nn.Module, RegNet):
    def __init__(self, in_channels=3, num_classes=1, act="relu", bias='True', batch_norm='False'):
        super().__init__()
        self.batch_norm = batch_norm == 'True'
        self.bias = bias == 'True'
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {act}")

        # Define network layers with fewer parameters
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=self.bias) 
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn_fc1 = nn.BatchNorm1d(512)  
        
        self.fc1 = nn.Linear(128, 512, bias=self.bias) 
        self.fc2 = nn.Linear(512, num_classes, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_fc1(x)
        x = self.act(x)
        x = self.fc2(x)  
        return x
    

class UnderparameterizedCNN_AvgPool(nn.Module, RegNet):
    def __init__(self, in_channels=3, num_classes=1, act="relu", bias='False', batch_norm='False'):
        super().__init__()
        self.batch_norm = batch_norm == 'False'
        self.bias = bias == 'False'

        # Set activation function
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {act}")

        # Define network layers with fewer parameters
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=self.bias) 
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn_fc1 = nn.BatchNorm1d(512) 
        
        self.fc1 = nn.Linear(128, 512, bias=self.bias)
        self.fc2 = nn.Linear(512, num_classes, bias=self.bias)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 2) 

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 2) 

        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 2)  

        x = self.conv4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_fc1(x)
        x = self.act(x)
        x = self.fc2(x)  
        return x
    
class UnderparameterizedCNN_LinearBias(nn.Module, RegNet):
    def __init__(self, in_channels=3, num_classes=1, act="relu", bias='False', batch_norm='False'):
        super().__init__()
        self.batch_norm = batch_norm == 'False'
        self.bias = bias == 'False'

        # Set activation function
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {act}")

        # Define network layers with fewer parameters
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=self.bias) 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=self.bias) 
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(128)
            self.bn_fc1 = nn.BatchNorm1d(512) 
        
        self.fc1 = nn.Linear(128, 512, bias=True) 
        self.fc2 = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 2) 

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 2) 

        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 2)  

        x = self.conv4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_fc1(x)
        x = self.act(x)
        x = self.fc2(x) 
        return x

class SuperUnderparameterizedCNN(nn.Module, RegNet):
    def __init__(self, in_channels=3, num_classes=1, act="relu", bias='True', batch_norm='False'):
        super().__init__()
        self.batch_norm = batch_norm == 'True'
        self.bias = bias == 'True'

        # Set activation function
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {act}")

        # Define network layers with minimal parameters
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1, bias=self.bias) 
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=self.bias)  
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(4)
            self.bn2 = nn.BatchNorm2d(8)
        
        self.fc1 = nn.Linear(8, 16, bias=self.bias) 
        self.fc2 = nn.Linear(16, num_classes, bias=self.bias) 

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)  
        return x




# Define a layer normalization class
class NormalizedLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, weight_v):
        # Weight Normalization: w = (v / ||v||) * g
        # Normalize the weight vector: v / ||v||
        return weight_v / torch.norm_except_dim(weight_v, 2, -1)

    def right_inverse(self, weight):
        # Inverse normalization (restore original scale): w / ||w||
        weight_v = weight / torch.norm_except_dim(weight, 2, -1)
        return weight_v

# Define a class to add a learnable scalar parameter (rho) to the output
class AddRho(nn.Module):
    def __init__(self, net, init_param=1.0, num_matrices=1) -> None:
        super().__init__()
        self.net = net
        # Initialize rho (scaling factor g) as a learnable parameter
        # ρ=init_param ** num_matrices
        self.rho = torch.nn.Parameter(torch.pow(torch.tensor(init_param), num_matrices))

    def forward(self, *args, **kwargs):
        # Apply the scaling factor to the normalized weight: w = (v / ||v||) * g
        return self.rho * self.net(*args, **kwargs)

# From Weight Normalization Paper, but with init
class PaperWeightNormWithInit(nn.Module):
    def __init__(self, init_param=1.0):
        super().__init__()
        self.init_param = init_param
        self.g = nn.Parameter(torch.tensor(init_param))

    def forward(self, weight_v):
        return self.g * weight_v / torch.norm(weight_v, 2)

    def right_inverse(self, weight):
        weight_v = weight / torch.norm(weight, 2)
        return weight_v

    def get_norm(self):
        return self.g

# From Weight Normalization Paper
class PaperWeightNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight_g, weight_v):
        return weight_g * weight_v / torch.norm(weight_v, 2)

    def right_inverse(self, weight):
        weight_g = torch.norm(weight, 2)
        weight_v = weight / torch.norm(weight, 2)
        return weight_g, weight_v

class PaperWNInitModel(nn.Module, RegNet): #always used this one
    def __init__(self, net, init_param=1.):
        super().__init__()
        self.net = net
        for m in net.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                parametrize.register_parametrization(m, "weight", PaperWeightNormWithInit(init_param))

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def _compute_norms(self, features_normalization):
        norms = []
        for m in self.net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                p_norm = m.parametrizations.weight[0].g ** 2
                if features_normalization == 'f_out':
                    p_norm /= m.weight.size(0)
                elif features_normalization == 'f_in':
                    p_norm /= m.weight.size(1)

                norms.append(p_norm)
        return norms

class PaperWNModel(nn.Module, RegNet):
    def __init__(self, net):
        super().__init__()
        self.net = net
        for m in net.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                parametrize.register_parametrization(m, "weight", PaperWeightNorm())

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def _compute_norms(self, features_normalization):
        norms = []
        for m in self.net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                p_norm = m.parametrizations.weight.original0 ** 2
                if features_normalization == 'f_out':
                    p_norm /= m.parametrizations.weight.original1.size(0)
                norms.append(p_norm)
        return norms

# Define a Weight Normalized Model class
class RhoWNModel(nn.Module):
    def __init__(self, net, init_param=1.):
        super().__init__()
        self.net = net
        self.num_matrices = 0
        # Apply weight normalization to linear and convolutional layers
        for m in net.modules():
            if isinstance(m, nn.Linear):
                parametrize.register_parametrization(m, "weight", NormalizedLayer())
                self.num_matrices += 1
            elif isinstance(m, nn.Conv2d):
                parametrize.register_parametrization(m, "weight", NormalizedLayer())
                self.num_matrices += 1
        # Wrap the network with AddRho
        self.net = AddRho(self.net, init_param, self.num_matrices)

    def forward(self, *args, **kwargs):
        # Forward pass through the wrapped network
        return self.net(*args, **kwargs)

    def compute_l2_sum(self, return_norms=False):
        raise ValueError('L2 Sum not supported for RhoWNModel')

    def compute_l2_mul(self, return_norms=False):
        if return_norms:
            warnings.warn('The only returnable norm is Rho ** 2 itself')
        l2_sum = self.net.rho ** 2
        if return_norms:
            return l2_sum, [l2_sum.detach().cpu().numpy()]
        return l2_sum
    
def set_model_norm_to_one(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # For layers with weight normalization
            if hasattr(m, 'parametrizations') and 'weight' in m.parametrizations:
                # Set g (scaling factor in weight normalization) to 1
                m.parametrizations.weight[0].g.data.fill_(1.0)
            else:
                # For layers without weight normalization, normalize weights directly
                weight_norm = torch.norm(m.weight.data, p=2)
                if weight_norm > 0:
                    m.weight.data.div_(weight_norm)