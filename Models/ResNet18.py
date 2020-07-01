import torch
import torch.nn as nn
import torch.nn.functional as F

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class Net(nn.Module):
    def __init__(self,num_classes=10):
        super(Net, self).__init__()
        block = ResidualBlock
        layers = [2, 2, 2]
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    # model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


    def get_grad(self):
        params = {}
        for name, parameters in self.named_parameters():
            params[name] = parameters.grad.cpu().numpy()
        return params
    
    def accumulate_grad(self, prev_grad):
        accumulated_grad = {}
        for name, parameters in self.named_parameters():
            param = parameters.grad.cpu().numpy()
            accumulated_grad[name] = param + prev_grad[name]
        return accumulated_grad

    def replace_grad(self, to_replace_grad):
        is_cuda = next(self.parameters()).is_cuda
        # If the model is on GPU, then move it to CPU, 
        # otherwise we can not convert the grad of each layer to numpy
        if is_cuda:
            self.to('cpu')

        for name, parameters in self.named_parameters():
            # param = parameters.grad.numpy()
            parameters.grad = torch.from_numpy(to_replace_grad[name])
        
        # If the model was previously on GPU, then move it back to GPU
        # ATTENTION: WE ASSUME THAT THERE IS ONLY ONE GPU ON THE MACHINE
        if is_cuda:
            self.to('cuda:0') # Move the model to the only GPU on the machine
    
    def clear_grad(self):
        is_cuda = next(self.parameters()).is_cuda
        # If the model is on GPU, then move it to CPU, 
        # otherwise we can not convert the grad of each layer to numpy
        if is_cuda:
            self.to('cpu')

        for _, parameters in self.named_parameters():
            param = parameters.grad.numpy()
            param[param!=0] = 0 # Set all elements in param to 0
            parameters.grad = torch.from_numpy(param)
        
        # If the model was previously on GPU, then move it back to GPU
        # ATTENTION: WE ASSUME THAT THERE IS ONLY ONE GPU ON THE MACHINE
        if is_cuda:
            self.to('cuda:0') # Move the model to the only GPU on the machine
    
    def get_weights(self):
        weights = {}
        for name, parameters in self.named_parameters():
            weights[name] = parameters.cpu().detach().numpy()
        return weights
    
    def replace_weights(self, to_replace_w):
        is_cuda = next(self.parameters()).is_cuda
        # If the model is on GPU, then move it to CPU, 
        # otherwise we can not convert the grad of each layer to numpy
        if is_cuda:
            self.to('cpu')

        for name, parameters in self.named_parameters():
            parameters.data = torch.from_numpy(to_replace_w[name])
        
        # If the model was previously on GPU, then move it back to GPU
        # ATTENTION: WE ASSUME THAT THERE IS ONLY ONE GPU ON THE MACHINE
        if is_cuda:
            self.to('cuda:0') # Move the model to the only GPU on the machine
    
    def update_weights_from_grad(self, new_grad, lr=0.001):
        is_cuda = next(self.parameters()).is_cuda
        # If the model is on GPU, then move it to CPU, 
        # otherwise we can not convert the grad of each layer to numpy
        if is_cuda:
            self.to('cpu')

        for name, parameters in self.named_parameters():
            grad = new_grad[name]
            origin_param = parameters.detach().numpy()
            updated_param = origin_param - grad * lr
            # if name == 'conv1.bias':
            #     print(updated_param)
            parameters.data = torch.from_numpy(updated_param)
        
        # If the model was previously on GPU, then move it back to GPU
        # ATTENTION: WE ASSUME THAT THERE IS ONLY ONE GPU ON THE MACHINE
        if is_cuda:
            self.to('cuda:0') # Move the model to the only GPU on the machine