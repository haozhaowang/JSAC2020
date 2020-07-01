import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- MNIST Network to train, from pytorch/examples -----
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

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