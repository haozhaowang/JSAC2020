import torch
import torch.distributed.rpc as rpc
import torchvision
from torchvision import datasets, transforms
import json
import numpy as np 
from Models.CNN import Net as CNN
from Models.AlexNet import Net as AlexNet
from Models.ResNet18 import Net as ResNet
# import http.client

config = json.load(open('config.json'))

# For convenience, read the config.json from another server 
# def read_config_from_internet(ip, port, save_dir): 
#     global config 
#     conn = http.client.HTTPConnection(ip, port)
#     conn.request("GET", "/")
#     r1 = conn.getresponse()
#     data1 = r1.read() 
#     print(data1)
#     f = open(save_dir, 'wb')
#     f.write(data1)
#     f.close()
#     conn.close()    

#     config = json.load(open('config.json')) # re-read the file content
#     return config

def get_ps_lr(cur_iteration, all_iteration):
    return config['initial_lr']

def get_worker_gamma(cur_iteration, all_iteration):
    return config['gamma']

bit_width = 16
quant_type = np.uint16
dequant_type = np.float32
quant_min = -config['quant_range'] # -0.05
quant_max = config['quant_range'] # 0.05
quant_range = quant_max - quant_min 
bins = np.arange(2 ** bit_width) / (2 ** bit_width)
bins -= 0.5
bins *= quant_range
# print(bins)
def quant(arr):
    norm = np.linalg.norm(arr)
    # norm = 1
    arr /= norm
    return np.digitize(arr, bins).astype(quant_type), norm

def dequant(arr, norm):
    dequanted = []
    for ele in arr:
        dequanted.append(bins[ele])
    dequanted = np.array(dequanted)
    dequanted *= norm
    return dequanted

def quant_grad(grad):
    # return grad
    new_grad = {}
    for k in grad:
        quanted, norm = quant(grad[k])
        new_grad[k] = [quanted, norm]
    return new_grad

def dequant_grad(grad):
    # return grad
    new_grad = {}
    for k in grad:
        quanted_grad = grad[k][0]
        norm = grad[k][1]
        dequanted = dequant(quanted_grad, norm)
        new_grad[k] = dequanted.astype(dequant_type)
    return new_grad


# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def get_test_loader(type):
    if type == 'MNIST':
        return torch.utils.data.DataLoader(
                    datasets.MNIST(
                        '../data',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.1307,),(0.3081,))])),
                    batch_size=config['batch_size'], shuffle=True,)
    elif type == "Cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                                shuffle=False, num_workers=2)
        print('Cifar test set size: ', len(testloader.dataset))
        return testloader
    else:
        raise 'Unsupported Dataset Type'


def get_train_loader(type):
    if type == 'MNIST':
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=config['batch_size'], shuffle=True,)
    elif type == 'Cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                                shuffle=True, num_workers=2)
        print('Cifar train set size: ', len(trainloader.dataset))
        return trainloader
    else:
        raise 'Unsupported Dataset Type'

def get_model(type):
    if type == 'CNN':
        return CNN
    elif type == 'AlexNet':
        return AlexNet
    elif type == 'ResNet':
        return ResNet
    else:
        raise 'Unsupported Model Type'