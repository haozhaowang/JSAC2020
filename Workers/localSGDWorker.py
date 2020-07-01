import torch
import torch.nn as nn
from torch import optim
import torch.distributed.rpc as rpc
import torch.nn.functional as F
from Utils import remote_method, get_train_loader, get_worker_gamma, get_model
from PS import ParameterServer, get_parameter_server
import os
import threading
import numpy as np
import time

# --------- Workers --------------------
class Worker(nn.Module):
    def __init__(self, rank, world_size):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.param_server_rref = rpc.remote("parameter_server", get_parameter_server, args=(world_size,))
        self.model_type = self.fetch_model_type_from_ps()
        self.model = get_model(self.model_type)()
        self.opt = optim.SGD(self.model.parameters(), lr=self.fetch_lr_from_ps())
        self.train_loader = get_train_loader(self.fetch_dataset_type_from_ps())
        self.train_enum = enumerate(self.train_loader)
        self.rank = rank
        self.K = self.fetch_worker_K_from_ps()
        self.avg_grad = False

        # Thread1: T1, calculate the grad
        # Thread2: T2, process data with master
        self.T1_run = False # semaphore for T1
        self.lock = threading.Lock()
        self.grad_record = {}
        self.train_loss = []

        PS_weights = self.fetch_weights_from_ps()
        self.model.replace_weights(PS_weights)

    def train(self):
        self.model.train()
        i, (data, target) = next(self.train_enum)
        data, target = data.to(self.device), target.to(self.device)
        self.opt.zero_grad()
        model_output = self.model(data)
        if self.model_type == 'ResNet':
            loss = F.cross_entropy(model_output, target)
        else:
            loss = F.nll_loss(model_output, target)
        print(f"training batch {i} loss {loss.item()}")
        loss.backward()
        self.opt.step()

    def send_msg_to_ps(self, msg):
        return remote_method(
            ParameterServer.process_msg,
            self.param_server_rref, msg)

    def still_wait(self):
        return remote_method(
            ParameterServer.still_wait,
            self.param_server_rref)
    
    def fetch_weights_from_ps(self):
        return remote_method(
            ParameterServer.fetch_weights,
            self.param_server_rref)
    
    def fetch_avg_grad_grom_ps(self):
        return remote_method(
            ParameterServer.fetch_workers_avg_grad,
            self.param_server_rref)
    
    def fetch_lr_from_ps(self):
        return remote_method(
            ParameterServer.fetch_lr,
            self.param_server_rref)
    
    def fetch_gamma_from_ps(self):
        return remote_method(
            ParameterServer.fetch_gamma,
            self.param_server_rref)

    def fetch_dataset_type_from_ps(self):
        return remote_method(
            ParameterServer.fetch_dataset_type,
            self.param_server_rref)
    
    def fetch_model_type_from_ps(self):
        return remote_method(
            ParameterServer.fetch_model_type,
            self.param_server_rref)
    
    def fetch_worker_K_from_ps(self):
        return remote_method(
            ParameterServer.fetch_worker_K,
            self.param_server_rref)
    
    def update_ps_weights(self, new_weights):
        return remote_method(
            ParameterServer.update_weights,
            self.param_server_rref, new_weights)
    
    def send_worker_info(self, grad, other_info={}):
        return remote_method(
            ParameterServer.receive_worker_info,
            self.param_server_rref, grad, other_info)

    def still_wait(self):
        ps_result = remote_method(
            ParameterServer.still_wait,
            self.param_server_rref)
        
        if ps_result == 'exit':
            print(f'From master: training process completed.')
            os._exit(0)
        else:
            return ps_result
    
    def calculate_grad(self):
        while True:
            self.lock.acquire()
            if len(self.train_loss) == 0:
                self.model.train()
                for iter in range(self.K):
                    try:
                        _, (data, target) = next(self.train_enum)
                    except StopIteration:
                        self.train_enum = enumerate(self.train_loader)
                        _, (data, target) = next(self.train_enum)
                    data, target = data.to(self.device), target.to(self.device)
                    self.opt.zero_grad()
                    model_output = self.model(data)
                    if self.model_type == 'ResNet':
                        loss = F.cross_entropy(model_output, target)
                    else:
                        loss = F.nll_loss(model_output, target)
                    print(f'loss: {loss.item()}')
                    loss.backward()
                    self.train_loss.append(loss.item())
                    if len(self.grad_record) == 0:
                        self.grad_record = self.model.get_grad()
                    else:
                        self.grad_record = self.model.accumulate_grad(self.grad_record)
                    self.model.update_weights_from_grad(self.model.get_grad(), self.fetch_lr_from_ps())
            self.lock.release()

    def process_grad_and_weights(self):
        while True:
            self.lock.acquire()
            if len(self.train_loss) == self.K:
                ps_weights = self.fetch_weights_from_ps()
                prev_grad = self.grad_record
                self.grad_record = {}
                prev_K = self.K 
                # self.K = 0
                prev_train_loss = self.train_loss
                self.train_loss = []

                if self.avg_grad:
                    for k in prev_grad: 
                        prev_grad[k] /= prev_K 

                self.model.replace_weights(ps_weights)
    
                # if prev_K != 0:
                #     self.model.update_weights_from_grad(prev_grad, self.fetch_gamma_from_ps())
                
                if prev_K != 0:
                    other_info = {}
                    other_info['rank'] = self.rank
                    other_info['train_loss'] = np.mean(prev_train_loss)
                    other_info['K'] = prev_K
                    other_info['send_time'] = time.time()
                    self.send_worker_info(prev_grad, other_info=other_info) 
                    print('Grad has been sent to master.')
                    while self.still_wait(): # Wait until the master receives the grad information from all workers
                        continue
            self.lock.release()

    def start(self):
        T1 = threading.Thread(target=self.calculate_grad, args=())
        T2 = threading.Thread(target=self.process_grad_and_weights, args=())
        T1.start()
        T2.start()
        T1.join()
        T2.join()

# Main loop for trainers.
# round: the worker has to run for how many rounds. in every round, the worker will run for K local_iterations
def run_worker(rank, world_size):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"worker_{rank}",
        rank=rank,
        world_size=world_size)

    print(f"Worker {rank} done initializing RPC")

    w = Worker(rank, world_size)
    w.start()

    rpc.shutdown()