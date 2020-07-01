from threading import Lock
import os
import torch
import torch.nn as nn
from torch import optim
import torch.distributed.rpc as rpc
import torch.nn.functional as F
from Utils import remote_method, get_test_loader, get_model, config, get_worker_gamma, get_ps_lr
import time

# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, world_size, dataset_type, model_type, output_dir='./output.txt'):
        super().__init__()
        self.model_type = model_type
        model = get_model(model_type)()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.world_size = world_size
        self.worker_amount = world_size - 1
        self.dataset_type = dataset_type
        self.test_loader = get_test_loader(self.dataset_type)
        self.output_dir = output_dir
        self.grad_record = []
        self.grad_record_lock = Lock()
        self.all_iteration_num = config['all_iterations'] # len(test_loader.dataset) * config['all_epochs']
        self.iteration_count = 0
        self.worker_K = config['K']
        # self.last_avg_grad = {}
        self.start_time = time.time()

    def process_msg(self, out):
        print(out)
    
    def output_to_file(self, output):
        os.system(f'echo "{output}" >> {self.output_dir}')
    
    def receive_worker_info(self, grad, other_info={}):
        with self.grad_record_lock:
            print('Received grad information from worker.')
            self.grad_record.append(grad)
            print('Saved grad information.')
            # print(other_info)
            assert 'rank' in other_info
            assert 'train_loss' in other_info
            assert 'K' in other_info
            assert 'send_time' in other_info
            worker_rank = other_info['rank']
            worker_K = other_info['K']
            send_time = other_info['send_time']
            self.iteration_count += int(worker_K)
            worker_train_loss = str(other_info['train_loss'])
            output = f'{worker_rank} {worker_K} {worker_train_loss} {time.time()-send_time}'
            self.output_to_file(output)
            # If grad information from all workers are received
            if len(self.grad_record) == self.worker_amount:
                print(f'Received all grad information from {self.worker_amount} workers.')
                avg_grad = self.calculate_avg_grad()
                self.model.update_weights_from_grad(avg_grad, lr=get_ps_lr(self.iteration_count, self.all_iteration_num))
                test_acc, test_loss = self.get_accruracy_loss()
                test_len = len(self.test_loader.dataset)
                output = f'{self.iteration_count} {self.iteration_count/self.all_iteration_num} {test_acc/test_len} {test_loss/test_len} {time.time()-self.start_time}\n'
                self.output_to_file(output)
                self.grad_record = []

    def still_wait(self):
        assert len(self.grad_record) <= self.worker_amount

        if self.iteration_count >= self.all_iteration_num:
            return 'exit' # when reach self.all_iteration_num (namely completed all the epochs), tell the workers to stop

        if len(self.grad_record) == 0:
            return False 
        else:
            return True

    def update_weights(self, new_weights):
        print('Received new weights from worker.')
        self.model.replace_weights(new_weights)
        print('Master weights updated.')


    def fetch_workers_avg_grad(self):
        if len(self.grad_record) == self.worker_amount:
            avg_grad = self.calculate_avg_grad()
            return avg_grad
        else:
            return None
    
    def fetch_weights(self):
        return self.model.get_weights()
    
    def fetch_lr(self):
        return get_ps_lr(self.iteration_count, self.all_iteration_num)
    
    def fetch_gamma(self):
        return get_worker_gamma(self.iteration_count, self.all_iteration_num)

    def fetch_dataset_type(self):
        return self.dataset_type

    def fetch_model_type(self):
        return self.model_type

    def fetch_worker_K(self):
        return self.worker_K

    def calculate_avg_grad(self):
        assert len(self.grad_record) == self.worker_amount
        layer_names = []
        for k in self.grad_record[0]:
            layer_names.append(k)
        
        avg_grad = {}
        for name in layer_names:
            avg_grad_of_name = self.grad_record[0][name]
            for i in range(1, self.worker_amount):
                avg_grad_of_name += self.grad_record[i][name]
            avg_grad_of_name /= 1.0 * self.worker_amount
            avg_grad[name] = avg_grad_of_name
        
        return avg_grad

    def get_accruracy_loss(self):
        self.model.eval()
        correct_sum = 0
        loss_sum = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                out = self.model(data)
                pred = out.argmax(dim=1, keepdim=True)
                pred, target = pred.to(self.device), target.to(self.device)
                correct = pred.eq(target.view_as(pred)).sum().item()
                correct_sum += correct

                if self.model_type == 'ResNet':
                    loss = F.cross_entropy(out, target)
                else:
                    loss = F.nll_loss(out, target) # the avg of the loss value of the batch
                loss_sum += loss * config['batch_size'] # restore the average value to the real sum
        return correct_sum, loss_sum.item()


param_server = None
ps_lock = Lock()

def get_parameter_server(world_size):
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with ps_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(world_size, config['dataset_type'], config['model_type'], output_dir=config['output_dir'])
        return param_server

def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")