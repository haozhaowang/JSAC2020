# JSAC2020
Code for Special Issue of JSAC-2020

* Guide for libs
     * Python 3.6
      * torch 3.5.0
      * torchvision

* Guide for Hyper-parameters
    * Configuration on `config.json` file
    * Starting PS with example like
    ```
    python3 main.py --ws 2 --psip 172.16.167.32 --psport 8880 --algorithm LOSP --rank 0
    ```
    * Starting worker with example like
    ```
    python3 main.py --ws 2 --psip 172.16.167.32 --psport 8880 --algorithm LOSP --rank 1
    ``` 
    * Guide for hyper-parameters
        * `--ws` world size，number of PS + worker
        * `--psip` PS's ip
        * `--psport` PS's port
        * `--algorithm` : localSGD、OSP、LOSP
        * `--rank` : PS = 0，worker = 1、2、3、... 、(worldsize-1)
    * `config.json`
        * `model_type` : CNN、AlexNet、ResNet. Define on functions `get_train_loader, get_test_loader` of `Utils.py`
        * `dataset_type` : MNIST、Cifar10
        * The data size of `Cifar10` is 50000. So, 100 epochs should be 50000 / batch_size * 100. When batch size is 32，all_iterations should be set as 156250
        * `MNIST` = 60000， 100 epochs --> iterations = 60000 / batch_size * 100
    * Learning rate eta and compensated parameter gamma
        * In `get_ps_lr` and `get_worker_gamma` of `Utils.py`

 * Guide for Files
     * `Model.py` Neural networks
     * `PS.py` Program of Parameter Server. Barrier, synchronization, broadcast.
     * `Worker.py` Program of worker.
     * `Utils.py` 
         * Dataset loader: `test_loader` and `train_loader`
         * strategies for eta and gamma
             * `get_ps_lr` for defining learning rate eta
             * `get_worker_gamma` is similar to eta
             
* The code for ICPP is in `https://github.com/AragornThorongil/ICPP`
