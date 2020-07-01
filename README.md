# JSAC2020
Code for Special Issue of JSAC-2020

# OSP

* 任务
    * 测试各个算法在 MNIST CNN, AlexNet on Cifar10，Restnet18 on Cifar10 上的表现
    * worker 数选择 8 和 16
    * 应当保存的输出（输出到文件是由 PS 进行的）
        * 每个 worker 上一批进行的 local iteration 数，在训练集上的平均 loss
        * 全局执行的 iteration 数（namely 所有 workers 执行的 local iteration 数）
        * PS 在测试集上的 accuracy 和 loss
        * 训练和通信时间、Worker 发给 PS 的数据量（之前的实验没有记录，但是之后的得记录，因为要说明 DQLOSP 的通信时间缩短了，而且通信数据量变少了）

* new! 使用说明
    * 现在不需要更改 worker 上的 config，只需要更改 PS 上的 `config.json` 就行
    * 但是在运行时需要手动传入一些参数
    * 运行 PS 时
    ```
    python3 main.py --ws 2 --psip 172.16.167.32 --psport 8880 --algorithm LOSP --rank 0
    ```
    * 运行 worker 时
    ```
    python3 main.py --ws 2 --psip 172.16.167.32 --psport 8880 --algorithm LOSP --rank 1
    ``` 
    * 参数说明
        * `--ws` world size，即 PS + worker 的个数
        * `--psip` PS's ip
        * `--psport` PS's port
        * `--algorithm` 可以取 localSGD、OSP、LOSP，在 `main.py` 中进行判断
        * `--rank` 和之前一样，PS 是 0，worker 是 1、2、3、... 、(worldsize-1)
    * `config.json` 字段说明
        * 大部分和之前一样，在其中不用再指定 PS 和 worker 的地址了
        * `model_type` 可以选 CNN、AlexNet、ResNet （注意大小写），这个部分在 `Utils.py` 的 `get_train_loader, get_test_loader` 中定义
        * `dataset_type` 可以选 MNIST、Cifar10，这个部分在 `Utils.py` 的 `get_model` 中定义
        * `Cifar10` 的训练集大小是 50000，所以 100 个 epochs 应该是 50000 / batch_size * 100，当batch size 是 32 时，all_iterations 应当是 156250
        * `MNIST` 训练集 60000，所以 100 epochs 对应 iterations 是 60000 / batch_size * 100，当 batch_size 取 32 时
    * 如果想要动态调整学习率和 gamma
        * `Utils.py` 里的 `get_ps_lr` 和 `get_worker_gamma`
    * 需要跑的实验
        * 都是 8 个 worker
        * 三个实验
            * MNIST + CNN
            * AlexNet + Cifar10
            * ResNet + Cifar10


* 使 用 说 明
    * worker 将本次计算的梯度更新情况发送给 PS，并等待 PS 完成汇总后才会发送下一次的计算结果
    * 依赖
        * Python 版本应该要大于 3.6（还是 3.5 来着），因为我用了比较新的语法
        * 需要安装的库
            * torch 3.5.0
            * torchvision
    * 运行方法
        * PS (master): `python3 main.py --rank 0`
        * worker: `python3 main.py --rank 1/2/3/4/5...`
    * 配置文件
        * `config.json`
        * 运行前记得更改每个节点上的 `config.json`，我现在是通过 `get_config_from_internet` 在每次运行前获取配置文件
            * 在某台服务器上用 Node.js 运行 `jsonServer.js` 搭建了一个非常简易的 HTTP Server 来支持每个节点动态获取配置文件
        * 或者你可以修改成通过运行时传入参数来指定学习率、gamma、master 的地址等等
    * 文件夹结构
        * `Model.py` 网络结构定义，我还加了一些获取参数、更换参数、更换梯度、手动进行梯度更新的类成员函数
        * `PS.py` 定义参数服务器，其中需要完成对各个 worker 的参数更新情况汇总、对各个 worker 进行同步（每个 worker 会通过 still_wait 来的值是否需要继续运算/是否需要发送下一批参数更新数据）
        * `Worker.py` 定义 worker
        * `Utils.py` 
            * 定义了 `test_loader` 和 `train_loader`
            * 学习率、gamma 的调整策略
                * `get_ps_lr` 定义学习率，两个参数分别是当前全局执行的总 iteration 数和应当执行的所有 iteration 数
                * 可以直接返回 `config['initial_lr']`，或是根据当前 iteration 数来做动态调整
                * `get_worker_gamma` 同理
            * quantization
                * 调用了 `numpy.digitize` 来完成
                * 可以修改 `Utils.py` 中的 `bit_width` 和 `quant_type` 来调整 quantization 后的数据位宽

* 其他说明
    * Bug
        * 目前 PS 输出到文件时，文件名可能会出错（比如上一次输出到 `file1`，这一次在 config 里面改了输出文件名为 `file2`，但是这一次仍然可能会输出到 `file1`）
            * 因为最终执行输出操作用到的是 `echo >>`，所以不会发生数据覆盖，只是可能需要手动把两次训练的输出分离一下
        * 训练结束时，PS 和部分 Worker 不会自动退出，需要手动退出一下
        * *最重要*的 BUG（我太菜了）：DQLOSP 算法里，worker 应当每次从 master（PS）拉取全局平均参数更新情况，并更新本地的 global_model，而这个 global_model 和 PS 上的 model 应该是一致的，但是我可能哪里写错了，这两个模型参数不一致 
            * 因为之前不需要记录通信时间什么的，所以我就直接从 PS 拉过来模型的参数，虽然这样不影响最终的模型效果，但是会影响通信时间
            * 现在的 code：
            ```Python
            # 现在是直接向 PS 获取模型的各个参数，从而保证一致性
            self.model.replace_weights(self.fetch_weights_from_ps()) # 162 行
            ```
            * 应当执行：
            ```Python
            # 但是应当是 PS 和 Worker 各自维护本地模型（PS.model 以及 Worker.global_model），而且这两个模型应当是一致的
            # 每次 Worker 都先复制 self.global_model 到 self.model，然后在 self.model 上进行计算
            self.model.replace_weights(self.global_model.get_weights())
            ```
            * 猜测问题所在：可能是 PS 发送的 `last_avg_grad` 不正确
