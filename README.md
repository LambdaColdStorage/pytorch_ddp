# Simple PyTorch Distributed Training (Multiple Nodes)

## Preparations

Do the following steps on all nodes:

__Clone Repo__

```
git clone https://github.com/lambdal/pytorch_ddp
cd pytorch_ddp
```

__Download the dataset on each node before starting distributed training__

```
mkdir -p data
cd data
wget -c --quiet https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
cd ..
```

__Creating directories for saving models before starting distributed training__

```
mkdir -p saved_models
```


## Run 

Do the following steps to run 2x Nodes distributed training (3xGPUs per node)

__Node One__

```
NCCL_DEBUG=INFO NCCL_ALGO=Ring NCCL_NET_GDR_LEVEL=4 python3 -m torch.distributed.launch \
--nproc_per_node=3 --nnodes=2 --node_rank=0 \
--master_addr="xxx.xxx.xxx.xxx" --master_port=1234 \
resnet_ddp.py \
--backend=nccl
```

__Node Two__

```
NCCL_DEBUG=INFO NCCL_ALGO=Ring NCCL_NET_GDR_LEVEL=4 python3 -m torch.distributed.launch \
--nproc_per_node=3 --nnodes=2 --node_rank=1 \
--master_addr="xxx.xxx.xxx.xxx" --master_port=1234 \
resnet_ddp.py \
--backend=nccl
```

Note: `backend` options are `nccl`, `gloo`, and `mpi`. "By default for Linux, the Gloo and NCCL backends are built and included in PyTorch distributed (NCCL only when building with CUDA). MPI is an optional backend that can only be included if you build PyTorch from source". More details can be found [here](https://pytorch.org/docs/stable/distributed.html)

## Credit

The `resnet_ddp.py` script is written by [Lei Mao](https://leimao.github.io/blog/PyTorch-Distributed-Training/). 