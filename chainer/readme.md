


You must have cupy installed for chainer



[CuPy : NumPy-like API accelerated with CUDA](https://github.com/cupy/cupy)

To install cupy , you must have NCCL installed ,else nccl.h not found !
```
Step 4 安装 NCCL库 
多GPUs进行并行计算，Caffe自带实现. 在多个 GPU 上运行 Caffe 需要使用 NVIDIA NCCL.
$ git clone https://github.com/NVIDIA/nccl.git
$ cd nccl
$ sudo make install -j4
$ sudo ldconfig
```
