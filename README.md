# Efficient Gradient Puzzles

> This is a work in progress. Not all the code is working and intended to be used as is.

Contains a collection of puzzles that help me experiment and learn about:
* Convert `nf4` to Triton
* Make QLoRA Work with FSDP2
* Make torch.compile Work Without Graph Breaks for QLoRA
* Memory Efficient Backprop


## Setup:

Machine type (minimum requirements):
* Linux 5.10.0-33-cloud-amd64 #1 SMP Debian 5.10.226-1
* CUDA 12.6
* At least 400GB of RAM
* At least 4X Tesla T4

Install the dependencies with:
```bash
$ pip install --no-deps -r requirements.txt
```


