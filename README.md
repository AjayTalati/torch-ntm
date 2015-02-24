A Neural Turing Machine in Torch
================================

A Torch implementation of the Neural Turing Machine model described in this 
[paper](http://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne and Ivo Danihelka.

This implementation uses an LSTM controller with one read head and one write head.

## Requirements

[Torch7](https://github.com/torch/torch7) (of course), as well as the following
libraries:

[penlight](https://github.com/stevedonovan/Penlight)

[nn](https://github.com/torch/nn)

[optim](https://github.com/torch/optim)

[nngraph](https://github.com/torch/nngraph)

All the above dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

## Usage

For the copy task:

```
th tasks/copy.lua
```

To train a NTM model for the copy task, and then demo it:

```
th tasks/copy_v2.lua
th tasks/copy_pretrined.lua
```

For the associative recall task:

```
th tasks/recall.lua
```

## To do

1. Document the code more, especially the single read and write case for `ntm_v2.lua`
2. Recreate figures 6 and 4 from the paper
3. Try to implement a feedforward version of the controller
