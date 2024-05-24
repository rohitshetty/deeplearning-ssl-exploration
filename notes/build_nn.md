# Building neural network

Source code: `./src/build_nn.py`
Source: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

`nn.Flatten` - this takes multi dimensional data, like image and transforms it into a contiguous array

`nn.Linear` - Linear transformation of data. `y=x.A^T+b`. Here `A` is the weight and `b` is the bias. What it does is Matrix multiplication. So, Weight is of size inputXoutput and the bias is same size as output.

`nn.Sequential` is a "Container" of sorts.

Whatever we pass through it creates a network from that - forwarding the input from `.forward` to each layer.
We don't want to use `forward` directly as calling the function apparently calls it internally and does some stuff

`logit`[^1] mathematically is a function where input in range of `[0,1]` can be mapped to `(-inf, inf)`

In case of ML/DL here, it is a "unnormalised" model prediction that is not that useful yet. Because it is raw. We use `softmax`[^2] to map it to a probability (i.e each output is in (0,1) and total sum is 1).

Softmax[^4] converts K real numbers into a Probability distribution (Basically, sum of the outputs would be 1)

argmax[^3] are the input points at which function output value is maximized.

# Building neural network with learning and backpropagation - quickstart

Source: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

[^1]: https://en.wikipedia.org/wiki/Logit
[^2]: https://en.wikipedia.org/wiki/Softmax_function#Softmax_Normalization
[^3]: https://en.wikipedia.org/wiki/Arg_max
[^4]: https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean
