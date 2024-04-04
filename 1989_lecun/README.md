# Backpropagation Applied to Handwritten Zip Code Recognition
- LeCun, Y., 1989. Backpropagation Applied to Handwritten Zip Code Recognition, Neural computation 1 (4), 541-551. http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf
- MNIST. http://yann.lecun.com/exdb/mnist/

C implementation of the architecture described in the paper. Converges well, though distinctly from the case described in the paper. This is most likely caused by:
- difference in bias initialization: this implementation initializes biases as zeros while paper doesn't specify.
- difference in learning rate: implementation uses 3e-2 learning rate as done by A. Karpathy in his implementation https://github.com/karpathy/lecun1989-repro, originally I started with lower learning rate and the model got stuck in suboptimal minima, parameter sweep hasn't been attempted.
- difference in dataset: original implementation used MNIST predecessor of smaller size (7,291 : 2,007 train/test split) that I couldn't track down (MNIST has 60,000 training cases and 10,000 test cases), instead subset of MNIST is used matching original training / test data split. I suspect images are of better quality. They were resized from 28x28 pixels to 16x16 pixels accepted by the described architecture.

![Original LeNet (?) architecture](lenet.png)

Each H2 layer runs its 12x filters on selected 8x H1 layers (their outputs aka. feature maps). Paper doesn't cover H1 -> H2 connection scheme - "The eight maps in H1 on which a map in H2 takes its inputs are chosen according a scheme that will not be described here." :( 
This implementation ensures equal propagation of H1 outputs, as demonstrated here:

![H1 -> H2 connections](h1_h2_connections.png)

## Running
```
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gzip -dv train-images-idx3-ubyte.gz
gzip -dv train-labels-idx1-ubyte.gz

cmake .
make -j
./lecun train-images-idx3-ubyte train-labels-idx1-ubyte
```
```
...
Epoch: 29

Avg loss training set [MSE]: 0.015938
Misclassified patterns training set: 2.02%

Avg loss test set [MSE]: 0.036277
Misclassified patterns test set: 5.43%
...
```
