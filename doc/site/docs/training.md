MatConvNet can be used to train models using stochastic gradient
descent and back-propagation.

The following learning demos are provided:

- **MNIST**. See `examples/cnn_mnist.m`.
- **CIFAR**. See `examples/cnn_cifar.m`.
- **ImageNet**. See `examples/cnn_imagenet.m`.

These are self-contained, and include downloading and unpacking of the
data. MNIST and CIFAR are small datasets (by today's standard) and
training is feasible on a CPU. For ImageNet, however, a powerful GPU
is highly recommended.
