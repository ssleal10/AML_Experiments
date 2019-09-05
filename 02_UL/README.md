# Unsupervised Learning 

## Homework

In a new file named "USER_Model" (e.g. mfroa_Model), implement a very simple architecture, e.g., composed of a convolutional layer, batch normalization, ReLu and a linear layer. 

The "main.py" code is using CIFAR10 but you will have to change it so it uses MNIST, in order to reduce computational time. 

In order to run the main code, use the following line:
  **python cifar.py --nce-k 0 --nce-t 0.1 --lr 0.03**
Here we set the parameter nce-k to 0, wich controls the number of negative samples. If nce-k sets to 0, the code also supports full softmax learning.
Default values have been modified for simplicity, now you can run the main code without the parser arguments.

**UPDATE (Sept-2)** Now we provide modified main.py and resnet_mnist.py scripts for use with MNIST database.

Once you have your new architecture and dataset, you will have to adapt the model to an autoencoder. The architecture you have right now, could be used as the encoder and you will have to built the decoder. Implement the reconstruction loss and compare your results with the architecture before being converted to an autoencoder, using MNIST. 

**(Optional)** Change the model to implement a variational autoencoder, present samples extracted from the distribution and compare with the first autoencoder you proposed.

## Documents you have to submit:

1. Script with your modified architecture
2. Main script using MNIST
3. PDF document with your results and discussion, show samples of your reconstructions
