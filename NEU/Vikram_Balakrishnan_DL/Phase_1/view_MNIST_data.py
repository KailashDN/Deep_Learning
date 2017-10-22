# Load the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc

# to retrieve batch from train set
batch_xs, batch_ys = mnist.train.next_batch(100)
X=batch_xs[21]
Y=batch_ys[21]

# to display image
X = X.reshape([28, 28])
plt.imshow(X)
plt.gray()
plt.show()

# to save image
plt.savefig("fig.png")

# to convert image to array and verify
f = misc.face()
misc.imsave('face.png', f)
