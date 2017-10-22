
# coding: utf-8

# In[1]:


import tensorflow as tf


# Lets see how tensorflow works for the MNIST data set. 

# ## Data Import

# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ## What exaclty is MNIST data set ?
# 
# -> The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation).
# -> Every MNIST data point has two parts: an image of a handwritten digit and a corresponding label. Example, the training images are mnist.train.images and the training labels are mnist.train.labels.
# 
# Images : 
# 1. We can flatten this array into a vector of 28x28 = 784 numbers. From this perspective, the MNIST images are just a bunch of points in a 784-dimensional vector space, with a very rich structure.
# 2. Flattening the data throws away information about the 2D structure of the image. Isn't that bad? Well,the simple method we will be using here, a softmax regression (defined below), won't.
# 3. The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]. The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image. Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.
# 
# Labels : 
# 1. For the purposes of this tutorial, we're going to want our labels as "one-hot vectors".
# 2. A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats.

# # # Softmax Regression
# 

# A softmax regression has two steps: 
# 1. first we add up the evidence of our input being in certain classes 
# 2. then we convert that evidence into probabilities.

# To tally up the evidence that a given image is in a particular class, we do a weighted sum of the pixel intensities. The weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor.

# To sum up, softmax can be written as 
#   soft = normalization(exp(evidence_func)) , 
#   where evidence_func = Wx + b

# In[4]:


## Implementing the regression 

x = tf.placeholder(tf.float32, [None, 784]) # Any length and 784 pixels 
W = tf.Variable(tf.zeros([784, 10])) ## 784 pixels with 10 input numbers
b = tf.Variable(tf.zeros([10])) # bias with 10 inputs, each one for one before softmax layer


# In[7]:


# Apply the softmax function
y = tf.nn.softmax(tf.matmul(x, W) + b) 


# ### Training 
# A Loss function is created for the function and to know how good it is.
# One very common, very nice function to determine the loss of a model is called "cross-entropy."cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 

# In[10]:


y_ = tf.placeholder(tf.float32, [None, 10])



# Where y is our predicted probability distribution, and y′ is the true distribution (the one-hot vector with the digit labels)
# 

# In[ ]:


Now lets look at the loss function. First, tf.log computes the logarithm of each element of y. Next, we multiply each element of y_ with the corresponding element of tf.log(y). Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.


# In[11]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


#  Now this function is said to be numerically unstable. Instead we apply, tf.nn.softmax_cross_entropy_with_logits on the unnormalized digits. This function internally computes the softmax activation function.  

# In[33]:


## We will implement that function 
## y_1 = tf.nn.softmax_cross_entropy_with_logits(logits = tf.matmul(x, W) + b, labels = y_)
## cross_entropy_1 =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.matmul(x, W) + b, labels = y_))


# In[18]:


## Building the training model.  

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[21]:


## Let us again tweak with the learning rate. we have used a learning rate of 0.5. Lets use two different learning rates. 

train_step_a = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
train_step_1 = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)


# In[22]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# Let's train -- we'll run the training step 1000 times!

# In[23]:


for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# ## Evaluation 

# How well does our model do?
# 
# Well, first let's figure out where we predicted the correct label. tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth. 

# In[24]:


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# In[25]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[26]:


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# Awesome. We have achieved 92% accuracy with the default cross_entropy and the learning rate of 0.5. 
# 
