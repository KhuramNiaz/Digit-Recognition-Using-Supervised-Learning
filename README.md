# Digit-Recognition-Using-Supervised-Learning
Handwritten Digit Recognition Using SVM, KNN and Neural Network and analyze the results and based on their accuracy try to figure out which one is more suitable and achieve near-human performance.

# Introduction
Automatic Handwritten Digits Recognition (HDR) is the process in which a machine trains itself
to interpret handwritten digits from images. There are many approaches for handwritten digit
recognition. In this assignment we used KNN, SVM with HOG and Forward Feed Neural
Network. Then we did a comparison of all these three classifiers in terms of performance,
accuracy, time, sensitivity, positive productivity. For evaluation purposes we use MNIST dataset.
The major problem in the Digit Recognition system is that different people write the same
number differently. Also, there is confusion between different numbers i.e. 1 and 7 may look the
same. Likewise 5 and 6, 3 and 8, and 1 and 7 may seem similar depending on handwriting. For
this system, we used python, openCV and sklearn to run classification and read the dataset. We
used MNIST dataset from the kaggle competition for training and evaluation for classification.

# Goal
The main goal of this report is to compare the different classifiers with different parameters and
analyze the results and based on their accuracy try to figure out which one is more suitable and
achieve near-human performance.
MNIST Dataset:
![image](https://user-images.githubusercontent.com/52096838/121337676-37695900-c936-11eb-8100-75a4c44c8619.png)

Each image in this dataset is a 28 by 28 pixel square (748 pixels total)
![image](https://user-images.githubusercontent.com/52096838/121337781-52d46400-c936-11eb-976f-6730310774c5.png)

# CLASSIFIERS
In this report, we have used 3 algorithms of machine learning for making predictions and
accuracy which include:
● KNN (K nearest neighbors)
● Forward Feed Neural Network
● SVM (Support Vector Machine) with HOG Features

# KNN (K nearest neighbors)
KNN is a lazy algorithm which runs until the last stage of classification. It uses the local search
approach to process the value of K nearest neighbors. It is a simple algorithm with no earlier
explicit training and does not generalize the training data set.
In this algorithm, we first calculate the distance between the test data point and all labeled data
points. Then we order the labeled data points in increasing order of distance metric. Then select
the top K labeled data points and look at class labels. The class labels that the majority of these K
labeled data points have is assigned to test data points.The Euclidean distance is the most common distance metric used in low dimensional data sets.
So, we have used the Euclidean function for computing distance. Some other parameters we have
used are as follows:
Algorithm = ‘auto’, leaf_size = 30, metric = ‘minkowski’, metric_params = None, n_jobs
= None, n_neighbors = 1, p= 2, weights = ‘uniform’
Since the data set was large and knn takes more time we divided the whole data into 14 batches
each of 2000 images, so that our program won't crash.
Train Time < 1 minute
Test Time = 51.3+52.7+53.7+52.4+50.9+50.8+ 50.5+ 55.1+ 51.9+54.1+53.4+56.5+51.7+53.8
= 738.8 secs = 12.3 minutes
Best Value of K:
It's necessary to choose the value of K wisely since by changing the value of K, the output for
the test data point can also vary. We have tried different values of k and realized that it gave
more accuracy at k =1, So we have used the k value equal to 1 which gave the highest accuracy
of 97.1 %.
K = 1 => Accuracy = 97.1 %
KNN basic implementation from:
https://www.kaggle.com/snshines/knn-from-scratch-in-python-at-97-1

# Forward Feed Neural Network:
A forward feed neural network is an artificial neural network in which nodes are connected to
each other such that no cycle is formed. The information moves from input nodes to output
nodes through hidden nodes (if exist).
We first randomly initialize the weights. Then implement forward propagation to achieve
hθ(x(i)). Then we compute the cost. After that evaluate backpropagation to compute partial
derivatives and use gradient checking to confirm that backpropagation is working fine. Then
disable gradient checking. Use gradient descent or any built-in optimization function to minimize
the cost function with weights of theta Θ.

So, In the first hidden layer we have used 32 nodes with a total number of inputs as 784 which
means all pixels of image. Activation function for the first hidden layer is "sigmoid" and"softmax'' for the output layer. We have used Adam Optimizer for compiling. We have used only one hidden layer. 20 epoches are taken and have used metrics as accuracy.
![image](https://user-images.githubusercontent.com/52096838/121338105-a050d100-c936-11eb-9546-cc4ef2ca5fd1.png)

FFNN implementation from:
https://medium.com/random-techpark/simple-feed-forward-neural-network-code-for-digital-handwritten-digit-recognition-a234955103d4

# SVM (Support Vector Machine) with HOG Features
SVM falls into the category of supervised learning, and with the bonus of classification as well
as regression problems. To calculate HOG descriptor we have used following values:
![image](https://user-images.githubusercontent.com/52096838/121338237-bfe7f980-c936-11eb-9c98-2df3dda52da4.png)

The HOG descriptor defined above can be used to compute the HOG features of an image. To
train a model that will classify the images in our training set we have used Support Vector
Machines (SVM) as our classification algorithm and trained our model. We choose the C that
provides the best classification on a held out test set, in our case 12.5 gave the best results. The
parameter Gamma controls the stretching of data in the third dimension. It helps in classification
but it also distorts the data. So it is also necessary to choose its value wisely. So we have used the
value of Gamma = 0.50. And it took 3 minutes and 29 secs to train it.
Then we tested our data using open CV with two kernels with CV = 5:
1) Linear Kernel (Accuracy = 100 %, Train time = 24 secs)
2) RBF Kernel ( Accuracy = 100 %, Train time = 42 secs )
SVM basic implementation from https://www.kaggle.com/xianng/deskew-hog-svmComparison:
We have compared these classifiers by their accuracies, train and test time taken by them and
scores on a kaggle board for a better conclusion, which are as follows:

![image](https://user-images.githubusercontent.com/52096838/121338306-d1310600-c936-11eb-9224-1959adc27a88.png)
![image](https://user-images.githubusercontent.com/52096838/121338515-050c2b80-c937-11eb-9744-368b3f085308.png)

It can be clearly seen that SVM with RBF kernel got the highest score, followed by SVM with
Linear kernel, KNN and FFNN.

Train Time Comparison:
![image](https://user-images.githubusercontent.com/52096838/121338594-15240b00-c937-11eb-975f-580b3b418100.png)
Test Time Comparison:
![image](https://user-images.githubusercontent.com/52096838/121338659-24a35400-c937-11eb-8adb-7a16217ebe4f.png)
KNN classifier have taken much more time in testing as compared to others.

# Conclusion
Accuracy and time are the most significant and decisive factors here. We have compared both
accuracy and time for each of the classifiers and resulting data is represented through graphs
above. According to above data, SVM out performs KNN(97.1%) and Neural Network
approach(97%). SVM with Linear and RBF kernel, both had accuracy of 100% but in terms of
Score RBF kernel outperformed the Linear by getting the highest score of 0.99157 followed by
SVM Linear with 0.98685, KNN with 0.97114 and FFNN with 0.90800 score. Moreover in
terms of train and test time KNN took the highest time (738.8 secs) in testing data and FFNN the
lowest of 10 seconds. To conclude, based on training time, testing time and most importantly
Accuracy SVM with RBF kernel using HOG features is the best one for handwritten digits
recognition. Moreover, results can be improved by using more training dataset and changing the
values of different parameters used in each classifier.

# Tools
Jupyter Notebook
