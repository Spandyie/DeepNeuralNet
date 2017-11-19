import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import math
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_matrix(y,c):
    c = tf.constant(c)
    one_hot_matrix = tf.one_hot(y,depth=c,axis=0)   #If indices is a matrix (batch) with shape [batch,features] depth x batch x features if axis == 0
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    yhat = one_hot(c,y.shape[1])
    return yhat

def convert_to_one_hot(Y, C):
    yhat_one  = np.eye(C)[Y.reshape(-1)].T
    return yhat_one

def create_placeholder(n_x, n_y):
    """ This function creates placeholders for X and Y """
    X = tf.placeholder(tf.float32,shape=(n_x,None))
    Y = tf.placeholder(tf.float32, shape=(n_y,None))
    return X, Y

def initialize_parameters(layer_dim):
    """This function will intialize the paramters"""
    #tf.set_random_seed(0)
    L= len(layer_dim)
    parameters={}
    for i in range(1,L):
        parameters["W" +str(i)] = tf.get_variable("W"+str(i), [layer_dim[i],layer_dim[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b" +str(i)] = tf.get_variable("b" +str(i),[layer_dim[i],1],initializer= tf.zeros_initializer())
        assert(parameters['W' + str(i)].shape == (layer_dim[i], layer_dim[i-1]))
        assert(parameters['b' + str(i)].shape == (layer_dim[i], 1))
    return parameters
########################################
def BatchNormalization(inputs, is_training, decay = 0.999,epsilon=1e-3):
    scale =    tf.Variable(tf.ones([inputs.get_shape()[0]]))
    beta =     tf.Variable(tf.zeros([inputs.get_shape()[0]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[0]]), trainable=False)
    pop_var =  tf.Variable(tf.ones([inputs.get_shape()[0]]), trainable=False)
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)
    ################################################3
def forward_propagation(X, parameters):
    """ This function creates the forward propagation model"""
    L= len(parameters)//2
    AL = X
    for i in range(1,L):
        A_Prev = AL
        z = tf.add(tf.matmul(parameters["W"+str(i)], A_Prev),parameters["b"+str(i)]) 
        z=  tf.layers.batch_normalization (z,axis =0, center =True, scale = True, training= True)
        #z=BatchNormalization(z, is_training = True)
        AL = tf.nn.relu(z)
    z = tf.add(tf.matmul(parameters["W" +str(L)],AL),parameters["b"+str(L)])
    #z=BatchNormalization(z, is_training = True)
    z=  tf.layers.batch_normalization (z, axis=0,center =True, scale = True, training= True)
    return z
######################################################
def cost_calculation(Z,Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=labels))
    return cost
##########################
def cost_cal_L2(Z,Y,parameters, lmbda):
    L =len(parameters)//2
    logit =tf.transpose(Z)
    label = tf.transpose(Y)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = label))
    #next step we will be creating regularizers
    regularizer = tf.get_variable("regularizer",shape=[1,1], initializer = tf.zeros_initializer())
    for i in range(L):
        regularizer = regularizer + tf.nn.l2_loss(parameters["W"+str(i+1)])
    loss = loss + lmbda * regularizer
    return loss
#####################################
def load_data():
    path ="cifar-10-batches-py/"
    file_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
    file_names_path = [path + x for x in file_names]
    Data_list=[]
    mini_batch=[]
    mini_batch_labels=[]
    #Xtrain= np.empty()
    for file in file_names_path:
        temp_data = unpickle(file)
        Data_list.append(temp_data)                      # this contain dictionaries
        mini_batch.append(temp_data[b'data'])            # this only contains the Pixel values
        mini_batch_labels.append(np.array(temp_data[b'labels']))
    X_train = np.hstack((mini_batch[0].T,mini_batch[1].T,mini_batch[2].T,mini_batch[3].T,mini_batch[4].T))
    X_test = mini_batch[5].T
    Y_train = np.hstack((mini_batch_labels[0],mini_batch_labels[1],mini_batch_labels[2],mini_batch_labels[3],mini_batch_labels[4]))
    Y_train = Y_train.reshape(1,X_train.shape[1])    
    Y_test = np.array(mini_batch_labels[5])
    Y_test= Y_test.reshape(1,X_test.shape[1])
    return X_train, Y_train, X_test, Y_test
    
def random_sampling(Xtrain,Ytrain, minibatch_size,seed=0):
    np.random.seed(seed)
    m=Xtrain.shape[1]
    random_perm = np.random.permutation(range(m))
    complete_batch = math.floor(m/minibatch_size)    
    ShuffleX= Xtrain[:,random_perm]
    ShuffleY = Ytrain[:,random_perm]
    minibatch=[]
    for i in range(complete_batch):
        tempX= ShuffleX[:, (i * minibatch_size) : ((i+1) * minibatch_size)]
        tempY = ShuffleY[:, (i * minibatch_size) :((i+1) * minibatch_size)]
        miniB =(tempX,tempY)
        minibatch.append(miniB)
    if  m % minibatch_size !=0:
        tempX = ShuffleX[:, complete_batch:m]
        tempY = ShuffleY[:,complete_batch:m]
        miniB = (tempX,tempY)
        minibatch.append(miniB)
    return minibatch
        
    
def accuracy(predictions, labels):
    predictions= np.array(predictions[0])
    return 100* (np.sum(np.argmax(predictions,axis=0)== np.argmax(labels,axis=0)))/predictions.shape[1]
    
       
        
ops.reset_default_graph()
tf.set_random_seed(1)
seed = 0
print_cost = True
learningRate=0.0001
num_epoch=150
Data_list=[]
mini_batch=[]
mini_batch_labels=[]
EpochCostList=[]
Xtrain, Ytrain, Xtest, Ytest = load_data()
Y_train = convert_to_one_hot(Ytrain, 10)
Y_test = convert_to_one_hot(Ytest, 10)
n_x = Xtrain.shape[0]
n_y = len(np.unique(Ytrain))  # shape of output
#layers_dim= [n_x,100,50,50, n_y]  #dimension of neural netowork

h_1 = 200#200 # No neurons in the 1st hidden layer
h_2 = 100#100 # No neurons in the 2nd hidden layer
h_3 = 50  # No neurons in the 3rd hidden layer
layers_dim = [n_x,h_1,h_2,h_3, n_y]
m= Xtrain.shape[1]

X,Y = create_placeholder(n_x,n_y)

parameters = initialize_parameters(layers_dim)

Z = forward_propagation(X,parameters)
#Z_test = forward_propagation(X,parameters)
#cost = cost_calculation(Z,Y)
cost = cost_cal_L2(Z,Y,parameters,0)  # 0 is the L2 regularization
optimizer = tf.train.GradientDescentOptimizer(learning_rate= learningRate).minimize(cost)
#optimizer = tf.train.tf.train.AdamOptimizer(learning_rate= alpha).minimize(cost)
predictions = tf.nn.softmax(Z)               #prediction for the training data
init =tf.global_variables_initializer()           #initialize all the variables
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        epoch_cost=0
        minibatch = random_sampling(Xtrain,Y_train, 1024,seed)
        seed =seed+1
        for batch in minibatch:
            (x,y)= batch
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:x,Y:y})
        epoch_cost+=minibatch_cost/len(minibatch)

        if print_cost == True and epoch%2 == 0:
            print (">> Cost at epoch ("+ str(epoch)+")"+ ">>" + str(np.squeeze(epoch_cost)))
        if print_cost == True and epoch%2 == 0:
            EpochCostList.append(epoch_cost)
    parameters = sess.run(parameters)
    train_prediction = sess.run([predictions],feed_dict={X:Xtrain,Y:Y_train})
    #test_prediction = sess.run([predictions],feed_dict={X:Xtest,Y:Y_test})
    # find the total prediction
#############################################################################
print(">> Training error>> ", accuracy(train_prediction,Y_train))
#print(">> Training error>> ", accuracy(test_prediction,Y_test))

#plot the cost
plt.plot(np.squeeze(EpochCostList))
plt.xlabel("Epoch")
plt.ylabel("Cost")






