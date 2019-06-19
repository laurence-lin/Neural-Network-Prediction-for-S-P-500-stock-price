import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import network as net



'''
This code is to test if moving average could help remove noise in data, and improve model predict performance.
Use stock prices prediction for testing, test if moving average smoothing effect helps.
'''

'''
IBM stock price data, contains 360 data samples
'''
ibm = pd.read_csv('ibm_stock.csv')
ibm = np.array(ibm.iloc[:, 1])

'''
Data 2: S & P 500 stock price data, contains 610000 data samples
'''
sp500 = pd.read_csv('all_stocks_5yr.csv')
stock_price = np.array(sp500.iloc[:, 1:5])

total_size = 610000
croped_size = 150000  # don't train all dataset at first
#close_price = stock_price[0:croped_size, 3]
data = stock_price[0:croped_size, 0]

# Data preprocessing
# 80% as training set, 20% as test set
train_size = int(np.floor(croped_size * 0.8))
test_size = int(np.floor(croped_size * 0.2))
train_end = train_size
test_start = train_end
train_set = data[0:train_end]
test_set = data[test_start:]

def create_pipeline(x, predict_slide = 10):
    '''
    x: 1D time series data sequence
    predict_slide: how many days to predict future day
    Create dataset pipeline: 1~n days data => predict n + 1 day data
    return: 
    '''
    data = x[:-1]
    x_data = []
    y_data = []
    
    for i in range(0, len(data) - predict_slide):
        x_data.append(data[i:i + predict_slide])
        y_data.append(data[i + predict_slide])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    return x_data, y_data

'''def predict_day(x, num_of_day = 5):
    num_of_pipe = x.shape[0] - num_of_day
    pipe = []
    for sample in range(num_of_pipe):
        pipe.append(x[sample:sample + num_of_day, :].ravel()) # use ravel to flatten'''

def fillnan(matrix):
    '''
    Fill in NaN missing values
    '''
    for i in range(len(matrix)):
        if np.isnan(matrix[i]):
                matrix[i] = np.mean([matrix[i-1], matrix[i+1]])
                
    return matrix

def MovingAverage(x):
    '''
    Compute moving average curve for array x
    '''
    decay = 0.9
    s_t_list = []
    s_t = x[0]
    s_t_list.append(s_t)
    for i in range(1, len(x)):
        s_t = s_t * decay + (1 - decay) * x[i]
        s_t_list.append(s_t)
    
    return np.array(s_t_list)

def Normalized(data): # normalize data in each column
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / denominator

def Denormalize(input_x):
    Max = np.max(ibm)
    Min = np.min(ibm)
    data_denorm = input_x*(Max - Min) + Min
    
    return data_denorm

''' IBM data
predict_len = 5
ibm_norm = Normalized(ibm)
#ibm_norm = MovingAverage(ibm_norm)
x_train, y_train = create_pipeline(ibm_norm, predict_len)'''

'''SP 500 data'''
# 1. Fill in missing value
train_set = fillnan(train_set)
test_set = fillnan(test_set)
predict_len = 7
sp_norm = Normalized(train_set)
sp_norm = MovingAverage(sp_norm)
x_train, y_train = create_pipeline(sp_norm, predict_len)

x = tf.placeholder(tf.float32, [None, predict_len])
y = tf.placeholder(tf.float32, [None])

predict_y = net.inference(x, reuse = tf.AUTO_REUSE)
with tf.name_scope('loss'):
    loss_op = tf.reduce_mean(tf.square(y - predict_y))

# Compute EMA
decay = 0.9
global_step = tf.Variable(0, trainable = False)
ema = tf.train.ExponentialMovingAverage(decay, global_step)
ema_op = ema.apply([loss_op])

learn_rate = 0.001
with tf.variable_scope('Opt', reuse = tf.AUTO_REUSE):
   optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss_op)
   

train_epoch = 1000
train_batch_size = 250
num_of_batch = int(x_train.shape[0]/train_batch_size) + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print('Train start')
    
    loss_list = []
    mse = []
    for iterate in range(train_epoch):
        
        loss = 0
        for batch in range(num_of_batch):
            offset = batch * train_batch_size
            if x_train[offset:].shape[0] < train_batch_size:
                x_batch = x_train[offset:]
                y_batch = y_train[offset:]
            else:
                x_batch = x_train[offset:offset + train_batch_size, :]
                y_batch = y_train[offset:offset + train_batch_size]
            
            _, err = sess.run([optimizer, loss_op], {x:x_batch, y:y_batch})
            
            loss += err
        
        loss /= num_of_batch
        loss_list.append(loss)
        print('Epoch:', iterate + 1, 'Loss:', loss)
    

    loss_list = Denormalize(np.array(loss_list[:]))
    result = sess.run(predict_y, {x:x_train})  
    result = Denormalize(result)
    y_train = Denormalize(y_train)
    plt.figure(1)
    plt.plot(y_train, label = 'original')
    plt.legend(loc = 'upper right')
    plt.plot(result, label = 'Predict')
    plt.legend(loc = 'upper right')
    
    plt.figure(2)
    plt.title('Loss curve  MSE: %s'%str(loss_list[-1]))
    plt.plot(loss_list)
    
    plt.show()
    
            


    
    
























