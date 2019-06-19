# Neural-Network-Prediction-for-S-P-500-stock-price
Simple neural network prediction for S &amp; P 500 stock price, and other stock dataset

Dataset: 

S & P 500 stock price, contains 610000 data samples

Train set: 120000 

Test set: 3000

Model:

3 hidden layer neural network + 1 output layer

hidden neuron = 30

output neuron = 1

Training Design:

Epoch: 200

learn rate = 0.001

weight initialization: [0, 0.001]

predict feature days: 10, use 10 days of data to predict the 11th days value

Features: only use open price as input feature


Result:

RMSE = 17
MSE = 307

Training Loss curve:

![img](https://github.com/laurence-lin/Neural-Network-Prediction-for-S-P-500-stock-price/blob/master/SP500_loss.png)

Training performance: predict on training set

![img](https://github.com/laurence-lin/Neural-Network-Prediction-for-S-P-500-stock-price/blob/master/SP500_performance.png)

Testing performance: predict on testing set

![img](https://github.com/laurence-lin/Neural-Network-Prediction-for-S-P-500-stock-price/blob/master/test%20result_2.png)


I found that model selection is important, originally set fewer neurons and get bad reesult.
