# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:04:17 2018

@author: Nikodemos
"""
import argparse
import pandas as pd
import numpy as np
import random

#data to feed the perceptron model
AND_DATA = {'x1':[0, 0, 1, 1],
            'x2':[0, 1, 0, 1],
            'Yd':[0, 0, 0, 1]}

OR_DATA =  {'x1':[0, 0, 1, 1],
            'x2':[0, 1, 0, 1],
            'Yd':[0, 1, 1, 1]}

XOR_DATA = {'x1':[0, 0, 1, 1],
            'x2':[0, 1, 0, 1],
            'Yd':[0, 1, 1, 0]}

#Hyper parameters for training
LEARNING_RATE = 0.1
THRESHOLD = 0.2
MAX_LOOP = 10000

#step1 : Initialisation
def initialisation():
    """
    Set initial weights w1, w2, ..., wn 
    to random numbers in ther range [-0.5, 0.5].
    """
    _w1 = round(random.uniform(-.5,.5),1)
    _w2 = round(random.uniform(-.5,.5),1)
    return _w1, _w2


#step2 : Activation
def activation(x, w):
    """
    Activate the perceptron by applying inputs 
    x1(p), x2(p), ..., xn(p) and desired output Yd(p)
    """
    sum_matrix = np.matmul(x,w)
    if sum_matrix >= THRESHOLD:
        return 1
    else:
        return 0

#step3 : Weight training
def weight_training(w, x, err):
    """Update the weights of the perceptrons"""
    _delta_w1 = LEARNING_RATE * x[0] * err
    _delta_w2 = LEARNING_RATE * x[1] * err
    
    _next_w1 = round(w[0] + _delta_w1, 1)
    _next_w2 = round(w[1] + _delta_w2, 1)
    
    return _next_w1, _next_w2

#step4 : Iteration
def convergence(df_check):
    """
    Increase iteration p by one, go back to Step 2 and 
    repeat the process until convergence.
    """
    if len(set(df_check['next_w1']))==1 and len(set(df_check['next_w2']))==1:
        return True
    else:
        return False
    
def run_train(data_type=AND_DATA):
    
    #dataframe for train[input value(x1), input value(x2), label(Yd)]
    df_train = pd.DataFrame(data_type)
    #dataframe for display
    df_view = pd.DataFrame(columns=('epoch', 'x1', 'x2', 'Yd', 'w1', 'w2',
                                'Ya', 'error', 'next_w1', 'next_w2'))
    
    #step1 : Initialize weight values(w1, w2)
    w1, w2 = initialisation()
    
    epoch = 0
    global_step = 0
    
    while True:
        epoch += 1
        for index, row in df_train.iterrows():
            w = [w1, w2]
            x = [row['x1'], row['x2']]
            
            #Ya = step[\sum_{i=1}^nx_i(p)w_i(p)-\theta]
            Ya = activation(x, w)
            
            #e(p) = desired Y - actual Y
            err = row['Yd'] - Ya
            
            #w_i(p+1)=w_i(p) + \Delta w_i(p)
            #\Delta w_i(p) = \alpha * x_i(p) * e(p)
            w1, w2 = weight_training(w, x, err)
            
            #append a row to dataframe
            df_view.loc[global_step] = [epoch, x[0], x[1], row['Yd'],
                        w[0], w[1], Ya, err, w1, w2]
            
            global_step += 1
        
        # process is repeated until all the weights converge 
        # to a uniform set of values
        df_check = df_view.loc[df_view['epoch']==epoch, ['next_w1','next_w2']]
        if convergence(df_check) or global_step>MAX_LOOP:
            break
    
    #display output
    print(df_view)
    
    #save a output dataframe to excel
    writer = pd.ExcelWriter('output.xlsx')
    df_view.to_excel(writer, 'train_process')
    df_train.to_excel(writer, 'train_data')
    writer.save()
    
def set_data(args=None):
    if args.data_type:
        if args.data_type==0:
            _dtype = AND_DATA
        elif args.data_type==1:
            _dtype = OR_DATA
        else:
            _dtype = XOR_DATA
    else:
        _dtype=AND_DATA
    return _dtype
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", 
                        help="select a data type[0:AND, 1:OR, 2:XOR]",
                        type=int)
    args = parser.parse_args()
    data_type = set_data(args)

    run_train(data_type=data_type)