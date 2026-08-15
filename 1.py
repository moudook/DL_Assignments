# lib
# numpy, matplotlib,
# code
# train
# convergence test

# import matplotlib
# import numpy

from socket import AF_DECnet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import math
def ActivationFunction(y,B):
    AF = 1 / ( 1 + math.exp( -1 * y * B ))
    # sigmoid AF
    # Suitable when output needs to be interpreted as probabilities
    # saturation possibilities on large data becuase there is a constant -> f * ( 1 - f ) for big values it becomes close to zero so lead to saturation
    # may lead to higher deflection and saturaion and is not adaptive may cause issue if 3 concurrent vlaues are not almost linear
    return AF







# code
df = pd.read_csv("data.csv")

print(df)
print("🚸")

print(df.head())
df["new"]=1

print(df)
# df.drop(new)
# df.drop(columns=new, inplace=True)
# df.drop('new',axis='columns')
# assigning it to df gave the df update insted to sut operatio it ans nevesaving it
df = df.drop('new',axis='columns')

print(df.head())
# df.add('new',axis='columns')
# print(df.head())

#adding column at specifc location
df.insert(loc=0, column='x0',value=1)

print(df.head())
# this is the df now

# now i know how to insert/drop a column at specific 'loc'ation with specific value
# its time to partition the df into data and the label

# data = df['x2']
# dsd = [df['x1'],df['x2']
# data = pd.DataFrame[df['x1','x2']]
#data = df['x1',''x2]
#finally created a df that only have inputs
data = df[['x0','x1','x2']]

print(data.head())
# print(data)

# its time to create label only dataframe
label=df[['label']]
print(label.head())

#separated the input and label
# now i have to see the distrubution of the data using plots

# plot start
# plt.scatter(df['x1'],df['x2'],c=df['label'])
# plt.xlabel("x1")
# plt.ylabel("y1")
# plt.show()
# plot end

data['x1'].max()

# print(df['label'])

print(data.head())

# the relevant comment about the plot
# this plot shows that no classes is overlappable so that we can seprate or use a linear discriminator fucntion here

# Train
# now what we are o ging to do it i haev to make a matrix or df of the saem size as the no. of iptu variable in the data (including the augmented x0) so its easy to multiply to get the dot product / bias because no data line can pass from origin so the bias is needed
# weights = ['x1','x2','x3'] #its a string, cant do integer math
from random import randrange as rdr
weights = [rdr(-2, 2), rdr(-2 ,2), rdr(-2,2)]
# i can use this warmup: random intialize, must be of length same as the data columns (x0,x1,x2 -> 3 weights)

# weights of 3 is created, the loop is the +algorithm:
# this is the training iterations with a loop  i (epoch number) = 1..epoch, outer loop is epoch, inner while traverses the naive data
# for each row: compute y = w * x, compare with label; if wrong, update weights, count the misses, and only the while ends reset the misclass count for the next epoch.
# inner loop uses while because we visit all rows; the for loop sets the number of epochs user defined.
# plot this by now, codes are sequential and complete: import -> data -> train loops -> test prints -> plot

epoch = 200
B = 0.7
#beta that is in the activation funciton
#just defining epoch value

for i in range(1 ,epoch + 1 ):
    #defined number of epoches
    count = 0
    # initialize count = 0 to traverse the rows one by one
    misclass=0
    # resets the misclass count for each epoch
    eta = 1/i
    # the learning rate = decays with epoch number (1/poch number)
    while count < len(data):
        # loop through all the rows of the data
        # correctly input the row one by one
        #predict the class from these weights
        # y = w * data
        y = np.dot(weights,data.iloc[count])
        #np.dot is the doits at product between weights and a data row
        # try to use this style: no error in np.dot for a vector

        # threshold it
        # if y > 0 : class = 1, otherwise class = 0
        # if(y > 0):
        #     y=1
        # else:
        #     y=0
        # Step function activation: threshold at zero
        #
        #
        #
        # sigmoid AF


        #activation funciton
        AF = ActivationFunction(y,B)
        y = AF

        #y comparison
        compare = label.iloc[count].label - y
        # use the . label Value as the dataset label (need .label here)
        # compare is 0 if correct else -1/+1 (update needed)

        if(compare != 0):
            #update weight vector toward the correct value
            AFDash = B * AF * ( 1 - AF )
            #activation fn differentiation
            weights = weights + np.array( eta * compare * AFDash * data.iloc[count] )
            # using vector + vector (each adds the features of the miss)
            misclass += 1
        count += 1

# NOW THE ISSEU I DON KNOW HOW TO MULTIP THE MATRICES IE THE DATAFRAME INTEH PYTHON
# HOW TO UPDATE THE WWIGTH MATEIX
# (solved: np.dot in the loop + vector addition above)

print("test")
print(weights)

inp = [1,2,3]

prod = np.dot(weights,inp)
if prod > 0.5: # find the class and 1 else 0
    print(1)
else:
    print(0)

print(type(weights))

# test plot it now
line = (-1) * (weights[1]* df['x1'] + weights[0])/weights[2]

plt.scatter(df['x1'],df['x2'],c=df['label'])
plt.plot(df['x1'],line,c='red')
plt.xlabel("x1")
plt.ylabel("y1")

plt.plot()
plt.show()
