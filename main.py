import numpy as np
import pandas as pd
import tensorflow as tf

def main():

    training_variables = 8

    array = np.arange(0, 50)
    x15 = np.empty((training_variables,len(array)-training_variables))
    #print(x15)
    for i in range(0,training_variables):
        x15[i] = array[i:-training_variables+i]
    x15 = np.transpose(x15)
    print(x15)


main()