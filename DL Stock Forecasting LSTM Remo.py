import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, layers
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import time, math, statistics
import pprint as pprint

def main():
    _start_time = time.time()

    # MODEL MIT 7 INPUTS: MIT 7 TAGEN LAG, MIT 6 TAGEN LAG usw. ... , MIT 1 TAG LAG
    ticker = ["JNJ"]

    # HYPERPARAMETERS

    architectures = ["Dense", "CNN", "RNN", "GRU", "LSTM", "DBN", "DWNN", "CAE", "bidirectional"]
    units = [1, 2, 4, 8, 16, 32, 64, 128] #units per layer
    activation_functions = ["sigmoid", "tanh", "relu", "leaky_relu", "swish", "softmax", "bipolar_sigmoid", "hard_sigmoid", "sign"]
    optimizers = ["SGD", "rmsprop", "adam"]

    years = 10
    start_date = (datetime.today() - relativedelta(years=years, days=1)).strftime('%Y-%m-%d')
    end_date = (datetime.today() - relativedelta(days=1)).strftime('%Y-%m-%d')
    X = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    X,y = X["Close"], X["Close"]
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    X, y = X.to_numpy(), y.to_numpy()
    # for i in X:
    #     X = [(i-min(X)/(max(X)-min(X)))]
    # print(X)


    training_variables = 5 ##Hyperparameter wie viele input variables

    # array = np.arange(0, 50) #zum probieren
    X15 = np.empty((training_variables,len(X)-training_variables))
    for i in range(0,training_variables):
        X15[i] = X[i:-training_variables+i]
    y = X[training_variables:].copy()
    X15 = np.transpose(X15)
    # print(X15[3][training_variables-1])

    # # print(np.shape(X), X)
    # X5, X4, X3, X2, X1, y = X[:-5].copy(), X[1:-4].copy(), X[2:-3].copy(), X[3:-2].copy(), X[4:-1].copy(), X[5:].copy()
    # # print(X4)
    # X16 = np.transpose([X5, X4, X3, X2, X1])
    # print(X16)
    # X17 = X16-X15
    # print(X17)


    split0, split1, split2 = .4 , .8, .9

    squared_errors = 0
    correct_direction = 0
    incorrect_direction = 0
    total_return = 1
    total_return_list = []
    total_return_final_list = []
    prediction_list = []
    actual_list = []
    evaluate_list = []

    for j in range (0,5):
        total_return_final_list.append(total_return)
        total_return = 1
        correct_direction = 0
        incorrect_direction = 0

        for i in range(0,int(len(X15)*(1-split2)+1)):
            total_return_list.append(total_return)
            X_train, X_val, X_test = np.array(X15[int(len(X15)*(split0+.1*j) + i):int(len(X15)*(split1+.1*j)+i)]), np.array(X15[int(len(X15)*(split1+.1*j)+i):int(len(X15)*(split2+.1*j)+i)]), np.array(X15[int(len(X15)*(split2+.1*j)+i):int(len(X15)*(split2+.1*j)+i+1)])
            y_train, y_val, y_test = np.array(y[int(len(y)*(split0+.1*j) + i):int(len(y)*(split1+.1*j)+i)]), np.array(y[int(len(y)*(split1+.1*j)+i):int(len(y)*(split2+.1*j)+i)]), np.array(y[int(len(y)*(split2+.1*j)+i):int(len(y)*(split2+.1*j)+i+1)])
            
            # model = Sequential([
            # layers.InputLayer((0,training_variables)),
            # layers.LSTM(units=64),
            # layers.Dense(units=1, activation="softmax")
            #             ])
            
            model = Sequential([
            layers.InputLayer((training_variables,1)),
            layers.LSTM(64),
            layers.Dense((1))
                        ])

            # model = Sequential()
            # model.add(layers.InputLayer((training_variables,1)))
            # model.add(layers.LSTM(64))
            # model.add(layers.Dense((1), activation="relu"))
            
            # cp1 = ModelCheckpoint("model/", save_best_only=True)
            model.compile(
                    optimizer = tf.optimizers.Adam(learning_rate=0.01),
                    loss = "mse"
                    )
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)#, callbacks=[cp1])
            # model = load_model('model/')
            prediction =  model.predict([X_test])
            model.evaluate(X_test, y_test)
            evaluate_list.append(model.evaluate)
            # print(f"Prediction: {prediction}, Actual: {y_test}")
            prediction_return = (prediction - X_test[0][training_variables-1]) / X_test[0][training_variables-1]
            stock_return = (y_test - X_test[0][training_variables-1]) / X_test[0][training_variables-1]
            direction_check = prediction_return / stock_return #Wenn Positiv gleiche Richtung, wenn negativ falsche Richtung
            squared_errors += (y_test - X_test[0][training_variables-1])**2
            mse = 1/(i+1) * squared_errors
            # print(f"X_test -1: {X_test[-1]}")
            # print(f"X15 letzter Tag davor: {X_test[0][training_variables-1]}")
            # print(f"prediction return {prediction_return}")
            # print(f"stock return: {stock_return}")
            # print(f"direction_check: {direction_check}")
            print(f"Test Round: {j+1}, Test Day: {i+1}")

            if direction_check >= 0:
                correct_direction += 1
                total_return = total_return * (1 + abs(stock_return)) #return errechnen. hier: wenn die richtung stimmt, dann einfach den return vom tag nehmen
            elif direction_check < 0:
                incorrect_direction += 1
                total_return = total_return * (1 - abs(stock_return))
            
            total_return_list.append(total_return)
            prediction_list.append(prediction)
            actual_list.append(stock_return)
            stock_total_return = y_test/y[int(len(y)*split2)]

            print(f"Mein total return ist: {np.round((total_return-1)*100, 2)}%, , Stock Total Return: {np.round((stock_total_return-1)*100, 2)}%")
            print(f"MSE: {mse}")
            print(f"Final returns: {total_return_final_list}")
            print(f"{np.round(correct_direction / (correct_direction + incorrect_direction)*100, 4)}% der Predictions gingen in die richtige Richtung!")
            print(f"Zeit für den Tag: {np.round((time.time() - _start_time)/(i+1),2)} Sekunden")

    total_return_final_list.append(total_return)
    print(f"Total returns (final list): {total_return_final_list}")
    print(f"Model Evaluations: {evaluate_list}")
    # testframe = pd.DataFrame(total_return_list, columns=["Return"])
    # excel_file = "returns.xlsx"
    # testframe.to_excel(excel_file, index = False)
    ## MEHRERE INPUT VARIABLEN DURCH 1. DOWNLOAD MIT LAG, 2. ALS EINZELNE DFs SETZEN 3. df1.join(df2, lsuffix="", rsuffix="-1")
    ## Grenzen der Datensätze als Variablen

    print("Zeit gebraucht alle Durchlaäufe zusammen:")
    print(f"Zeit für den Durchlauf gebraucht: {np.round((time.time() - _start_time)/3600,2)}h")

if __name__ == "__main__":
    with tf.device('/GPU:0'): main()
    # with tf.device('/CPU:0'): main()