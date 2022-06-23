import numpy as np
from sympy import symbols


class LinearRegression():
    def __init__(self):

        self.__slope = 0
        self.__intercept = 0
        self.__scalingValue = 1
        self.__innerPrediction = True
        self.__linearEquation = self.__intercept


    def __SumofSquaredError(self,x,y):

        prediction = self.predict(x)
        sse = np.sum((prediction - y) ** 2)

        return sse

    
    def accuracy(self,x,y):
        
        prediction = self.predict(x)

        rss = np.sum((y - prediction)**2)
        tss = np.sum((y - y.mean())**2)

        accuracy = 1 - (rss/tss)

        return accuracy


    def __GradientDescent(self,x,y,learning_rate,n_epochs):

        sse_list = [0] * n_epochs

        for epoch in range(n_epochs):

            prediction = self.predict(x)

            loss = prediction - y

            weight_gradient = x.T.dot(loss)/len(y)

            bias_gradient = np.sum(loss)/len(y)


            self.__slope -= learning_rate * weight_gradient

            self.__intercept -= learning_rate * bias_gradient

            sse = self.__SumofSquaredError(x,y)
            sse_list[epoch] = sse

            accuracy = self.accuracy(x,y)


            print(f"Learning model | epoch = {epoch+1}/{n_epochs} | loss = {sse} | accuracy = {accuracy}")


    def fit(self,x,y,learning_rate = 0.1,epochs = 1000):

        self.__scalingValue = max([np.amax(x),np.amax(y)])
   
        x = x / self.__scalingValue
        y = y / self.__scalingValue

        self.__slope = np.zeros(x.shape[1])
        self.__GradientDescent(x,y,learning_rate,epochs)

        self.__innerPrediction = False

        self.__linearEquation = self.__intercept

        for i in range(len(x[0])):
            self.__linearEquation += self.__slope[i] * symbols(f"x{i+1}")

        print(f"\nLinear Equation : y = {self.__linearEquation}\n")


    def predict(self,x):

        if self.__innerPrediction == True:
            prediction = x.dot(self.__slope) + self.__intercept
            return prediction

        else:
            x = x / self.__scalingValue
            prediction = x.dot(self.__slope) + self.__intercept
            return prediction * self.__scalingValue