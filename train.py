import numpy as np
#Set up the random seed
def sigmoid(z):
	return np.divide(1,1+np.power(2.718,np.mult(-1,z))) #this approximation of e shouldnt really matter, tbh using 3 would prob work
def siggrad(z):
        return np.multiply(sigmoid(z),1-sigmoid(z))
def error(thetas,X,y,l = 0):
        #these functions for breaking up theta will need to be reworked
        thetas = thetas.split(thetas.size/2)
        theta1 = thetas[0,:]
        theta2 = thetas[1,:]
        m = X.shape[1]
        h1 = sigmoid(np.dot(X,theta1.transpose()));
        h2 = sigmoid(np.dot(h1,theta2.transpose()));
        #will stick regularization on there as a comment
        J = (1/m)*np.sum(np.multiply(-y,log(h2))-np.multiply(1-y,log(1-h2))) #+ (l/(2*m)) * (sum(np.power(theta1,2)) + sum(np.power(theta2,2)))
        theta1grad = 0
        theta2grad = 0
        for t in range(m):
                a1 = X[t,:]
                z2 = np.dot(theta1,a1.transpose())
                a2 = sigmoid(z2)
                z3 = np.dot(Theta2,a2)
                a3 = sigmoid(z3)
                delta3 = a3-y[:,t]
                delta2 = np.multiply(np.transpose(theta2),delta3)
                theta1grad = (theta1grad + dot(delta2,a1))/m # + (l/m) * theta1
                theta2grad = (theta2grad + dot(delta3,a1))/m # + (l/m) * theta2
        return (J,np.append(theta1grad,theta2grad))
insize = 10 #size of input vectors
hlsize = 10 #size of hidden layer
#random initialization of theta
thetas = np.random.rand(2*hlsize, insize) #this matrix should be (2hlsize x insize)
X = np.random.rand(insize,10)#X should be the data when all is said and done
