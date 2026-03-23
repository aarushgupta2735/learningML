import numpy as np
#Parameters
Y = np.array([[1]])
X = np.array([[1],[2]])
W1 = np.array([[0.1,0.2],[-0.3,0.4],[0.5,-0.1]])
B1 = np.array([[0.1],[-0.2],[0.3]])
W2 = np.array([[0.3,-0.2,0.5]])
B2 = np.array([0.1])
alpha = 0.1

#Activations

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu_grad(z):
    return (z>0).astype(float)

def relu(z):
    return np.maximum(0,z)

#Forward Pass

Z1 = np.dot(W1,X)+B1
A1 = relu(Z1)
Z2 = np.dot(W2,A1)+B2
A2 = sigmoid(Z2)

Loss = -np.log(A2)
print("y_hat = ",A2)
print("Loss = ",Loss)

#Backward Pass  
dZ2 = A2-Y
dW2 = np.dot(dZ2,A1.T)
dB2 = dZ2

dA1 = np.dot(W2.T,dZ2)
dZ1 = dA1*relu_grad(A1)
dW1 = np.dot(dZ1,X.T)
dB1 = dZ1

print("Gradient wrt W1 = ",dW1)
print("Gradient wrt B1 = ",dB1)
print("Gradient wrt W2 = ",dW2)
print("Gradient wrt B2 = ",dB2)

#Gradient Descent (Single Iteration)
W2 = W2-alpha*dW2
print("New W2 = ",W2)