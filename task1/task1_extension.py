'''
Second: Add a loop that runs 1000 gradient descent steps and prints the 
loss every 100 iterations. 
You should see the loss decrease. 
If it doesn't, something is wrong with your gradients. 
This is your first experience of "watching a network learn."
Third: After the loop, print ŷ. It should be close to 1.0 since y=1. 
Verify this.
'''


import numpy as np
#Parameters
Y = np.array([[1]])
X = np.array([[1],[2]])
W1 = np.array([[0.1,0.2],[-0.3,0.4],[0.5,-0.1]])
B1 = np.array([[0.1],[-0.2],[0.3]])
W2 = np.array([[0.3,-0.2,0.5]])
B2 = np.array([0.1])
alpha = 0.1
n = 1000
t = 100
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

print("Initial Values")
print("Gradient wrt W1 = ",dW1)
print("Gradient wrt B1 = ",dB1)
print("Gradient wrt W2 = ",dW2)
print("Gradient wrt B2 = ",dB2)

#Gradient Descent (Single Iteration)
def gradient_descent():
    global W2,W1,B1,B2
    W2= W2-alpha*dW2
    W1= W1-alpha*dW1
    B1= B1-alpha*dB1
    B2= B2-alpha*dB2

#1000 gradient descent steps and printing after every 100 iterations
def forward_pass():
    global W2,Z1,A1,Z2,A2,X,W1,B1,B2, Loss
    Z1 = np.dot(W1,X)+B1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1)+B2
    A2 = sigmoid(Z2)
    Loss = -np.log(A2)

def backward_pass():
    global dZ2,dW2,dB2,dA1,dZ1,dW1,dB1
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)
    dB2 = dZ2

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = dA1*relu_grad(A1)
    dW1 = np.dot(dZ1,X.T)
    dB1 = dZ1

for i in range(1,n+1):
    gradient_descent() #UPDATE W2 USING dW2
    forward_pass() #FIND Y_HAT AND LOSS
    backward_pass() #UPDATE dW2
    if(i%t==0):
        print(f"Iteration: {i}, Updated_W2 = {W2}, Updated y_hat = {A2}, Loss = {Loss}")

print("-----FINAL RESULTS AFTER 1000 ITERATIONS-----")
print(f"Final W2 = {W2}")
print(f"Final y_hat = {A2} ")
