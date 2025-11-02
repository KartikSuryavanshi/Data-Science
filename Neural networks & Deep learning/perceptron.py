def step(x):
    return 1 if x > 0 else 0 #step --> activation function(to bring non-linearity to our overall fn)

def perceptron(x1,x2,w1,w2,b):
    return step(x1*w1 + x2*w2 + b)

print(perceptron(0,0,1,1,-1.5)) # and gate
print(perceptron(0,1,1,1,-1.5)) # and gate
print(perceptron(1,0,1,1,-1.5)) # and gate
print(perceptron(1,1,1,1,-1.5)) # and gate
