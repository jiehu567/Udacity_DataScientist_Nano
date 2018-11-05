# Perceptron

## Dimension

n-dimensional space $x_1, x_2, ..., x_n$

BOUNDARY:
n-1 dimensional hyperplane $w_1x_1 + w_2x_2 + ... + w_nx_n + b = 0$
$Wx + b = 0$

Prediction:

$$\hat{y}= 1, if Wx + b >= 0; else 0
$$

$$x: n\times1$$
$$W: 1\times n$$
$$b: 1\times 1$$

## Perceptron as Logic Operators

- AND

$$X+y-1.5=0$$

- OR
line posisiont different with AND, but slope is the same

Increase both weights or decrease magnitude of bias will have OR

- NOT

```{python}
# TODO: Set weight1, weight2, and bias
weight1 = 0.0
weight2 = -1.0
bias = 0.5


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
```

pay attention to directions.

- XOR

$$(x_1|x_2) AND (\bar {x_1x_2})$$

NAND(x1, x2) AND OR(x1, x2)

### Algorithm

1. Start with random weights
2. for every misclassified point:
- If prediction = 0;
  - for 1 = 1...n
    - Change $w_i+\alpha x_i$
  - Change b to b + $\alpha$

- If prediction = 1;
  - for 1 = 1...n
    - Change $w_i-\alpha x_i$
  - Change b to b - $\alpha$

```{python}
import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

```