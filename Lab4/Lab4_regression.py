'''
# **Lab 4 : Neural Network**

In *lab 4*, you need to finish:

1. Basic Part (65%):
  Implement a deep neural network from scratch

  > * Section 1: Neural network implementation
    >> * Part 1: Linear layer
    >> * Part 2: Activation function layer
    >> * Part 3: Build model

  > * Section 2: Loss function
    >> * Part 1: Binary cross-entropy loss (BCE)
    >> * Part 2: Categorical cross-entropy loss (CCE)
    >> * Part 3: Mean square error (MSE)
  > * Section 3: Training and prediction
    >> * Part 1: Training function & batch function
    >> * Part 2: Regression
    >> * Part 3: Binary classification


2. Advanced Part (30%): Multi class classification
3. Report (5%)

'''

'''
## **Important  notice**

* Please **do not** change the code outside this code bracket in the basic part.
  ```
  ### START CODE HERE ###
  ...
  ### END CODE HERE ###
  ```

* Please **do not** import any other packages in both basic and advanced part

* Please **do not** change the random seed **np.random.seed(1)**.

'''

'''
## Import Packages

'''

import numpy as np
# import cupy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import f1_score
from matplotlib.animation import FuncAnimation

outputs = {}

'''
### Common Notation
  * $C$: number of classes
  * $n$: number of samples
  * $f^{[l]}$: the dimension of outputs in layer $l$, but $f^{[0]}$ is the input dimension
  * $Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}$
      * $Z^{[l]}$: the output of layer $l$ in the shape $(n, f^{[l]})$
      * $A^{[l]}$: the activation of $Z^{[l]}$ in the shape $(n, f^{[l]})$, but $A^{[0]}$ is input $X$
      * $W^{[l]}$: the weight in layer $l$ in the shape $(f^{[l-1]}, f^{[l]})$
      * $b^{[l]}$: the bias in layer $l$ in the shape $(1, f^{[l]})$

'''

'''
# **Basic Part (65%)**
In the Basic Part, you will implement a neural network framework capable of handling both regression, binary classification and multi-class classification tasks.

**Note:**
After implementing each class/function, test it with the provided input variables to verify its correctness. Save the results in the **outputs** dictionary. (The code for testing and saving results is already provided.)
## Section 1: Neural network implementation
* Part 1: Linear layer
> * Step 1: Linear Initialize parameters
> * Step 2: Linear forward
> * Step 3: Linear backward
> * Step 4: Linear update parameters
* Part 2: Activation function layer
> * Step 1: Activation forward
> * Step 2: Activation backward
* Part 3: Build model
> * Step 1: Model Initialize parameters
> * Step 2: Model forward
> * Step 3: Model backward
> * Step 4: Model update parameters

## Section 2: Loss function
* Part 1: Binary cross-entropy loss (BCE)
* Part 2: Categorical cross-entropy loss (CCE)
* Part 3: Mean square error (MSE)

## Section 3: Training and prediction
* Part 1: Training function & batch function
* Part 2: Regression
* Part 3: Binary classification

'''

'''
## **Section 1: Neural network implementation(30%)**
To implement a neural network, you need to complete 3 classes: **Dense**, **Activation**, and **Model**.
The process of training a deep neural network is composed of 3 steps: *forward propagation*, *backward propagation*, and *update*.
'''

'''
## Part 1: Linear layer (10%)
Dense layer (fully-connected layer) performs linear transformation:

$Z = AW + b$, where W is weight matrix and b is bias vector.

> ### Step 1: Initialize parameters (0%)
 * You don't need to write this part.
 * W is randomly initialized using uniform distribution within $[\text\{-limit\}, \text\{limit\}]$, where $\text\{limit\} = \sqrt{\frac{6}{\text\{fanin\} + \text\{fanout\}}}$ (fanin: number of input features, fanout: number of output features)
 * b is initialized to 0

> ### Step 2: Linear forward (4%)
* Compute Z using matrix multiplication and addition

> ### Step 3: Linear backward (4%)
* Use backpropagation to compute gradients of loss function with respect to parameters
* For layer l: $Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$ (followed by activation)
* Given $dZ^{[l]}$ (gradient of loss with respect to Z), we need to compute three gradients:
  * $dW^{[l]}$: gradient of loss with respect to weights
  * $db^{[l]}$: gradient of loss with respect to bias
  * $dA^{[l-1]}$: gradient of loss with respect to previous layer output

> Formulas:
$$ dW^{[l]} = \frac{1}{n} A^{[l-1] T} dZ^{[l]} $$
$$ db^{[l]} = \frac{1}{n} \sum_{i = 1}^{n} dZ_i^{[l]} $$
$$ dA^{[l-1]} = dZ^{[l]} W^{[l] T} $$

> ### Step 4: Linear update parameters (2%)
* Update parameters using gradient descent:
$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} $$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} $$
'''

class Dense():
    def __init__(self, n_x, n_y, seed=1):
        self.n_x = n_x
        self.n_y = n_y
        self.seed = seed
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Argument:
        self.n_x -- size of the input layer
        self.n_y -- size of the output layer
        self.parameters -- python dictionary containing your parameters:
                           W -- weight matrix of shape (n_x, n_y)
                           b -- bias vector of shape (1, n_y)
        """
        sd = np.sqrt(6.0 / (self.n_x + self.n_y))
        np.random.seed(self.seed)
        W = np.random.uniform(-sd, sd, (self.n_y, self.n_x)).T      # the transpose here is just for the code to be compatible with the old codes
        b = np.zeros((1, self.n_y))

        assert(W.shape == (self.n_x, self.n_y))
        assert(b.shape == (1, self.n_y))

        self.parameters = {"W": W, "b": b}

    def forward(self, A):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data) with the shape (n, f^[l-1])
        self.cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter with the shape (n, f^[l])
        """

        # GRADED FUNCTION: linear_forward
        ### START CODE HERE ###
        Z = A @ self.parameters["W"] + self.parameters["b"]
        self.cache = (A, self.parameters["W"], self.parameters["b"])
        ### END CODE HERE ###

        assert(Z.shape == (A.shape[0], self.parameters["W"].shape[1]))

        return Z

    def backward(self, dZ):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the loss with respect to the linear output (of current layer l), same shape as Z
        self.cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        self.dW -- Gradient of the loss with respect to W (current layer l), same shape as W
        self.db -- Gradient of the loss with respect to b (current layer l), same shape as b

        Returns:
        dA_prev -- Gradient of the loss with respect to the activation (of the previous layer l-1), same shape as A_prev

        """
        A_prev, W, b = self.cache
        m = A_prev.shape[0]

        # GRADED FUNCTION: linear_backward
        ### START CODE HERE ###
        self.dW = A_prev.T @ dZ / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ W.T
        ### END CODE HERE ###

        assert (dA_prev.shape == A_prev.shape)
        assert (self.dW.shape == self.parameters["W"].shape)
        assert (self.db.shape == self.parameters["b"].shape)

        return dA_prev

    def update(self, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        learning rate -- step size
        """

        # GRADED FUNCTION: linear_update_parameters
        ### START CODE HERE ###
        self.parameters["W"] = self.parameters["W"] - learning_rate * self.dW
        self.parameters["b"] = self.parameters["b"] - learning_rate * self.db
        ### END CODE HERE ###

'''
## Part 2: Activation function layer (10%)

Implement forward and backward propagation for activation function layers, including Sigmoid, Softmax, and ReLU.

> ### Step 1: Forward Propagation (5%)
 Implement the following activation functions:
>> #### a) Sigmoid
- Use the numerically stable version to prevent exponential overflow:
  $$\sigma(Z) = \begin{cases}
    \frac{1}{1+e^{-Z}},& \text{if } Z \geq 0\\
    \frac{e^{Z}}{1+e^{Z}}, & \text{otherwise}
  \end{cases}$$

>> #### b) ReLU
- Simple implementation:
  $$RELU(Z) = \max(Z, 0)$$

>> #### c) Softmax
- Implement using the numerically stable version:
  $$\sigma(\vec{Z})_i = \frac{e^{Z_i-b}}{\sum_{j=1}^{C} e^{Z_j-b}}$$
  where $b = \max_{j=1}^{C} Z_j$

>> #### d) Linear
- You don't need to implement this part

> ### Requirements
- Each function should return:
  1. Activation value "a"
  2. Cache containing "z" for backward propagation

> ### Step 2: Backward Propagation (5%)
Implement backward functions for:
- Sigmoid
- ReLU
- Softmax
- linear

> ### General Form
$$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]})$$
where $g(.)$ is the activation function

> ### Specific Implementations

>> #### a) Sigmoid Backward
$$\sigma'(Z^{[l]}) = \sigma(Z^{[l]}) (1 - \sigma(Z^{[l]}))$$
Use numerically stable sigmoid

>> #### b) ReLU Backward
$$g'(Z^{[l]}) = \begin{cases}
    1,& \text{if } Z^{[l]} > 0\\
    0,              & \text{otherwise}
\end{cases}$$

>> #### c) Softmax Backward
For the special case of Softmax combined with Categorical Cross-Entropy loss:
$$dZ^{[l]} = s - y$$
where $s$ is softmax output, $y$ is true label (one-hot vector)

Note: This is a simplified form specific to Softmax + CCE loss combination.

>> #### d) linear Backward
You don't need to implement this part

> ### Note
For softmax, use the normalized exponential function to prevent overflow, but use the simplified gradient equation for backwards propagation.
'''

class Activation():
    def __init__(self, activation_function, loss_function):
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.cache = None

    def forward(self, Z):
        if self.activation_function == "sigmoid":
            """
            Implements the sigmoid activation in numpy

            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation

            Returns:
            A -- output of sigmoid(z), same shape as Z
            """

            # GRADED FUNCTION: sigmoid_forward
            ### START CODE HERE ###
            A = np.where(Z >= 0, 1 / (1 + np.exp(-Z)), np.exp(Z) / (1 + np.exp(Z)))
            self.cache = Z
            ### END CODE HERE ###

            return A
        elif self.activation_function == "relu":
            """
            Implement the RELU function in numpy
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            Returns:
            A -- output of relu(z), same shape as Z

            """

            # GRADED FUNCTION: relu_forward
            ### START CODE HERE ###
            A = np.where(Z >= 0, Z, 0)
            self.cache = Z
            ### END CODE HERE ###

            assert(A.shape == Z.shape)

            return A
        elif self.activation_function == "softmax":
            """
            Implements the softmax activation in numpy

            Arguments:
            Z -- np.array with shape (n, C)
            self.cache -- stores Z as well, useful during backpropagation

            Returns:
            A -- output of softmax(z), same shape as Z
            """

            # GRADED FUNCTION: softmax_forward
            ### START CODE HERE ###
            b = np.max(Z, axis=1, keepdims=True)
            A = np.exp(Z - b) / np.sum(np.exp(Z - b), axis=1, keepdims=True)
            self.cache = Z
            ### END CODE HERE ###

            return A
        elif self.activation_function == "linear":
            """
            Linear activation (returns Z directly).
            """
            self.cache = Z.copy()
            return Z

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")


    def backward(self, dA=None, Y=None):
        if self.activation_function == "sigmoid":
            """
            Implement the backward propagation for a single SIGMOID unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the loss with respect to Z
            """

            # GRADED FUNCTION: sigmoid_backward
            ### START CODE HERE ###
            Z = np.where(self.cache >= 0, 1 / (1 + np.exp(-self.cache)), np.exp(self.cache) / (1 + np.exp(self.cache)))
            dZ = dA * (Z * (1 - Z))
            ### END CODE HERE ###

            assert (dZ.shape == Z.shape)

            return dZ

        elif self.activation_function == "relu":
            """
            Implement the backward propagation for a single RELU unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the loss with respect to Z
            """

            # GRADED FUNCTION: relu_backward
            ### START CODE HERE ###
            Z = self.cache
            dZ = dA * np.where(Z > 0, 1, 0)
            ### END CODE HERE ###

            assert (dZ.shape == Z.shape)

            return dZ

        elif self.activation_function == "softmax":
            """
            Implement the backward propagation for a [SOFTMAX->CCE LOSS] unit.
            Arguments:
            Y -- true "label" vector (one hot vector, for example: [1,0,0] represents rock, [0,1,0] represents paper, [0,0,1] represents scissors
                                      in a Rock-Paper-Scissors, shape: (n, C)
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """

            # GRADED FUNCTION: softmax_backward
            ### START CODE HERE ###
            Z = self.cache
            s = self.forward(Z)
            dZ = s - Y
            ### END CODE HERE ###

            assert (dZ.shape == self.cache.shape)

            return dZ

        elif self.activation_function == "linear":
            """
            Backward propagation for linear activation.
            """
            return dA

        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")

'''
## Part 3: Model (10%)

Use the functions that you had previously written to implement the complete neural network model, including initialization, forward propagation, backward propagation, and parameter updates.

> ### Step 1: Model Initialization (0%)
Initialize the model by creating linear and activation function layers.

>> #### Requirements:
- Store linear layers in a list called `linear`
- Store activation function layers in a list called `activation`
- Use iteration number as seed for each Dense layer initialization

>> #### Note:
A linear-activation pair counts as a single layer in the neural network.

> ### Step 2: Forward Propagation (4%)
Implement the model's forward pass by calling each layer's forward function sequentially.

>> #### Process:
1. For layers 1 to N-1: [LINEAR -> ACTIVATION]
2. Final layer: LINEAR -> SIGMOID (binary) or SOFTMAX (multi-class)

>> #### Note:
For binary classification, use one output node with sigmoid activation. For K-class classification, use K output nodes with softmax activation.

> ### Step 3: Backward Propagation (4%)
Implement the model's backward pass by calling each layer's backward function in reverse order.

>> #### Process:
1. Initialize backpropagation:
   - Regression:
     $$dAL = AL - Y$$
   - Binary classification:
     $$dAL = - (\frac{Y}{AL + \epsilon} - \frac{1 - Y}{1 - AL + \epsilon})$$
     where $\epsilon = 10^{-5}$ to prevent division by zero
   - Multi-class classification:
     Use `softmax_backward` function
2. Backpropagate through layers L to 1

>> #### Note:
Use cached values from the forward pass in each layer's backward function.

> ### Step 4: Parameter Update (2%)
Update model parameters using gradient descent.

>> #### Update Rule:
For each layer $l = 1, 2, ..., L$:
$$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$
where $\alpha$ is the learning rate

This revised structure provides a clear, step-by-step breakdown of the model implementation process, mirroring the format used in Part 2. It covers all the essential components while maintaining a concise and logical flow.
'''

class Model():
    def __init__(self, units, activation_functions, loss_function):
        self.units = units
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize layers of the neural network

        Arguments:
            self.units -- array defining network structure (e.g., [4,4,1]):
                - Input layer: 4 nodes
                - Hidden layer: 4 nodes
                - Output layer: 1 node
            self.activation_functions -- activation function for each layer (e.g., ["relu","sigmoid"]):
                - First layer uses ReLU
                - Second layer uses Sigmoid
            self.loss_function -- loss function type: "cross_entropy" or "mse"
        """
        self.linear = []        # Store all Dense layers (weights & biases)
        self.activation = []    # Store all activation function layers

        for i in range(len(self.units)-1):
            dense = Dense(self.units[i], self.units[i+1], i)
            self.linear.append(dense)

        for i in range(len(self.activation_functions)):
            self.activation.append(Activation(self.activation_functions[i], self.loss_function))

    def forward(self, X):
        """
        Forward propagation through the network

        Arguments:
        X -- input data: shape (n, f)
        Returns:
        A -- model output:
            - For binary classification: probability (0-1)
            - For multi-class: probability distribution across classes
            - For regression: predicted values
        """
        A = X

        # GRADED FUNCTION: model_forward
        ### START CODE HERE ###
        for i in range(len(self.linear)):
            Z = self.linear[i].forward(A)
            A = self.activation[i].forward(Z)
        ### END CODE HERE ###

        return A

    def backward(self, AL=None, Y=None):
        """
        Backward propagation to compute gradients

        Arguments:
            AL -- model output from forward propagation:
                - For binary: probability (n,1)
                - For multi-class: probabilities (n,C)
            Y -- true labels:
                - For binary: 0/1 labels (n,1)
                - For multi-class: one-hot vectors (n,C)
                - For regression: true values (n,1)

        Returns:
            dA_prev -- gradients for previous layer's activation
        """

        L = len(self.linear)
        C = Y.shape[1]

        # assertions
        warning = 'Warning: only the following 3 combinations are allowed! \n \
                    1. binary classification: sigmoid + cross_entropy \n \
                    2. multi-class classification: softmax + cross_entropy \n \
                    3. regression: linear + mse'
        assert self.loss_function in ["cross_entropy", "mse"], "you're using undefined loss function!"
        if self.loss_function == "cross_entropy":
            if Y.shape[1] == 1:  # binary classification
                assert self.activation_functions[-1] == 'sigmoid', warning
            else:  # multi-class classification
                assert self.activation_functions[-1] == 'softmax', warning
                assert self.units[-1] == Y.shape[1], f"you should set last dim to {Y.shape[1]}(the number of classes) in multi-class classification!"
        elif self.loss_function == "mse":
            assert self.activation_functions[-1] == 'linear', warning
            assert self.units[-1] == Y.shape[1], "output dimension mismatch for regression!"

        # GRADED FUNCTION: model_backward
        ### START CODE HERE ###
        if self.activation_functions[-1] == "linear":
            # Initializing the backpropagation
            dAL = AL - Y
            # Lth layer (LINEAR) gradients. Inputs: "dAL". Outputs: "dA_prev"
            dZ = self.activation[-1].backward(dA=dAL)
            dA_prev = self.linear[-1].backward(dZ)

        elif self.activation_functions[-1] == "sigmoid":
            # Initializing the backpropagation
            dAL = -(Y / (AL + 1e-5) - (1 - Y) / (1 - AL + 1e-5))

            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL". Outputs: "dA_prev"
            dZ = self.activation[-1].backward(dA=dAL)
            dA_prev = self.linear[-1].backward(dZ)

        elif self.activation_functions[-1] == "softmax":
            # Initializing the backpropagation
            dZ = self.activation[-1].backward(Y = Y)

            # Lth layer (LINEAR) gradients. Inputs: "dZ". Outputs: "dA_prev"
            dA_prev = self.linear[-1].backward(dZ)

        # Loop from l=L-2 to l=0
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "dA_prev". Outputs: "dA_prev"
        for l in range(L-2, -1, -1):
            dZ = self.activation[l].backward(dA=dA_prev, Y=Y)
            dA_prev = self.linear[l].backward(dZ)
        ### END CODE HERE ###

        return dA_prev

    def update(self, learning_rate):
        """
        Arguments:
        learning_rate -- step size
        """

        L = len(self.linear)

        # GRADED FUNCTION: model_update_parameters
        ### START CODE HERE ###
        for i in range(L):
            self.linear[i].update(learning_rate)
        ### END CODE HERE ###

'''
## **Section 2: Loss function(10%)**
In this section, you need to implement the loss function. We use binary cross-entropy loss for binary classification and categorical cross-entropy loss for multi-class classification.

'''

'''
## Part 1: Binary cross-entropy loss (BCE) (5%)
Compute the binary cross-entropy loss $L$, using the following formula:  $$-\frac{1}{n} \sum\limits_{i = 1}^{n} (y^{(i)}\log\left(a^{[L] (i)}+ϵ\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}+ϵ\right)), where\ ϵ=1e-5$$
'''

# GRADED FUNCTION: compute_BCE_loss

def compute_BCE_loss(AL, Y):
    """
    Implement the binary cross-entropy loss function using the above formula.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (n, 1)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (n, 1)

    Returns:
    loss -- binary cross-entropy loss
    """

    n = Y.shape[0]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 line of code)
    loss = -(np.sum(Y * np.log(AL + 1e-5) + (1 - Y) * np.log(1 - AL + 1e-5))) / n
    ### END CODE HERE ###

    loss = np.squeeze(loss)      # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())

    return loss

'''
## Part 2: Categorical cross-entropy loss (CCE) (5%)
Compute the categorical cross-entropy loss $L$, using the following formula: $$-\frac{1}{n} \sum\limits_{i = 1}^{n} (y^{(i)}\log\left(a^{[L] (i)}+ϵ\right)),\ ϵ = 1e-5$$






'''

# GRADED FUNCTION: compute_CCE_loss

def compute_CCE_loss(AL, Y):
    """
    Implement the categorical cross-entropy loss function using the above formula.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (n, C)
    Y -- true "label" vector (one hot vector, for example: [1,0,0] represents rock, [0,1,0] represents paper, [0,0,1] represents scissors
                                      in a Rock-Paper-Scissors, shape: (n, C)

    Returns:
    loss -- categorical cross-entropy loss
    """

    n = Y.shape[0]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 line of code)
    loss = -(np.sum(Y * np.log(AL + 1e-5))) / n
    ### END CODE HERE ###

    loss = np.squeeze(loss)      # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())

    return loss

'''
## Part 3: Mean square error (MSE) (0%)
You don't need to write this part.
'''

# compute_MSE_loss (MSE)
def compute_MSE_loss(AL, Y):
    m = Y.shape[0]
    loss = (1/m) * np.sum(np.square(AL - Y))
    return loss

'''
## **Section 3: Training and prediction(35%)**
In this section, you will apply your implemented neural network to regression and binary classification tasks.
'''

'''
## Helper function

'''

def predict(x, y_true, model):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    x -- data set of examples you would like to label
    model -- trained model

    Returns:
    y_pred -- predictions for the given dataset X
    """

    n = x.shape[0]

    # Forward propagation
    y_pred = model.forward(x)

    # this transform the output and label of binary classification when using sigmoid + cross entropy for evaluation
    # eg. y_pred: [[0.8], [0.2], [0.1]] -> [[0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]
    # eg. y_true: [[1], [0], [0]] -> [[0, 1], [1, 0], [1, 0]]
    if y_pred.shape[-1] == 1:
        y_pred = np.array([[1 - y[0], y[0]] for y in y_pred])
        if y_true is not None:
            y_true = np.array([[1,0] if y == 0 else [0,1] for y in y_true.reshape(-1)])

    # make y_pred/y_true become one-hot prediction result
    # eg. y_true: [[1, 0, 0], [0, 0, 1], [0, 1, 0]] -> [0, 2, 1]
    # eg. y_pred: [[0.2, 0.41, 0.39], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] -> [1, 1, 2]
    if y_true is not None:
        y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    if y_true is not None:
        # compute accuracy
        correct = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == yp:
                correct += 1
        print(f"Accuracy: {correct/n * 100:.2f}%")

        f1_scores = f1_score(y_true, y_pred, average=None)
        print(f'f1 score for each class: {f1_scores}')
        print(f'f1_macro score: {np.mean(np.array(f1_scores)):.2f}')

    return y_pred

def save_prediction_data(predicted_y):
    # Create DataFrame with ID, x, and y columns
    df = pd.DataFrame({
        'ID': range(len(predicted_y)),  # Add ID column starting from 0
        'y': predicted_y
    })

    # Ensure ID is the first column
    df = df[['ID', 'y']]

    # Save to CSV file
    df.to_csv('Lab4_basic_regression.csv', index=False)
    print("Prediction data saved as 'Lab4_basic_regression.csv'")

def animate_training(history, X_train, Y_train):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 11)
    ax.set_ylim(-5, 5)
    line, = ax.plot([], [], 'b-', lw=1, label='Predicted')

    ground_truth_x = X_train.flatten()
    ground_truth_y = Y_train.flatten()
    ax.plot(ground_truth_x, ground_truth_y, 'r-', lw=1, label='Ground Truth')

    # show current epoch on the animation / 100 epoch
    epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def init():
        line.set_data([], [])
        epoch_text.set_text('')
        return line, epoch_text

    def update(frame):
        epoch = (frame + 1) * 100
        _, predicted_y = history[frame]
        predicted_x = X_train.flatten()
        line.set_data(predicted_x, predicted_y.flatten())

        epoch_text.set_text(f'Epoch: {epoch}')

        return line, epoch_text

    ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, interval=50)

    # save as gif
    ani.save('Lab4_basic_regression.gif', writer='pillow')
    plt.close(fig)
    print(f"Animation saved as 'Lab4_basic_regression.gif'")


def save_final_result(model, X_train, Y_train):
    AL = model.forward(X_train)

    predicted_x = X_train.flatten()
    predicted_y = AL.flatten()

    plt.plot(predicted_x, predicted_y, 'b-', label="Predicted", lw=1)

    ground_truth_x = X_train.flatten()
    ground_truth_y = Y_train.flatten()

    save_prediction_data(predicted_y)

    plt.plot(ground_truth_x, ground_truth_y, 'r-', label='Ground Truth', lw=1)

    plt.legend()

    plt.ylim(-5, 5)
    plt.xlim(0, 11)
    plt.savefig("Lab4_basic_regression.jpg")
    plt.show()
    print("Prediction saved as 'Lab4_basic_regression.jpg'")



'''
## Part1: Training function & batch function (5%)
The functions defined in this part will be utilized in the subsequent training parts.
'''

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (n, f^{0})
    Y -- true "label" vector, of shape (n, C)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    ### START CODE HERE ###

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[-(m % mini_batch_size):]
        mini_batch_Y = shuffled_Y[-(m % mini_batch_size):]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    ### END CODE HERE ###

    return mini_batches

def train_model(model, X_train, Y_train, learning_rate, num_iterations, batch_size=None, print_loss=True, print_freq=1000, decrease_freq=100, decrease_proportion=0.99):
    """
    Trains the model using mini-batch gradient descent

    Arguments:
    model -- the model to be trained
    X_train -- training set, of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels, of shape (1, m_train)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    batch_size -- size of a mini batch
    print_loss -- if True, print the loss every print_freq iterations
    print_freq -- print frequency
    decrease_freq -- learning rate decrease frequency
    decrease_proportion -- learning rate decrease proportion

    Returns:
    model -- the trained model
    losses -- list of losses computed during the optimization
    history -- list of (X_train, Y_pred) tuples for visualization
    """

    history = []
    losses = []

    for i in range(num_iterations):
        ### START CODE HERE ###
        # Define mini batches
        if batch_size:
            mini_batches = random_mini_batches(X_train, Y_train, batch_size)
        else:
            # if batch_size is None, batch is not used, mini_batch = whole dataset
            mini_batches = random_mini_batches(X_train, Y_train, X_train.shape[0])

        epoch_loss = 0
        for batch in mini_batches:
            X_batch, Y_batch = batch

            # Forward pass
            AL = model.forward(X_batch)

            # Compute loss
            loss = 0
            if model.loss_function == 'cross_entropy':
                if model.activation_functions[-1] == "sigmoid": # Binary classification
                    loss = compute_CCE_loss(AL, Y_batch)
                elif model.activation_functions[-1] == "softmax": # Multi-class classification
                    loss = compute_CCE_loss(AL, Y_batch)
            elif model.loss_function == 'mse': # Regression
                loss = compute_MSE_loss(AL, Y_batch)
            epoch_loss += loss

            # Backward pass
            model.backward(AL, Y_batch)

            # Update parameters
            model.update(learning_rate)

        epoch_loss /= len(mini_batches)
        losses.append(epoch_loss)
        ### END CODE HERE ###

        # Print loss
        if print_loss and i % print_freq == 0:
            print(f"Loss after iteration {i}: {epoch_loss}")

        # Store history
        if i % 100 == 0:
            history.append((X_train, model.forward(X_train)))

        # Decrease learning rate
        if i % decrease_freq == 0 and i > 0:
            learning_rate *= decrease_proportion

    return model, losses, history


'''
## Part 2: Regression (10%)
In this part, Your task is to train a neural network model to approximate the following mathematical function:

$$y = sin(2 * sin(2 * sin(2 * sin(x))))$$
'''

'''
> ### Step 1: Data generation
Generate the mathematical function :  $$y = sin(2 * sin(2 * sin(2 * sin(x))))$$
'''

def generate_data(num_points=1000):

    x = np.linspace(0.01, 11, num_points)
    y = np.sin(2 * np.sin(2 * np.sin(2 * np.sin(x))))

    return x.reshape(-1, 1), y.reshape(-1, 1)

'''
> ### Step 2: Train model
Implement and train your model using the generated data.
'''

### START CODE HERE ###
x_train, y_train = generate_data()
loss_function = "mse"
layers_dims = [1, 32, 64, 64, 32, 1]
activation_fn = ['linear', 'relu', 'relu', 'relu', 'linear']
learning_rate = 0.01
num_iterations = 100000
print_loss = True
print_freq = 1000
decrease_freq = 5000
decrease_proportion = 0.9
# You don't necessarily need to use mini_batch in this part
batch_size = None

model = Model(layers_dims, activation_fn, loss_function)
model, losses, history = train_model(model, x_train, y_train, learning_rate, num_iterations, batch_size, print_loss, print_freq, decrease_freq, decrease_proportion)
### END CODE HERE ###

# Plot the loss
plt.figure(figsize=(5, 3))
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Training Loss (Initial LR: {learning_rate})')
plt.show()

'''
> ### Step 3: Save prediction
Save your model's predictions to:
> * *Lab4_basic_regression.csv*
> * *Lab4_basic_regression.jpg*
> * *Lab4_basic_regression.gif*
'''

save_final_result(model, x_train, y_train)
animate_training(history, x_train, y_train)