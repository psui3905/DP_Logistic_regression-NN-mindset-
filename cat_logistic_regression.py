import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import scipy
# from PIL import Image
# from scipy import ndimage
# from lr_utils import load_dataset

# Overview of the Problem set
# Given a dataset (data.h5) containing:
#     - a traninging set of m_train images labeled as cat (y = 1) or non-cat (y = 0)
#     - a test set of m_test image labeled as cat or non-def funcname(self, parameter_list)
#     - Each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
# Thus, each image is square (height = num_px) and (width = num_px)

# You will build a simple image-recognition algorithm that can correctly classify pictures
# Let's get more familiar with the dataset. Load the data by running the following code.


def load_dataset():
    """
    Convert the data in a training set (data.h5) in to a set of matrix
    with training/text x and y data.

    Return:

    train_set_x_orig -- a numpy array of shape (m_train, num_pxm num_px,3)
        - m_train (number of training examples)
        - m_test (number of test examples)
        - num_px (= height = width of a training image)

    train_set_y shape -- a numpy array with shape (1, m_train)

    test_set_x_orig -- a numpy array of shape (m_text, num_pxm num_px,3)
        - m_text (number of text exmples)

    test_set_y --  a numpy array of shape (1, m_text)
    """

    train_set_x_orig = 0;
    train_set_y = 0;
    test_set_x_orig = 0;
    test_set_y = 0;
    classes = 0;

    return (train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)

# Helper functions:
#     -- implment Sigmoid function

def sigmoid(z):
    """
    Compute the sigmoid of z 

    Arguments:
    z -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+ np.exp(-z));
    return s;


# Initializing parameters:
# Initial w as a vector of zeros. 

def initialize_with_zeros(dim):

    """
    This function creates a vector of zeros of shape (dim, 1) for 
    w and initializs b to 0;

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) 
    """
    
    w = np.zeros([dim, 1])
    b = 0;

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int));
    
    return w,b;

def propagate(w, b, X, Y):

    """
    Implement the cost function and its gradient for the propagation explained above 

    Arguments:
    w -- weights, a numpy array of size (num_px * nump_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, i if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelohood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    """

    m = X.shape[1];

    # A.shape should be (1, m), m -- number of training examples
    A = sigmoid(np.dot(w.T, X) + b);
    # single training exmaple cost = - (y * log(A) + (1-y) * log(1-A))
    cost = (-1/m) * (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T));

    # Backward propagation 
    dw = (1/m) * np.dot(X, (A - Y).T)
    # np.sum() : axis = 0 means along the column and axis = 1 means working along the row.
    db = (1/m) * ((A - Y).sum(1))

    # dw.shape should be (num_px * nump_px * 3, 1), db should be a scalar
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (Gradient Descent)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the change of costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs





