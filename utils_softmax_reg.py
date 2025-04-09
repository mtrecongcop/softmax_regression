import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def softmax_regression(X, y, iterations, learning_rate, _lambda, verbose=True, seed=1, epsilon=1e-4):
    """
    Update randomized thetas (feature weights) from a standard normal distribution
    associated with the nth feature under the kth class

    Parameters
    ---------
    X : array like with shape (m, n)
        Input features of m number samples with n features
    y: array like with shape (m, 1)
        Input ground truth labels. Must be label encoded i.e. not using 'words' for 
        description
    learning_rate: float
        Step size for gradient updates
    _lambda : float
        Regularization strength
    verbose : bool, default=True
        Show more information like the error stops and the loss plot
    seed : int
        Manually set the seed to randomize initialized array of thetas
    epsilon : float, default=1e-4
        The difference threshold between the cost of the last iteration and the cost
        of the current iteration. For early stopping purposes
    Returns
    -------
    thetas : array like with shape (n, k)
        Optimized weights associated with the nth feature under the kth class for the 
        optimized softmax output
    """
    # Get number of classes (k) and number of parameters (n)
    # Case when y is a DataFrame

    # Initialize randomizer
    np.random.seed(seed)

    # Input checks
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values.ravel() # Ensures it's a flat NumPy array

    # Define training size m and feature number n
    m, n = X.shape
    k = np.unique(y).size
    
    # Initialize the thetas of shape (n, k) NumPy ndarray
    temp_theta = np.random.rand(n, k) # n pertains to the nth feature of the kth class

    # Create the one hot encoded y Y of shape (m, k)
    # m = the number of samples
    # One hot encode using get_dummies
    # case when y is a numpy array
    if isinstance(y, np.ndarray):
        y_s = pd.Series(data=y)
    else:
        y_s = y

    # Convert to NumPy ndarray
    Y = pd.get_dummies(y_s, dtype=float).values # shape (m, k)

    # Get list of cost values per iteration
    cost_list = []

    # Perform SGD
    for iter in range(iterations):
        temp_softmax_output = softmax_output(X, temp_theta)
        gradient = -X.T @ (Y - temp_softmax_output) # shape (n, k)
        gradient += (_lambda / m) * temp_theta
        temp_theta -= learning_rate * gradient
        cost_list.append(softmax_cost(temp_softmax_output, Y, _lambda, temp_theta))
        # Early stopping case based on abs(cost_list[-1] - cost_list[-2]) provided the iter > 1 
        if iter > 1 and abs(cost_list[-1] - cost_list[-2]) < epsilon:
             if verbose:
                print(f"Stopping at iteration: {iter} as delta cost: {abs(cost_list[-1] - cost_list[-2])} < epsilon: {epsilon}")
             break
    
    # Create a plot of cost per iteration
    if verbose:
        np_iterations = np.arange(iterations)
        np_cost_list = np.array(cost_list)
        data = np.column_stack((np_iterations, np_cost_list))
        pd_data = pd.DataFrame(data=data, columns=['Iterations', 'Cost'])
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost trend with increasing iterations')
        sns.lineplot(data=pd_data, x='Iterations', y='Cost')
    
    if verbose:
        return temp_theta, cost_list
    else: 
        return temp_theta

def softmax_output(X, theta):
    """
    Computes the softmax hypothesis

    Parameters
    ----------
    X : array like of shape (m, n)
        Input features (n) of m samples

    Theta:  array like of shape (n, k)
        Parameter estimation for each class k

    Returns
    -------
    Softmax output: array like of shape (m, k).
        The normalized estimate of the softmax hypothesis
        which is the probability of each sample belonging to a specific class K

    """
    # Create the softmax output
    z = X @ theta # shape (m, k)
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    # Normalizer of the distribution
    _sum_exp_z = np.sum(exp_z, axis=1, keepdims=True) # shape (m, 1)
    softmax_output = exp_z / _sum_exp_z # shape (m, k)
    if not np.allclose(np.sum(softmax_output, axis=1), 1, atol=1e-6):
        raise ValueError("Softmax outputs are not properly normalized.")

    return softmax_output

def softmax_cost(softmax_output, Y, _lambda, theta):
    """
    Computes the cross-entropy loss for softmax regression.

    Parameters
    ----------
    softmax_output : array like of shape (m, k). 
        The normalized estimate of the softmax hypothesis
        which is the probability of each sample belonging to a specific class K
    
    Y : array like of shape (m, k). 
        The one hot encoded ground truth label for each sample.
    
    _lambda : float
        The regularization strength for L2 penalty.
    
    theta : array-like of shape (n, k)
        The parameter of softmax regression, where n is the number of input features
        and k is the number of classes.

    Method
    ------

    Computes the negative log-likelihood by:
        1. Perform an element wise multiplication of Y and the log of the softmax output using '*' or the 
        Hadamard product resulting to an array of shape (m, k).
        2. Sum along K where the Hadamard product collapses the columns resulting to a shape of
        (m,)
        3. Sum the resulting (m,) along the rows resulting to a scalar estimate of the cost
        4. Do some ridge regularization
    Returns
    -------
    cost : float 
        The scalar cross-entropy loss for the current batch of predictions
    """
    m = Y.shape[0] # number of samples
    loss_per_sample = np.sum(Y * np.log(softmax_output + 1e-15), axis=1) # shape (m,)
    cost = np.mean(loss_per_sample)
    regularization = (_lambda / (2 * m)) * np.sum(theta**2, axis=None) # Sum all elements in the array
    return cost + regularization

def softmax_predict(X, theta):
    """
    Parameters:
    ----------
    X : array like of shape (m, n)
        Input features
    theta : array like of shape (n, k)
        Optimized theta values for softmax prediction
    Returns:
    --------
    An array of shape (m,) containing the index of the highest probability
    corresponding to the predicted class of the mth sample
    """
    probs = softmax_output(X, theta)
    return np.argmax(probs, axis=1)
