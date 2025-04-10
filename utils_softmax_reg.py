import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling functions

# Note: y or ground truth labels MUST be in numerical 0,1,2 form. Our functions
# automatically one-hot encodes this. 

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


def softmax_cost(softmax_output, Y, _lambda, theta, regularize=True):
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

    regularize : bool, default=True
        Allow option if L2 regularization is desired.

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
    cost = -np.mean(loss_per_sample)
    regularization = (_lambda / (2 * m)) * np.sum(theta**2, axis=None) # Sum all elements in the array
    if regularize:
        return cost + regularization
    else:
        return cost
    
def softmax_regression(X, y, iterations, learning_rate, _lambda, verbose=True, return_final_cost=False, seed=1, early_stopping=True, epsilon=1e-4):
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
    iterations: int
        Set the number of iterations
    learning_rate: float
        Step size for gradient updates
    _lambda : float
        Regularization strength
    verbose : bool, default=True
        Show more information like the error stops and the loss plot
    seed : int
        Manually set the seed to randomize initialized array of thetas
    early_stopping : bool, default=True
        Stop iteration once convergence is reached based on the difference of the current
        cost and the last cost being less than epsilon
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
        # Note: We always regularize when fitting!
        cost_list.append(softmax_cost(temp_softmax_output, Y, _lambda, temp_theta, regularize=True))
        # Early stopping case based on abs(cost_list[-1] - cost_list[-2]) provided the iter > 1
        if early_stopping: 
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
    
    if return_final_cost:
        return temp_theta, cost_list[-1]
    else: 
        return temp_theta

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
    # Ensure X is numpy
    X_np = X.values
    probs = softmax_output(X_np, theta)
    return np.argmax(probs, axis=1)

# Model selection functions 

def train_test_split(X, y, train_size, random_state, shuffle=True):
    """
    Splits the data into training and testing parts
    
    Parameters
    ----------
    X : array like of shape (m, n)
        Input features
    y : array like of shape (m, 1)
        Input labels
    train_size : float
        Percentage of the dataset to be used as training
    random_state : int
        To fix the randomizer seed for reproducibility
    shuffle : bool, default=True
        Shuffle the data
    Returns 
    -------
    X_train : array like of shape (m, n)
        Training set features
    X_test : array like of shape (m, n)
        Testing set features
    y_train : array like of shape (m, 1)
        Training set labels
    y_test : array like of shape (m, 1)
        Testing set labels
    """
    if not (0 < train_size <= 1):
        raise ValueError("train_size must be between 0 and 1.")
    
    # Set training size
    train_size = int(X.shape[0] * train_size) 

    # Shuffle the data or not
    if shuffle:
        # Set the seed for the randomizer
        np.random.seed(int(random_state))
        # Randomize the indices for the dataset
        indices = np.random.permutation(X.shape[0]) # Randomize rows
    
    else:
        indices = np.arange(X.shape[0])

    # Access rows. Note these are NumPy arrays!
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # If input X, y is a pandas df
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_indices, :], X.iloc[test_indices, :]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # If input X, y is assumed as a numpy array
    else:
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]

        # Ensure y_train and y_test retain 1D shape if y is originally 1D
        if y.ndim == 1:
            y_train, y_test = y_train.flatten(), y_test.flatten()

    return X_train, X_test, y_train, y_test


def k_fold(X, n_splits, shuffle, random_state):
    """
    Splits the data into folds (n_splits)
    
    Parameters
    ----------
    X : array like of shape (m, n)
        The training set to be split into (n_splits) folds
    n_splits : int
        Number of splits
    shuffle : bool
        Shuffle the dataset
    random_state : int
        To fix the randomizer seed for reproducibility
    Returns
    -------
    folds : list
        A list of (n_splits) size of dictionaries showing the 
        training indices and validation indices
    """
    
    # Store dictionary of test and train indices per fold
    folds = []
    
    # Shuffle the data
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(X.shape[0])
    
    else:
        indices = np.arange(X.shape[0])

    # Compute the base validation size and distribute extra samples
    base_validation_size = X.shape[0] // n_splits
    extra_samples = X.shape[0] % n_splits
    
    # Set starting point for the indices and increment it per fold
    starting_index = 0

    for fold in range(n_splits):
        
        # Compute actual validation size for this fold
        validation_size = base_validation_size + (1 if fold < extra_samples else 0)

        # Specify which indices to use for training and validation
        validation_indices = indices[starting_index:starting_index + validation_size]
        training_indices = np.concatenate((indices[:starting_index], indices[starting_index + validation_size:]))

        # Create a dictionary of indices for train and validation
        train_valid_dict = {
            "Training indices" : training_indices,
            "Validation indices" : validation_indices
        }
        
        # Append the created dictionary
        folds.append(train_valid_dict)
        
        # Increment starting index to a new starting index
        starting_index += validation_size

    return folds

def optimize_lambda(lambda_range, X, y, n_splits, random_state, verbose=True, 
                    no_iterations=1000, show_loss_plot=False, log_scale= True, shuffle=True):
    """
    Determine the optimal regularization strength L2 lambda using KFold cross validation

    Parameters
    ----------
    lambda_range : array like showing a sequence of numbers
        Specify the range of lambdas to which the function will 
        iterate. 
    X : array like of shape (m, n)
        The training set.
    y : array like of shape (m, 1)
        The training labels
    n_splits : int
        Number of splits in the training set setting a different
        training and validation in each split through KFold
    random_state : int
        To fix the randomizer seed for reproducibility
    verbose : bool, default=True
        Show how the average loss updates per lambda in the lambda 
        range.
    no_iterations : int , default=1000
        Set the number of iterations per fold. Use 1000 by default
        as this is robust for a computationally expensive procedure.
    show_loss_plot : bool, default=False
        Show the loss plot of average loss vs lambda
    log_scale : bool, default=True
        Log scale the lambda in the plot should you be using logspace
        for the range of lambdas.
    shuffle : bool, default=True
        Shuffle the data
    
    Method
    ------
    1. Create the indices of training and validation split per fold using the k_fold function.
    2. Iterating through each lambda, iterate through each fold doing the following in 
    succession:
        - Obtain X_tr, y_tr, X_valid, y_valid using the indices generated by k_fold.
        - Convert X_tr, y_tr, X_valid, y_valid to ndarray.
        - For X_tr, y_tr in each fold, determine thetas. Please remember that we 
        regularize in this procedure.
        - Using thetas obtained after fitting the softmax to X_tr, y_tr, determine 
        the softmax output inputting X_valid.
        - Evaluate the loss using the softmax_cost function by comparing the softmax
        output using X_valid to the y_valid. Please remember that we DO NOT regularize
        in this procedure.
        - Append each loss associated per fold to the fold_loss. 
        - Obtain the mean of fold_loss per lambda which we will append to the avg_lam_loss
        for evaluation of each lambda in the lambda range.
        - Obtain the best lambda by index by obtaining the index of the lowest avg_lam_loss
        using np.argmin().

    Returns
    -------
    Return user readable information on Best lambda and average validation loss. User has the option
    to visualize the trend of average validation loss in a given range of lambdas.
    """
    
    folds = k_fold(X, n_splits, shuffle, random_state)

    avg_lam_loss = []

    for lam in lambda_range:

        fold_loss = []

        for split, fold in enumerate(folds):
            train_idx = fold['Training indices']
            val_idx = fold['Validation indices']

            # Check if X is a dataframe
            # Convert everything to NumPy ndarray 
            # Note y_tr and y_val are one hot encoded to k classes i.e. shape (no_indices, k) 
            # Only one hot encode y_val as y_tr is only used on the softmax_regression function
            if isinstance(X, pd.DataFrame):
                X_tr = X.iloc[train_idx,:].values
                y_tr = y.iloc[train_idx].values
                X_val = X.iloc[val_idx,:].values
                y_val = pd.get_dummies(y.iloc[val_idx], dtype=float).values
            # Case when X is NumPy
            elif isinstance(X, np.ndarray):
                X_tr = X[train_idx,:]
                y_tr = y[train_idx]
                X_val = X[val_idx,:]
                # Convert to Series first
                y_val_pd = pd.Series(data=y[val_idx])
                # One hot encode y_val! Then convert to NumPy
                y_val = pd.get_dummies(y_val_pd, dtype=float).values

            # Predict using the estimated theta of each fold
            theta_lam = softmax_regression(X=X_tr, y=y_tr, iterations=no_iterations, learning_rate=0.001, _lambda=lam,
                                           verbose=False, seed=random_state, early_stopping=False)
            
            softmax_output_lam = softmax_output(X=X_val, theta=theta_lam)
            
            # Note: The softmax_cost uses Y so you need to One hot encode the y_vals.

            # Evaluate the softmax output on the y_val
            loss = softmax_cost(softmax_output_lam, Y=y_val, _lambda=lam, theta=theta_lam, regularize=False)

            # Append each loss per split
            fold_loss.append(loss)

        # Append average loss per lambda value
        avg_lam_loss.append(np.mean(fold_loss))
        if verbose:
            print(f"Lambda: {lam:.4f} | Avg Log Loss: {np.mean(fold_loss):.4f}")

    best_lambda = lambda_range[np.argmin(avg_lam_loss)]
    if show_loss_plot:
        # plot x vals lambda range
        x = lambda_range.flatten()
        y = np.array(avg_lam_loss).flatten()
        plt.plot(x, y)
        plt.xlabel('lambda (Regularization strength)')
        plt.ylabel('loss')
        plt.grid(True)
        plt.title('lambda vs log loss')
        if log_scale:
            plt.xscale('log')
        plt.show()

    return f"Best lambda: {best_lambda:.4f}, Average validation Error: {np.min(avg_lam_loss)}"

def determine_optimal_iterations(iteration_range, X, y, _lambda, seed, show_plot=True):
    """
    Return what is the optimal number of iterations after fitting the whole training set
    from a range of iteration values.

    Parameters
    ----------
    iteration_range : array like showing a sequence of numbers
        Evenly spaced sequence (using linspace or logspace) 
    X : array like with shape (m, n)
        The training set features.
    y : array like with shape (m, 1)
        The training set labels.
    _lambda : int
        The best regularization strength as evaluated using KFold.
    seed : int
        Set the randomizer to a fixed value for reproducibility.
    show_plot : bool, default=True
        Option to show the plot of number of iterations with end_costs.
    
    Method
    ------
    1. Perform softmax_regression function given a number of iterations
    2. Determine the end_cost toggling return_final_cost as True in the 
    softmax_regression function.
    3. From the list of end_costs, determine the index of the lowest cost.
    This is asscoiated with the index of the optimal iterations.
    Returns
    -------
    The optimal number of iterations in a human readable format. Option to 
    show plots too!
    """

    # Final cost estimates
    end_costs = []
    for iteration in iteration_range:
        theta, end_cost = softmax_regression(X=X, y=y, iterations=iteration, learning_rate=0.001, _lambda=_lambda, verbose=False, return_final_cost=True, seed=seed)
        end_costs.append(end_cost)

    # Determine optimal num_iterations
    optimal_num_iterations = iteration_range[np.argmin(end_costs)]
    if show_plot:
        data = np.column_stack((iteration_range, end_costs))
        iteration_end_cost = pd.DataFrame(data, columns=['Iterations', 'End Cost'])
        plt.title('Number of iterations vs end cost')
        sns.lineplot(data=iteration_end_cost, x='Iterations', y='End Cost')

    return f"Optimal number of iterations: {optimal_num_iterations}"   


