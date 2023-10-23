from sklearn.base import BaseEstimator
import numpy as np

class CategoricalNaiveBayes(BaseEstimator):
    def __init__(self, b_i=2.0, b_j=2.0, alpha=1.0, use_map=False):
        self.b_i = b_i  # Beta prior hyperparameter
        self.b_j = b_j    # Beta prior hyperparameter
        self.alpha = alpha # Dirichlet prior hyperparameter
        self.use_map = use_map

    def fit(self, X, y):
        # All part of the Base Estimator parameters
        assert len(X) == len(y)
        self.classes_ = np.unique(y) # unique class labels in the training data
        self.n_classes_ = len(self.classes_) # Number of provided classes
        self.n_features_ = X[0].shape[0] * X[0].shape[1]
        self.class_counts = np.bincount(y) # number of samples from each class
        self.mean_features = []
        
        for c in self.classes_:
            class_mask = (y == c) # in the labels array marks True if C matches the class
            class_features = X[class_mask] # gets all the samples matching the class C
            
            # Adds all the pixels together then normalize
            # p(x|y) probability of a feature appearing given some class C
            if self.use_map:
                self.mean_features.append((class_features.sum(axis=0) + self.b_i - 1) / (self.class_counts[c] + self.b_i + self.b_j - 2))
            else:
                self.mean_features.append((class_features.sum(axis=0) +1) / (self.class_counts[c]+2))
            
        #######
        # We have the following info
        # - Feature Probs: Has the probability that you see a pixel given some class C
        # - Class Counts: The number of samples in each class
        #######
        
        # p(y) probability of Y or probability of a class
        if self.use_map:
            self.class_probs = [(c + self.alpha - 1) / (len(y) + (self.n_classes_ * self.alpha) - self.n_classes_ )for c in self.class_counts]
        else:
            self.class_probs = [c / len(y) for c in self.class_counts]
        
        # get the log values (this will help for prediction)
        self.log_feature_probs = [np.log(fp) for fp in self.mean_features]
        self.log_feature_probs_neg = [np.log(1 - fp) for fp in self.mean_features]
        self.log_class_priors = [np.log(cp) for cp in self.class_probs]
        
    def bayes_calculate(self, x):
        x = np.expand_dims(x, 0)
        
        # See how many of the pixels line up with the priors (can be 0 or 1)
        prob_x_given_y = self.log_feature_probs * x + self.log_feature_probs_neg * (1 - x)
        prob_x_given_y = prob_x_given_y.reshape(self.n_classes_, -1).sum(axis=1)
        return prob_x_given_y + self.log_class_priors   

    def score(self, X, y):
        y_scores = [sum(self.bayes_calculate(x)) for x in X]
        return sum(y_scores)/len(y)
