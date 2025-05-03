import numpy as np
from scipy import optimize
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Custom Logistic Regression with L2 regularization and additional solvers.
    
    Parameters
    ----------
    solver : str, default='bfgs'
        Algorithm to use in the optimization problem.
        - 'bfgs': Full BFGS algorithm (not memory limited)
        - 'fixed-hessian': Fixed Hessian algorithm (uses initial Hessian throughout)
    
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    
    C : float, default=1.0
        Inverse of regularization strength; smaller values specify stronger
        regularization.
    
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    
    intercept_scaling : float, default=1.0
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True.
    
    class_weight : dict or 'balanced', default=None
        Weights associated with classes. If not given, all classes
        are supposed to have weight one.
    
    Attributes
    ----------
    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.
    
    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        Set to 0.0 if fit_intercept=False.
    """
    
    def __init__(self, solver='bfgs', max_iter=100, C=1.0, 
                 fit_intercept=True, intercept_scaling=1, class_weight=None):
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
    
    def _sigmoid(self, z):
        """Sigmoid function to compute probability estimates."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    def _prepare_data(self, X, y=None):
        """Prepare data: add intercept if needed."""
        if self.fit_intercept:
            if sparse.issparse(X):
                X_intercept = sparse.hstack(
                    [sparse.csr_matrix(np.ones((X.shape[0], 1))), X]
                ).tocsr()
            else:
                X_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
            return X_intercept
        return X
    
    def _logistic_loss(self, w, X, y, alpha):
        """Computes the logistic loss and gradient."""
        n_samples, n_features = X.shape

        # Compute the logistic loss
        z = safe_sparse_dot(X, w)
        p = self._sigmoid(z)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        # Add L2 regularization
        loss += 0.5 * alpha * np.dot(w, w)
        
        # Compute the gradient
        diff = p - y
        grad = safe_sparse_dot(X.T, diff) / n_samples
        grad += alpha * w
        
        return loss, grad
    
    def _logistic_loss_with_hessian(self, w, X, y, alpha):
        """Computes the logistic loss, gradient, and Hessian."""
        n_samples, n_features = X.shape
        
        # Loss and gradient
        z = safe_sparse_dot(X, w)
        p = self._sigmoid(z)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        loss += 0.5 * alpha * np.dot(w, w)
        
        diff = p - y
        grad = safe_sparse_dot(X.T, diff) / n_samples
        grad += alpha * w
        
        # Hessian
        p = p * (1 - p)  # p * (1-p) for each sample
        hessian = np.zeros((n_features, n_features))
        
        # This is a bottleneck for large datasets - more efficient implementations exist
        for i in range(n_samples):
            x_i = X[i:i+1].T
            hessian += p[i] * np.dot(x_i, x_i.T)
        
        hessian /= n_samples
        
        # Add regularization to Hessian
        hessian += alpha * np.eye(n_features)
        
        return loss, grad, hessian
    
    def _newton_step(self, w, grad, hessian):
        """Computes a Newton step."""
        # Solve H * delta = -grad
        try:
            L = np.linalg.cholesky(hessian)
            # First solve L * y = -grad
            y = np.linalg.solve(L, -grad)
            # Then solve L.T * delta = y
            delta = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # Fallback: if Hessian not positive definite
            delta = np.linalg.solve(hessian + 1e-8 * np.eye(hessian.shape[0]), -grad)
        
        return delta
    
    def _solve_fixed_hessian(self, X, y, alpha):
        """Fixed Hessian optimization method."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        
        # Calculate initial loss, gradient and Hessian
        loss_old, grad, hessian = self._logistic_loss_with_hessian(w, X, y, alpha)
        
        # Store the fixed Hessian (computed only once)
        fixed_hessian = hessian.copy()
        
        # Optimization loop
        for iteration in range(self.max_iter):
            # Compute Newton step using fixed Hessian
            newton_step = self._newton_step(w, grad, fixed_hessian)
            
            # Line search (simple backtracking)
            step_size = 1.0
            while step_size > 1e-10:
                w_new = w + step_size * newton_step
                loss_new, grad_new = self._logistic_loss(w_new, X, y, alpha)
                
                if loss_new <= loss_old:
                    break
                
                step_size *= 0.5
            
            # Update parameters
            w = w_new
            grad = grad_new
            loss_old = loss_new
            
            # Check convergence
            if np.linalg.norm(grad) < 1e-5:
                break
        
        return w
    
    def _solve_bfgs(self, X, y, alpha):
        """Full BFGS optimization method."""
        n_samples, n_features = X.shape
        
        def func(w):
            return self._logistic_loss(w, X, y, alpha)[0]
        
        def grad(w):
            return self._logistic_loss(w, X, y, alpha)[1]
        
        w0 = np.zeros(n_features)
        
        # Use scipy's BFGS optimizer
        result = optimize.minimize(
            func, w0, method='BFGS', jac=grad,
            options={'maxiter': self.max_iter, 'gtol': 1e-5, 'disp': False}
        )
        
        return result.x
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse='csr', y_numeric=True)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) > 2:
            raise ValueError("This solver only supports binary classification")
        
        # Preprocessing for binary classification
        self.label_binarizer_ = LabelBinarizer()
        y_bin = self.label_binarizer_.fit_transform(y).ravel()
        
        # Apply class weights
        sample_weight = np.ones(X.shape[0])
        if self.class_weight:
            if self.class_weight == 'balanced':
                n_samples = X.shape[0]
                n_classes = len(self.classes_)
                weight_per_class = n_samples / (n_classes * np.bincount(y))
                for i, c in enumerate(self.classes_):
                    sample_weight[y == c] = weight_per_class[i]
            else:
                for c in self.class_weight:
                    sample_weight[y == c] = self.class_weight[c]
        
        # Add intercept if needed
        X_intercept = self._prepare_data(X)
        
        # Regularization strength
        alpha = 1.0 / (self.C * X.shape[0])
        
        # Select solver
        if self.solver == 'fixed-hessian':
            w = self._solve_fixed_hessian(X_intercept, y_bin, alpha)
        elif self.solver == 'bfgs':
            w = self._solve_bfgs(X_intercept, y_bin, alpha)
        else:
            raise ValueError(f"Solver {self.solver} not supported.")
        
        # Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = np.array([w[0]])
            self.coef_ = w[1:].reshape(1, -1)
        else:
            self.intercept_ = np.array([0.0])
            self.coef_ = w.reshape(1, -1)
        
        return self
    
    def predict_proba(self, X):
        """
        Probability estimates.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored.
        
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        
        X_intercept = self._prepare_data(X)
        scores = safe_sparse_dot(X_intercept, np.hstack([self.intercept_, self.coef_[0]]))
        
        proba = self._sigmoid(scores)
        return np.vstack([1 - proba, proba]).T
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.label_binarizer_.inverse_transform(proba[:, 1].reshape(-1, 1))
    
    def get_confusion_matrix(self, X, y):
        """
        Compute confusion matrix to evaluate the accuracy of a classification.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        cm : ndarray of shape (n_classes, n_classes)
            Confusion matrix whose i-th row and j-th column entry indicates the
            number of samples with true label being i-th class and predicted
            label being j-th class.
        """
        check_is_fitted(self)
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def get_classification_report(self, X, y, output_dict=False):
        """
        Build a text report showing the main classification metrics.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        y : array-like of shape (n_samples,)
            True labels for X.
            
        output_dict : bool, default=False
            If True, return output as dict, else return as string.
            
        Returns
        -------
        report : string or dict
            Classification report with precision, recall, F1 score for each class.
        """
        check_is_fitted(self)
        y_pred = self.predict(X)
        return classification_report(y, y_pred, output_dict=output_dict)