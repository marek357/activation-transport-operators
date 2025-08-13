import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Optional, Dict, Any, List
from src.activation_loader import ActivationDataset
import warnings

class TransportOperator(BaseEstimator, TransformerMixin):
    """
    Transport operator for mapping upstream residual stream vectors to downstream vectors.
    
    This operator learns a linear transformation that maps activations from one layer
    (upstream) to another layer (downstream) in a neural network's residual stream.
    """
    
    def __init__(self, 
                 method: str = 'linear', 
                 regularization: Optional[float] = None,
                 l1_ratio: Optional[float] = 0.5,
                 normalize: bool = False,
                 auto_tune: bool = False,
                 cv_folds: int = 5,
                 scoring: str = 'r2',
                 random_state: Optional[int] = None):
        """
        Initialize the transport operator.
        
        Args:
            method: Type of transformation ('linear', 'ridge', 'lasso', 'elasticnet')
            regularization: Regularization strength (alpha) for regularized methods
            l1_ratio: ElasticNet mixing parameter (0=Ridge, 1=Lasso)
            normalize: Whether to normalize features before fitting
            auto_tune: Whether to automatically tune hyperparameters
            cv_folds: Number of cross-validation folds for hyperparameter tuning
            scoring: Scoring metric for cross-validation ('r2', 'neg_mean_squared_error')
            random_state: Random state for reproducibility
        """
        self.method = method
        self.regularization = regularization
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        self.auto_tune = auto_tune
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_fitted_ = False
        self.best_params_ = None
        self.cv_results_ = None
        self.feature_importance_ = None

    def _get_param_grid(self) -> Dict[str, List]:
        """Get parameter grid for hyperparameter tuning."""
        if self.method == 'ridge':
            return {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        elif self.method == 'lasso':
            return {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        elif self.method == 'elasticnet':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        else:
            return {}

    def _create_model(self, **params) -> BaseEstimator:
        """Create the appropriate regression model with given parameters."""
        if self.method == 'linear':
            return LinearRegression(fit_intercept=True)
        elif self.method == 'ridge':
            alpha = params.get('alpha', self.regularization or 1.0)
            return Ridge(alpha=alpha, fit_intercept=True, random_state=self.random_state)
        elif self.method == 'lasso':
            alpha = params.get('alpha', self.regularization or 1.0)
            return Lasso(alpha=alpha, fit_intercept=True, random_state=self.random_state, max_iter=2000)
        elif self.method == 'elasticnet':
            alpha = params.get('alpha', self.regularization or 1.0)
            l1_ratio = params.get('l1_ratio', self.l1_ratio)
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, 
                            random_state=self.random_state, max_iter=2000)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'linear', 'ridge', 'lasso', 'elasticnet'")

    def fit(self, dataset: ActivationDataset) -> 'TransportOperator':
        """
        Fit the transport operator on upstream-downstream vector pairs.
        
        Args:
            dataset: ActivationDataset containing upstream-downstream vector pairs.

        Returns:
            self: Fitted transport operator
        """

        # Convert activation dataset object to X and y
        X_list = []
        y_list = []
        
        print("Loading data from ActivationDataset...")
        for i, (x_up, y_down) in enumerate(dataset):
            # Convert PyTorch tensors to numpy and ensure they're 1D vectors
            x_np = x_up.detach().cpu().numpy().flatten()
            y_np = y_down.detach().cpu().numpy().flatten()
            
            X_list.append(x_np)
            y_list.append(y_np)
            
            # Optional: Print progress for large datasets
            if (i + 1) % 1000 == 0:
                print(f"Loaded {i + 1} samples...")
        
        if len(X_list) == 0:
            raise ValueError("No valid samples found in the dataset")
        
        # Convert lists to numpy arrays
        X = np.vstack(X_list)
        y = np.vstack(y_list)
        
        print(f"Dataset loaded: X shape {X.shape}, y shape {y.shape}")

        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}")
        
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must be 2D arrays")
        
        # Store original data for potential normalization
        X_fit, y_fit = X.copy(), y.copy()
        
        # Apply normalization if requested
        if self.normalize:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            X_fit = self.scaler_X.fit_transform(X_fit)
            y_fit = self.scaler_y.fit_transform(y_fit)
        
        # Hyperparameter tuning
        if self.auto_tune and self.method != 'linear':
            param_grid = self._get_param_grid()
            if param_grid:
                base_model = self._create_model()
                
                grid_search = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=-1,
                    random_state=self.random_state
                )
                
                grid_search.fit(X_fit, y_fit)
                self.best_params_ = grid_search.best_params_
                self.cv_results_ = grid_search.cv_results_
                self.model = grid_search.best_estimator_
            else:
                self.model = self._create_model()
                self.model.fit(X_fit, y_fit)
        else:
            self.model = self._create_model()
            self.model.fit(X_fit, y_fit)
        
        # Calculate feature importance for regularized methods
        if hasattr(self.model, 'coef_'):
            self.feature_importance_ = np.abs(self.model.coef_).mean(axis=0)
        
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict downstream residual vectors from upstream vectors.
        
        Args:
            X: Upstream residual stream vectors, shape (n_samples, n_features_upstream)
            
        Returns:
            Predicted downstream vectors, shape (n_samples, n_features_downstream)
        """
        if not self.is_fitted_:
            raise ValueError("Transport operator must be fitted before making predictions")
            
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        
        X_pred = X.copy()
        
        # Apply normalization if it was used during fitting
        if self.normalize and self.scaler_X is not None:
            X_pred = self.scaler_X.transform(X_pred)
        
        y_pred = self.model.predict(X_pred)
        
        # Inverse transform predictions if normalization was used
        if self.normalize and self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)
        
        return y_pred

    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
        """
        Perform cross-validation to evaluate model performance.
        
        Args:
            X: Input features
            y: Target values
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with mean and std of cross-validation scores
        """
        if not self.is_fitted_:
            raise ValueError("Transport operator must be fitted first")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            f'{scoring}_mean': scores.mean(),
            f'{scoring}_std': scores.std(),
            'scores': scores
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        # Add per-output metrics for multi-output case
        if y.ndim > 1 and y.shape[1] > 1:
            for i in range(y.shape[1]):
                metrics[f'r2_output_{i}'] = r2_score(y[:, i], y_pred[:, i])
                metrics[f'mse_output_{i}'] = mean_squared_error(y[:, i], y_pred[:, i])
        
        return metrics

    def get_feature_importance(self, top_k: Optional[int] = None) -> np.ndarray:
        """
        Get feature importance scores.
        
        Args:
            top_k: Return only top k most important features
            
        Returns:
            Feature importance scores
        """
        if not self.is_fitted_ or self.feature_importance_ is None:
            raise ValueError("Feature importance not available")
        
        if top_k is not None:
            indices = np.argsort(self.feature_importance_)[-top_k:]
            return self.feature_importance_[indices]
        
        return self.feature_importance_

    def get_transport_matrix(self) -> np.ndarray:
        """
        Get the learned transport matrix (weights).
        
        Returns:
            Transport matrix, shape (n_features_upstream, n_features_downstream)
        """
        if not self.is_fitted_:
            raise ValueError("Transport operator must be fitted first")
            
        return self.model.coef_.T

    def get_bias(self) -> np.ndarray:
        """
        Get the learned bias/intercept vector.
        
        Returns:
            Bias vector, shape (n_features_downstream,)
        """
        if not self.is_fitted_:
            raise ValueError("Transport operator must be fitted first")
            
        return self.model.intercept_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform upstream vectors to downstream space (alias for predict)."""
        return self.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        if not self.is_fitted_:
            raise ValueError("Transport operator must be fitted before scoring")
            
        X_score = X.copy()
        if self.normalize and self.scaler_X is not None:
            X_score = self.scaler_X.transform(X_score)
            
        return self.model.score(X_score, y)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'method': self.method,
            'regularization': self.regularization,
            'l1_ratio': self.l1_ratio,
            'normalize': self.normalize,
            'auto_tune': self.auto_tune,
            'cv_folds': self.cv_folds,
            'scoring': self.scoring,
            'random_state': self.random_state
        }

    def set_params(self, **params) -> 'TransportOperator':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """String representation of the transport operator."""
        params = ', '.join([f'{k}={v}' for k, v in self.get_params().items() if v is not None])
        return f"TransportOperator({params})"

