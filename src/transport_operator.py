from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List
from src.activation_loader import ActivationDataset, EfficientActivationDataset
from torch.utils.data import DataLoader
import time
import os
import hashlib
import pickle


class TransportOperator(BaseEstimator, TransformerMixin):
    """
    Transport operator for mapping upstream residual stream vectors to downstream vectors.

    This operator learns a linear transformation that maps activations from one layer
    (upstream) to another layer (downstream) in a neural network's residual stream.
    """

    def __init__(
        self,
        L: int,
        k: int,
        method: str = "ridge",
        regularization: Optional[float] = None,
        l1_ratio: Optional[float] = 0.1,
        normalize: bool = False,
        auto_tune: bool = True,
        cv_folds: int = 5,
        scoring: str = "r2",
        random_state: Optional[int] = 42,
        max_iter: int = 5000,
        tol: float = 1e-3,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        n_proc_cv: int = -1,
    ):
        """
        Initialize the transport operator.

        Args:
            L: Layer number
            k: Offset for the target layer
            method: Type of transformation ('linear', 'ridge', 'lasso', 'elasticnet')
            regularization: Regularization strength (alpha) for regularized methods
            l1_ratio: ElasticNet mixing parameter (0=Ridge, 1=Lasso)
            normalize: Whether to normalize features before fitting
            auto_tune: Whether to automatically tune hyperparameters
            cv_folds: Number of cross-validation folds for hyperparameter tuning
            scoring: Scoring metric for cross-validation ('r2', 'neg_mean_squared_error')
            random_state: Random state for reproducibility
            max_iter: Maximum iterations for iterative solvers
            tol: Tolerance for convergence
            cache_dir: Directory to store cached X, y matrices. If None, uses 'cache' in current directory
            use_cache: Whether to use caching for X, y matrices
            n_proc_cv: Number of processes to use for cross-validation. Defaults to -1 (all available cores).
        """
        self.L = L
        self.k = k
        self.method = method
        self.regularization = regularization
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        self.auto_tune = auto_tune
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.cache_dir = cache_dir or "cache"
        self.use_cache = use_cache
        self.n_proc_cv = n_proc_cv

        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_fitted_ = False
        self.best_params_ = None
        self.cv_results_ = None
        self.feature_importance_ = None

    def _get_param_grid(self) -> Dict[str, List]:
        """Get parameter grid for hyperparameter tuning."""
        if self.method == "ridge":
            return {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 2000.0, 5000.0, 10000.0]}
        elif self.method == "lasso":
            return {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        elif self.method == "elasticnet":
            return {
                "alpha": [0.1, 1.0, 10.0, 100.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        else:
            return {}

    def _create_model(self, **params) -> BaseEstimator:
        """Create the appropriate regression model with given parameters."""
        if self.method == "linear":
            return LinearRegression(fit_intercept=True)
        elif self.method == "ridge":
            alpha = params.get("alpha", self.regularization or 1.0)
            return Ridge(
                alpha=alpha, fit_intercept=True, random_state=self.random_state
            )
        elif self.method == "lasso":
            alpha = params.get("alpha", self.regularization or 1.0)
            return Lasso(
                alpha=alpha,
                fit_intercept=True,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        elif self.method == "elasticnet":
            alpha = params.get("alpha", self.regularization or 1.0)
            l1_ratio = params.get("l1_ratio", self.l1_ratio)
            return ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Choose from 'linear', 'ridge', 'lasso', 'elasticnet'"
            )

    def _get_cache_filename(
        self, dataset: ActivationDataset | EfficientActivationDataset
    ) -> str:
        """Generate a unique cache filename based on dataset characteristics."""
        # Create a hash based on dataset properties that would affect the X, y matrices
        dataset_info = {
            "dataset_type": type(dataset).__name__,
            # Add dataset id or other identifying properties if available
            "dataset_id": getattr(dataset, "dataset_id", "default"),
        }

        # Create hash from dataset info
        dataset_str = str(sorted(dataset_info.items()))
        dataset_hash = hashlib.md5(dataset_str.encode()).hexdigest()[:8]

        return f"transport_data_{dataset_hash}.pkl"

    def _get_model_cache_filename(
        self, dataset: ActivationDataset | EfficientActivationDataset
    ) -> str:
        """Generate a unique model cache filename based on dataset and model parameters."""
        # Create a hash based on both dataset and model parameters
        dataset_info = {
            "dataset_type": type(dataset).__name__,
            "dataset_id": getattr(dataset, "dataset_id", "default"),
        }

        # Include model parameters that affect training
        model_params = {
            "L": self.L,
            "k": self.k,
            "method": self.method,
            "regularization": self.regularization,
            "l1_ratio": self.l1_ratio,
            "normalize": self.normalize,
            "auto_tune": self.auto_tune,
            "cv_folds": self.cv_folds,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

        # Combine dataset and model info
        combined_info = {**dataset_info, **model_params}
        combined_str = str(sorted(combined_info.items()))
        combined_hash = hashlib.md5(combined_str.encode()).hexdigest()[:8]

        return f"transport_model_{combined_hash}.pkl"

    def _save_cache(self, X: np.ndarray, y: np.ndarray, cache_path: str) -> None:
        """Save X and y matrices to cache file."""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cache_data = {
                "X": X,
                "y": y,
                "timestamp": time.time(),
                "X_shape": X.shape,
                "y_shape": y.shape,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"  Cached data saved to: {cache_path}")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

    def _save_model_cache(self, cache_path: str) -> None:
        """Save the complete fitted model state to cache file."""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            model_data = {
                "model": self.model,
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
                "is_fitted_": self.is_fitted_,
                "best_params_": self.best_params_,
                "cv_results_": self.cv_results_,
                "feature_importance_": self.feature_importance_,
                "timestamp": time.time(),
                "method": self.method,
                "normalize": self.normalize,
                "L": self.L,
                "k": self.k,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"  Cached model saved to: {cache_path}")
        except Exception as e:
            print(f"  Warning: Failed to save model cache: {e}")

    def _load_model_cache(self, cache_path: str) -> bool:
        """Load the complete fitted model state from cache file."""
        try:
            if not os.path.exists(cache_path):
                return False

            with open(cache_path, "rb") as f:
                model_data = pickle.load(f)

            # Validate that the cached model parameters match current instance
            if (
                model_data.get("method") != self.method
                or model_data.get("normalize") != self.normalize
            ):
                print(f"  Warning: Cached model parameters don't match, ignoring cache")
                return False

            # Load the model state
            self.model = model_data["model"]
            self.scaler_X = model_data["scaler_X"]
            self.scaler_y = model_data["scaler_y"]
            self.is_fitted_ = model_data["is_fitted_"]
            self.best_params_ = model_data["best_params_"]
            self.cv_results_ = model_data["cv_results_"]
            self.feature_importance_ = model_data["feature_importance_"]
            self.L = model_data["L"]
            self.k = model_data["k"]

            timestamp = model_data.get("timestamp", 0)
            cache_age = time.time() - timestamp

            print(f"  Loaded cached model from: {cache_path}")
            print(f"  Model cache age: {cache_age / 3600:.1f} hours")

            return True

        except Exception as e:
            print(f"  Warning: Failed to load model cache: {e}")
            return False

    def _load_cache(
        self, cache_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load X and y matrices from cache file."""
        try:
            if not os.path.exists(cache_path):
                return None, None

            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            X = cache_data["X"]
            y = cache_data["y"]
            timestamp = cache_data.get("timestamp", 0)

            # Basic validation
            if X.ndim != 2 or y.ndim != 2:
                print(f"  Warning: Invalid cached data shape, ignoring cache")
                return None, None

            cache_age = time.time() - timestamp
            print(f"  Loaded cached data from: {cache_path}")
            print(f"  Cache age: {cache_age / 3600:.1f} hours")
            print(f"  X shape: {X.shape}, y shape: {y.shape}")

            return X, y

        except Exception as e:
            print(f"  Warning: Failed to load cache: {e}")
            return None, None

    def clear_cache(
        self, dataset: Optional[ActivationDataset | EfficientActivationDataset] = None
    ) -> None:
        """
        Clear cached data files.

        Args:
            dataset: If provided, clear cache only for this specific dataset.
                    If None, clear all cache files in the cache directory.
        """
        try:
            if not os.path.exists(self.cache_dir):
                print("Cache directory does not exist.")
                return

            if dataset is not None:
                # Clear cache for specific dataset
                cache_filename = self._get_cache_filename(dataset)
                eval_cache_filename = f"eval_{cache_filename}"
                model_cache_filename = self._get_model_cache_filename(dataset)

                files_to_remove = [
                    cache_filename,
                    eval_cache_filename,
                    model_cache_filename,
                ]
                removed_count = 0

                for filename in files_to_remove:
                    cache_path = os.path.join(self.cache_dir, filename)
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                        removed_count += 1
                        print(f"Removed cache file: {filename}")

                if removed_count == 0:
                    print("No cache files found for the specified dataset.")
                else:
                    print(f"Cleared {removed_count} cache file(s) for dataset.")
            else:
                # Clear all cache files
                cache_files = [
                    f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")
                ]

                if not cache_files:
                    print("No cache files found.")
                    return

                for filename in cache_files:
                    cache_path = os.path.join(self.cache_dir, filename)
                    os.remove(cache_path)
                    print(f"Removed cache file: {filename}")

                print(f"Cleared {len(cache_files)} cache file(s).")

                # Remove cache directory if empty
                if not os.listdir(self.cache_dir):
                    os.rmdir(self.cache_dir)
                    print("Removed empty cache directory.")

        except Exception as e:
            print(f"Error clearing cache: {e}")

    def fit(
        self, dataset: ActivationDataset | EfficientActivationDataset
    ) -> "TransportOperator":
        """
        Fit the transport operator on upstream-downstream vector pairs.

        Args:
            dataset: ActivationDataset | EfficientActivationDataset containing upstream-downstream vector pairs.

        Returns:
            self: Fitted transport operator
        """
        start_time = time.time()

        # Check for cached model first
        if self.use_cache:
            model_cache_filename = self._get_model_cache_filename(dataset)
            model_cache_path = os.path.join(self.cache_dir, model_cache_filename)
            print(f"Checking for cached model: {model_cache_path}")

            if self._load_model_cache(model_cache_path):
                total_time = time.time() - start_time
                print(f"Model loaded from cache in {total_time:.2f}s")
                return self

        # Check for cached data
        X, y = None, None
        if self.use_cache:
            cache_filename = self._get_cache_filename(dataset)
            cache_path = os.path.join(self.cache_dir, cache_filename)
            print(f"Checking for cached data: {cache_path}")
            X, y = self._load_cache(cache_path)

        # If no cached data found, load from dataset
        if X is None or y is None:
            print("Loading training data from dataset...")
            X_list = []
            y_list = []

            sample_count = 0
            skipped_samples = 0
            batch_size = 128

            # use dataloader with 0 workers and batches
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

            for i, (x_up, y_down) in enumerate(dataloader):
                # Convert PyTorch tensors to numpy and ensure they're 1D vectors
                x_np = x_up.detach().cpu().numpy()  # .flatten()
                y_np = y_down.detach().cpu().numpy()  # .flatten()

                # Check for NaN or inf values
                if (
                    np.any(np.isnan(x_np))
                    or np.any(np.isinf(x_np))
                    or np.any(np.isnan(y_np))
                    or np.any(np.isinf(y_np))
                ):
                    skipped_samples += 1
                    continue

                X_list.append(x_np)
                y_list.append(y_np)
                sample_count += batch_size

                # Progress update every 10 batches
                if sample_count % (10 * batch_size) == 0:
                    print(f"  Loaded {sample_count:,} samples...")

            if len(X_list) == 0:
                raise ValueError("No valid samples found in the dataset")

            # Convert lists to numpy arrays
            # X = np.vstack(X_list)
            # y = np.vstack(y_list)
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)

            load_time = time.time() - start_time
            print(
                f"Data loading complete: {sample_count:,} samples loaded ({skipped_samples} skipped)"
            )
            print(f"  X shape: {X.shape}, y shape: {y.shape}")
            print(f"  Loading time: {load_time:.2f}s")

            # Save to cache if enabled
            if self.use_cache:
                cache_filename = self._get_cache_filename(dataset)
                cache_path = os.path.join(self.cache_dir, cache_filename)
                self._save_cache(X, y, cache_path)
        else:
            load_time = time.time() - start_time
            print(f"Data loaded from cache in {load_time:.2f}s")

        # Store original data for potential normalization
        X_fit, y_fit = X.copy(), y.copy()

        # Apply normalization if requested
        if self.normalize:
            print("Applying feature normalization...")
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            X_fit = self.scaler_X.fit_transform(X_fit)
            y_fit = self.scaler_y.fit_transform(y_fit)

        # Model training
        train_start = time.time()

        # Hyperparameter tuning
        if self.auto_tune and self.method != "linear":
            param_grid = self._get_param_grid()
            if param_grid:
                print(f"Starting hyperparameter tuning for {self.method} regression...")
                print(f"  Parameter grid: {param_grid}")
                print(f"  CV folds: {self.cv_folds}")

                base_model = self._create_model()

                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=self.n_proc_cv,
                    verbose=100,
                    # random_state=self.random_state
                )

                grid_search.fit(X_fit, y_fit)
                self.best_params_ = grid_search.best_params_
                self.cv_results_ = grid_search.cv_results_
                self.model = grid_search.best_estimator_

                print(f"  Best parameters: {self.best_params_}")
                print(f"  Best CV score: {grid_search.best_score_:.4f}")
            else:
                print(f"Training {self.method} regression model...")
                self.model = self._create_model()
                self.model.fit(X_fit, y_fit)
        else:
            print(f"Training {self.method} regression model...")
            self.model = self._create_model()
            self.model.fit(X_fit, y_fit)

        train_time = time.time() - train_start

        # Calculate feature importance for regularized methods
        if hasattr(self.model, "coef_"):
            self.feature_importance_ = np.abs(self.model.coef_).mean(axis=0)

        self.is_fitted_ = True

        # Save trained model to cache if enabled
        if self.use_cache:
            model_cache_filename = self._get_model_cache_filename(dataset)
            model_cache_path = os.path.join(self.cache_dir, model_cache_filename)
            self._save_model_cache(model_cache_path)

        total_time = time.time() - start_time
        print(f"Training complete:")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")

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
            raise ValueError(
                "Transport operator must be fitted before making predictions"
            )

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

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "r2"
    ) -> Dict[str, float]:
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
            f"{scoring}_mean": scores.mean(),
            f"{scoring}_std": scores.std(),
            "scores": scores,
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
            "r2_score": r2_score(y, y_pred),
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        }

        # Add per-output summary statistics for multi-output case
        if y.ndim > 1 and y.shape[1] > 1:
            # Calculate per-output metrics
            per_output_r2 = []
            per_output_mse = []

            for i in range(y.shape[1]):
                r2_i = r2_score(y[:, i], y_pred[:, i])
                mse_i = mean_squared_error(y[:, i], y_pred[:, i])
                per_output_r2.append(r2_i)
                per_output_mse.append(mse_i)

            # Add summary statistics instead of individual values
            per_output_r2 = np.array(per_output_r2)
            per_output_mse = np.array(per_output_mse)

            metrics.update(
                {
                    "r2_per_output_mean": per_output_r2.mean(),
                    "r2_per_output_std": per_output_r2.std(),
                    "r2_per_output_min": per_output_r2.min(),
                    "r2_per_output_max": per_output_r2.max(),
                    "r2_per_output_median": np.median(per_output_r2),
                    "mse_per_output_mean": per_output_mse.mean(),
                    "mse_per_output_std": per_output_mse.std(),
                    "mse_per_output_min": per_output_mse.min(),
                    "mse_per_output_max": per_output_mse.max(),
                    "mse_per_output_median": np.median(per_output_mse),
                    "num_outputs": y.shape[1],
                }
            )

        return metrics

    def evaluate_dataset(self, dataset) -> Dict[str, float]:
        """
        Evaluate the model on an ActivationDataset | EfficientActivationDataset.

        Args:
            dataset: ActivationDataset | EfficientActivationDataset to evaluate on

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted_:
            raise ValueError("Transport operator must be fitted before evaluation")

        start_time = time.time()

        # Check for cached evaluation data first
        X, y = None, None
        if self.use_cache:
            cache_filename = f"eval_{self._get_cache_filename(dataset)}"
            cache_path = os.path.join(self.cache_dir, cache_filename)
            print(f"Checking for cached evaluation data: {cache_path}")
            X, y = self._load_cache(cache_path)

        # If no cached data found, load from dataset
        if X is None or y is None:
            print("Loading evaluation data from dataset...")
            X_list = []
            y_list = []

            sample_count = 0
            skipped_samples = 0
            batch_size = 128

            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
            for i, (x_up, y_down) in enumerate(dataloader):
                x_np = x_up.detach().cpu().numpy()
                y_np = y_down.detach().cpu().numpy()

                # Check for NaN or inf values
                if (
                    np.any(np.isnan(x_np))
                    or np.any(np.isinf(x_np))
                    or np.any(np.isnan(y_np))
                    or np.any(np.isinf(y_np))
                ):
                    skipped_samples += 1
                    continue

                X_list.append(x_np)
                y_list.append(y_np)
                sample_count += batch_size

                # Less frequent progress updates
                if sample_count % (batch_size * 10) == 0:
                    print(f"  Loaded {sample_count:,} evaluation samples...")

            if len(X_list) == 0:
                raise ValueError("No valid evaluation samples found")

            # X = np.vstack(X_list)
            # y = np.vstack(y_list)
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)

            load_time = time.time() - start_time
            print(
                f"Evaluation data loaded: {sample_count:,} samples ({skipped_samples} skipped, {load_time:.2f}s)"
            )

            # Save to cache if enabled
            if self.use_cache:
                cache_filename = f"eval_{self._get_cache_filename(dataset)}"
                cache_path = os.path.join(self.cache_dir, cache_filename)
                self._save_cache(X, y, cache_path)
        else:
            load_time = time.time() - start_time
            print(f"Evaluation data loaded from cache in {load_time:.2f}s")

        # Evaluate
        metrics = self.evaluate(X, y)

        # Log key metrics in a clean format
        print(f"Evaluation results:")
        print(f"  Overall R² Score: {metrics['r2_score']:.4f}")
        print(f"  Overall RMSE: {metrics['rmse']:.6f}")
        print(f"  Overall MSE: {metrics['mse']:.6f}")

        # Print per-output summary if multi-output
        if "num_outputs" in metrics:
            print(f"  Multi-output summary ({metrics['num_outputs']} outputs):")
            print(
                f"    R² per output - Mean: {metrics['r2_per_output_mean']:.4f}, "
                f"Std: {metrics['r2_per_output_std']:.4f}, "
                f"Range: [{metrics['r2_per_output_min']:.4f}, {metrics['r2_per_output_max']:.4f}]"
            )
            print(
                f"    MSE per output - Mean: {metrics['mse_per_output_mean']:.6f}, "
                f"Std: {metrics['mse_per_output_std']:.6f}, "
                f"Range: [{metrics['mse_per_output_min']:.6f}, {metrics['mse_per_output_max']:.6f}]"
            )

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
            "method": self.method,
            "regularization": self.regularization,
            "l1_ratio": self.l1_ratio,
            "normalize": self.normalize,
            "auto_tune": self.auto_tune,
            "cv_folds": self.cv_folds,
            "scoring": self.scoring,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "cache_dir": self.cache_dir,
            "use_cache": self.use_cache,
        }

    def set_params(self, **params) -> "TransportOperator":
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """String representation of the transport operator."""
        params = ", ".join(
            [f"{k}={v}" for k, v in self.get_params().items() if v is not None]
        )
        return f"TransportOperator({params})"


class IdentityBaselineTransportOperator(TransportOperator):
    """
    Baseline transport operator that simply returns the upstream vectors as they are.
    This is useful for comparing against more complex models.
    """

    def __init__(self, L: int, k: int, **kwargs):
        """
        Initialize the Identity baseline transport operator.

        Args:
            L: Layer number
            k: Offset for the target layer
            **kwargs: Additional parameters for TransportOperator
        """
        super().__init__(L, k, method="identity", **kwargs)
        self.is_fitted_ = True  # Identity operator does not require fitting

    def fit(
        self, dataset: ActivationDataset | EfficientActivationDataset
    ) -> "IdentityBaselineTransportOperator":
        """
        Fit the identity transport operator on downstream vectors.

        Args:
            dataset: ActivationDataset | EfficientActivationDataset containing upstream-downstream vector pairs.

        Returns:
            self: Fitted identity transport operator (no actual fitting needed).
        """
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict downstream vectors by returning upstream vectors as they are.

        Args:
            X: Upstream residual stream vectors, shape (n_samples, n_features_upstream)

        Returns:
            Predicted downstream vectors, shape (n_samples, n_features_upstream)
        """

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        return X


def load_transport_operator(
    L: int,
    k: int,
    operators_dir: str,
) -> TransportOperator:
    """Load the transport operator from the cache or create a new one."""
    # Warning: This is a temporary and arguably a little bit dodgy, duck tape solution
    # TODO: this is really not the nicest way of doing it. Consider reimplementing
    operator = TransportOperator(
        L,
        k,
        regularization=10.0,
        max_iter=500,
    )
    dummy_ds = EfficientActivationDataset(None, [], "", 0, 0, f"train_L{L}_k{k}")
    file_name = operator._get_model_cache_filename(dummy_ds)
    is_loaded = operator._load_model_cache(Path(operators_dir) / file_name)
    if not is_loaded:
        raise FileNotFoundError(f"Transport operator model not found: {file_name}")
    return operator
