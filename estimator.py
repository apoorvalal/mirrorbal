import numpy as np
import scipy.sparse as sps


class SimplexSolver:
    """
    Solves simplex-constrained least squares problems using mirror descent.

    Finds beta that minimizes ||X @ beta - y||^2 subject to beta_i >= 0
    and sum(beta_i within block) = 1 for each block.
    """

    def __init__(self, iters=10_000, tolerance=1e-9):
        self.iters = iters
        self.tolerance = tolerance

        # Intermediate quantities for diagnostics
        self.beta_ = None
        self.n_iterations_ = 0
        self.convergence_history_ = []
        self.Lf_ = None
        self.n_vector_ = None
        self.y_hat_ = None  # The predicted y: X @ beta_
        self.mse_ = None  # Mean squared error: mean((y - y_hat)^2)

    def fit(self, X, y, blocks=None):
        """
        Fits the simplex-constrained least squares model.

        Args:
            X (array-like): Shape (n_observations_for_y, n_coefficients).
                            The matrix transforming coefficients to match y.
            y (array-like): Shape (n_observations_for_y,). The target vector.
            blocks (list of int, optional):
                    A list where each element is the size of a block of coefficients.
                    The sum of elements in `blocks` must be equal to `X.shape[1]`.
                    Coefficients within each block are constrained to sum to 1.
                    If None, all coefficients form a single block, summing to 1.
        Returns:
            self
        """
        if X.ndim == 1:  # Reshape if X is 1D (e.g. one feature, multiple controls)
            X = X.reshape(-1, X.shape[0])
        if y.ndim == 0:  # Reshape if y is scalar (e.g. one feature target)
            y = np.array([y])
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} observations (rows), but y has {y.shape[0]} elements. These must match."
            )

        if X.shape[1] == 0:
            self.beta_ = np.array([])
            self.n_iterations_ = 0
            self.mse_ = np.mean((y - 0) ** 2) if y.size > 0 else 0
            return self

        current_blocks = blocks
        if current_blocks is None:
            current_blocks = [X.shape[1]]

        n_vector_list = []
        for block_size_val in current_blocks:
            if block_size_val > 0:
                n_vector_list.extend([float(block_size_val)] * block_size_val)
        self.n_vector_ = np.array(n_vector_list)

        if len(self.n_vector_) != X.shape[1]:
            raise ValueError(
                f"Sum of block sizes ({sum(current_blocks)}) must equal number of "
                f"coefficients X.shape[1] ({X.shape[1]})"
            )

        if sps.issparse(X):
            if (
                X.shape[0] > 0 and X.shape[1] > 0 and min(X.shape) >= 1
            ):  # svds k must be < min(shape)
                # Ensure k=1 is valid. If min(X.shape)=1, svds(k=1) fails.
                if min(X.shape) > 1:
                    Lf_values = sps.linalg.svds(X, 1, return_singular_vectors=False)
                    self.Lf_ = Lf_values[0] if Lf_values.size > 0 else 1.0
                elif X.shape[0] > 0 and X.shape[1] > 0:  # rank 1 matrix or vector
                    self.Lf_ = np.sqrt(
                        sps.linalg.norm(X, "fro") ** 2
                    )  # Frobenius norm for single singular value
                else:  # empty or zero matrix
                    self.Lf_ = 1.0
            else:
                self.Lf_ = 1.0
        else:  # Dense matrix
            if X.shape[0] > 0 and X.shape[1] > 0:
                Lf_values = np.linalg.svd(X, compute_uv=False)
                self.Lf_ = Lf_values[0] if Lf_values.size > 0 else 1.0
            else:
                self.Lf_ = 1.0

        if self.Lf_ == 0:
            self.Lf_ = 1.0

        def t_func(k_iter, n_vec, Lf_val):
            log_n_vector = np.log(np.maximum(n_vec, 1))
            val = np.sqrt(2 * log_n_vector) / (np.sqrt(max(k_iter, 1)) * Lf_val)
            return np.squeeze(np.asarray(val))

        def compute_gradient(current_X, current_y, x_coeffs):
            inside = np.squeeze(np.asarray(current_X.dot(x_coeffs))) - current_y
            return np.squeeze(np.asarray(current_X.T.dot(inside)))

        beta = np.ones(X.shape[1], dtype=float)
        current_pos = 0
        for block_size_val in current_blocks:
            if block_size_val > 0:
                beta[current_pos : current_pos + block_size_val] = 1.0 / block_size_val
                current_pos += block_size_val
        beta = np.squeeze(np.asarray(beta))

        self.convergence_history_ = []
        for _iter in range(1, self.iters + 1):
            beta_prev = beta.copy()
            grad = compute_gradient(X, y, beta)

            step_size_t_k = t_func(_iter, self.n_vector_, self.Lf_)
            if not isinstance(step_size_t_k, np.ndarray) or step_size_t_k.ndim == 0:
                step_size_t_k = np.full(beta.shape, step_size_t_k)

            up = grad * step_size_t_k
            beta = beta * np.exp(-up)

            beginning = 0
            for block_size_val in current_blocks:
                if block_size_val > 0:
                    beta_section = beta[beginning : beginning + block_size_val]
                    sum_beta_section = np.sum(beta_section)
                    if sum_beta_section > 1e-12:  # Avoid division by zero
                        beta[beginning : beginning + block_size_val] = (
                            beta_section / sum_beta_section
                        )
                    else:  # Re-initialize if sum is too small (e.g. all elements became zero)
                        beta[beginning : beginning + block_size_val] = (
                            1.0 / block_size_val
                        )
                    beginning += block_size_val
            beta = np.squeeze(np.asarray(beta))

            norm_diff = np.linalg.norm(beta - beta_prev, np.inf)
            self.convergence_history_.append(norm_diff)
            if norm_diff < self.tolerance:
                break

        self.beta_ = beta
        self.n_iterations_ = _iter
        self.y_hat_ = X @ self.beta_
        self.mse_ = np.mean((y - self.y_hat_) ** 2)
        return self


######################################################################
######################################################################
######################################################################


class BalancingWeightsEstimator:
    """
    Estimates the Average Treatment Effect on the Treated (ATT)
    using balancing weights derived from SimplexSolver.
    """

    def __init__(self, solver_iters=10_000, solver_tolerance=1e-9):
        self.solver_params = {"iters": solver_iters, "tolerance": solver_tolerance}
        self.gamma_ = None  # Control unit weights
        self.att_ = None
        self.mean_Y_treated_ = None
        self.weighted_mean_Y_control_ = None
        self.X_treated_mean_ = None
        self.X_control_weighted_mean_ = None
        self.solver_ = None  # To store the solver instance for diagnostics

    def fit(self, X_covariates, Y_outcome, W_treatment):
        """
        Fits the estimator by calculating balancing weights and ATT.

        Args:
            X_covariates (np.ndarray): Covariate matrix for all units.
            Y_outcome (np.ndarray): Outcome vector for all units.
            W_treatment (np.ndarray): Treatment assignment vector (0 or 1) for all units.

        Returns:
            self
        """
        X_t = X_covariates[W_treatment == 1]
        Y_t = Y_outcome[W_treatment == 1]
        X_c = X_covariates[W_treatment == 0]
        Y_c = Y_outcome[W_treatment == 0]

        self.n_c, self.n_t = X_c.shape[0], X_t.shape[0]

        if X_t.shape[0] == 0:
            raise ValueError("No treated units found.")
        if X_c.shape[0] == 0:
            raise ValueError("No control units found.")

        # Target for SimplexSolver: mean covariates of treated units
        self.X_treated_mean_ = np.mean(X_t, axis=0)

        # Input for SimplexSolver: control unit covariates (transposed)
        # X_for_solver shape: (n_features, n_control_units)
        X_for_solver = X_c.T

        self.solver_ = SimplexSolver(**self.solver_params)
        self.solver_.fit(
            X_for_solver, self.X_treated_mean_
        )  # blocks=None is default (all controls form one block)
        self.gamma_ = self.solver_.beta_

        # Sanity check for weights
        if not (
            np.all(self.gamma_ >= -1e-9) and np.isclose(np.sum(self.gamma_), 1.0)
        ):  # allow for small numerical error
            print(
                f"Warning: Weights sum to {np.sum(self.gamma_)} or have negative values: {self.gamma_[self.gamma_ < 0]}. Clamping negative weights to 0 and re-normalizing."
            )
            self.gamma_[self.gamma_ < 0] = 0.0
            if np.sum(self.gamma_) > 1e-9:
                self.gamma_ /= np.sum(self.gamma_)
            else:  # all weights became zero, reinitialize (should not happen with solver fixes)
                self.gamma_ = np.ones(X_c.shape[0]) / X_c.shape[0]

        self.X_control_mean_ = np.mean(X_c, axis=0)
        # Calculate weighted mean of control covariates using the weights
        self.X_control_weighted_mean_ = X_c.T @ self.gamma_
        # Alternative: self.X_control_weighted_mean_ = self.solver_.y_hat_ (if X_for_solver was not sparse)

        self.mean_Y_treated_ = np.mean(Y_t)
        self.weighted_mean_Y_control_ = np.sum(self.gamma_ * Y_c)
        self.att_ = self.mean_Y_treated_ - self.weighted_mean_Y_control_

        return self

    def balance_table(self):
        """
        Returns a table of covariate means for treated and weighted control units,
        along with the difference.
        """
        self.baltab = {
            "Treated Mean": self.X_treated_mean_,
            "Control Mean": self.X_control_mean_,
            "Weighted Control Mean": self.X_control_weighted_mean_,
            "Pre-weighting Imbalance": self.X_treated_mean_ - self.X_control_mean_,
            "Post-weighting Imbalance": self.X_treated_mean_
            - self.X_control_weighted_mean_,
        }
        return self.baltab

    def summary(self, debug=False):
        if self.att_ is None:
            print("Estimator has not been fitted yet.")
            return

        print("--- Balancing Weights Estimator Summary ---")
        print(f"Estimated ATT: {self.att_:.4f}")
        print(f"Mean Y treated: {self.mean_Y_treated_:.4f}")
        print(f"Weighted Mean Y control: {self.weighted_mean_Y_control_:.4f}")
        print(f"Number of treated units: {self.n_c}")
        print(f"Number of control units: {self.n_t}")

        if debug:
            print("\nSolver Diagnostics:")
            print(f"  Solver Iterations: {self.solver_.n_iterations_}")
            print(f"  Solver Final MSE (for covariates): {self.solver_.mse_:.6e}")
            print(f"  Sum of control weights (gamma): {np.sum(self.gamma_):.4f}")
            if self.solver_.convergence_history_:
                print(
                    f"  Final convergence norm: {self.solver_.convergence_history_[-1]:.2e}"
                )
