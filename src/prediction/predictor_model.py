import pymc3 as pm
import numpy as np
import theano.tensor as tt
from sklearn.exceptions import NotFittedError
import joblib
import os
import pandas as pd

PREDICTOR_FILE_NAME = "bayesian_regressor.joblib"

class Regressor:
    """A wrapper class for Bayesian Regression using PyMC3.

    This class provides a consistent interface that can be used with other
    regressor models.

    Attributes:
        model_name (str): Name of the regressor model.
    """

    model_name = "Bayesian Regression"

    def __init__(self, alpha_mu: float = 0., alpha_sigma: float = 10.,
                 beta_mu: float = 0., beta_sigma: float = 10.,
                 init: str = 'auto'):
        """Construct a new Bayesian Regressor.

        Args:
            alpha_mu (float): Mean of the normal prior for the intercept.
            alpha_sigma (float): Standard deviation of the normal prior for the intercept.
            beta_mu (float): Mean of the normal priors for the regression coefficients.
            beta_sigma (float): Standard deviation of the normal priors for the regression coefficients.
            init (str): Initialization method for the sampler.
        """
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.beta_mu = beta_mu
        self.beta_sigma = beta_sigma
        self.init = init
        self.model = None
        self.trace = None
        self._is_trained = False

    def build_model(
            self, train_inputs: pd.DataFrame, train_targets: np.ndarray
        ) -> pm.Model:
        """Build a new Bayesian regression model."""
        model = pm.Model()
        with model:
            # Priors for regression coefficients
            alpha = pm.Normal(
                'alpha', mu=self.alpha_mu, sigma=self.alpha_sigma
            )
            beta = pm.Normal(
                'beta',
                mu=self.beta_mu,
                sigma=self.beta_sigma,
                shape=train_inputs.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=1)

            # Linear regression relationship
            mu = alpha + pm.math.dot(train_inputs, beta)

            # Likelihood
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=train_targets)

        return model


    def fit(self, train_inputs: pd.DataFrame, train_targets) -> None:
        """Fit the Bayesian regressor to the training data.

        Args:
            train_inputs (pd.DataFrame): Training input data.
            train_targets: Training target data, can be either pd.DataFrame or
                            np.ndarray.
        """
        # Ensure target is 1D
        if isinstance(train_targets, pd.DataFrame) and train_targets.shape[1] == 1:
            y = train_targets.values.ravel()
        else:
            y = train_targets

        self.model = self.build_model(train_inputs, y)

        with self.model:
            # Sample from the posterior
            self.trace = pm.sample(draws=2000, tune=1000, init=self.init, cores=2)

        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.
        
        Args:
            inputs (pd.DataFrame): Input data for prediction.
            
        Returns:
            np.ndarray: Predicted regression targets.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        
        alpha_posterior = self.trace['alpha'].mean()
        beta_posterior = self.trace['beta'].mean(axis=0)
        
        predictions = alpha_posterior + np.dot(inputs, beta_posterior)
        
        return predictions


    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the Bayesian regressor and return the mean squared error.

        Args:
            test_inputs (pd.DataFrame): Test input data.
            test_targets (pd.Series): Test target data.

        Returns:
            float: Mean squared error of the regressor.

        Raises:
            NotFittedError: If the model is not trained yet.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        predictions = self.predict(test_inputs)
        mse = ((predictions - test_targets) ** 2).mean()
        return mse

    def save(self, model_dir_path: str) -> None:
        """Save the Bayesian regressor to disk.

        Args:
            model_dir_path (str): Directory path to save the model.

        Raises:
            NotFittedError: If the model is not trained yet.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        with self.model:
            pm.save_trace(
                self.trace,
                os.path.join(model_dir_path, PREDICTOR_FILE_NAME),
                overwrite=True
            )

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the Bayesian regressor from disk.

        Args:
            model_dir_path (str): Directory path from where to load the model.

        Returns:
            Regressor: Loaded Bayesian regressor model.
        """
        model = cls()
        # Dummy data to build the model
        model.model = cls().build_model(pd.DataFrame(), np.array([]))  
        with model.model:
            model.trace = pm.load_trace(
                os.path.join(model_dir_path, PREDICTOR_FILE_NAME)
            )
        model._is_trained = True
        return model

    def __str__(self) -> str:
        """String representation of the Bayesian Regressor.

        Returns:
            str: Information about the Bayesian regressor.
        """
        return (
            f"Model name: {self.model_name} ("
            f"alpha_mu: {self.alpha_mu}, "
            f"alpha_sigma: {self.alpha_sigma}, "
            f"beta_mu: {self.beta_mu}, "
            f"beta_sigma: {self.beta_sigma}, "
            f"init: {self.init})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
