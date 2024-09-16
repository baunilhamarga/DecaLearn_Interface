import argparse
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor
import os
import time
import warnings

random_state = 12227

def load_data(file_path):
    """
    Load dataset from a .npz file.

    Parameters:
    - file_path (str): Path to the .npz file containing the dataset.

    Returns:
    - X_train (numpy.ndarray): Training features.
    - y_train (numpy.ndarray): Training labels.
    - X_test (numpy.ndarray): Test features.
    - y_test (numpy.ndarray): Test labels.
    """
    tmp = np.load(file_path, allow_pickle=True)
    X_train = tmp['X_train']
    y_train = tmp['y_train']
    X_test = tmp['X_test']
    y_test = tmp['y_test']
    return X_train, y_train, X_test, y_test

# LightGBM Classification Model
class LGBMClassificationModel:
    """
    A LightGBM model for classification tasks.

    Attributes:
    - model (LGBMClassifier): The LightGBM classifier.
    - scaler (StandardScaler): Scaler to normalize features.
    """
    
    def __init__(self):
        """Initialize the LightGBM classification model."""
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        """
        Train the LightGBM classification model.

        Parameters:
        - X_train (numpy.ndarray): Training features.
        - y_train (numpy.ndarray): Training labels.
        """
        hiperparameters = {'verbosity': -1, 'random_state': random_state}
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LGBMClassifier(**hiperparameters).fit(X_train_scaled, y_train)

    def predict(self, X_test):
        """
        Predict labels using the trained model.

        Parameters:
        - X_test (numpy.ndarray): Test features.

        Returns:
        - y_pred (numpy.ndarray): Predicted labels.
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

# LightGBM Regression Model
class LGBMRegressionModel:
    """
    A LightGBM model for regression tasks.

    Attributes:
    - model (LGBMRegressor): The LightGBM regressor.
    - scaler (StandardScaler): Scaler to normalize features.
    """
    
    def __init__(self):
        """Initialize the LightGBM regression model."""
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        """
        Train the LightGBM regression model.

        Parameters:
        - X_train (numpy.ndarray): Training features.
        - y_train (numpy.ndarray): Training labels (continuous values).
        """
        hiperparameters = {'verbosity': -1, 'random_state': random_state}
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LGBMRegressor(**hiperparameters).fit(X_train_scaled, y_train)

    def predict(self, X_test):
        """
        Predict values using the trained model.

        Parameters:
        - X_test (numpy.ndarray): Test features.

        Returns:
        - y_pred (numpy.ndarray): Predicted values (continuous).
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

# Model Runner class
class ModelRunner:
    """
    A class to run models (classification or regression) on a given dataset.

    Attributes:
    - model: A model object (either LGBMClassificationModel or LGBMRegressionModel).
    - task (str): Type of task ('classification' or 'regression').
    """
    
    def __init__(self, model, task='classification'):
        """
        Initialize the model runner.

        Parameters:
        - model: An instance of either LGBMClassificationModel or LGBMRegressionModel.
        - task (str): Type of task ('classification' or 'regression'). Default is 'classification'.
        """
        self.model = model
        self.task = task

    def run_model(self, file_path, X_train, y_train, X_test, y_test, timeout=True, timeout_seconds=60, verbose=False):
        """
        Run the model on the given dataset, measure performance, and print the results.

        Parameters:
        - file_path (str): Path to the dataset file.
        - X_train (numpy.ndarray): Training features.
        - y_train (numpy.ndarray): Training labels.
        - X_test (numpy.ndarray): Test features.
        - y_test (numpy.ndarray): Test labels (classification) or values (regression).
        - timeout (bool): Whether to enforce a time limit (default is True).
        - timeout_seconds (int): Maximum allowed time for the model to run (default is 60 seconds).
        - verbose (bool): If True, print detailed information (default is False).
        """
        file_name = os.path.basename(file_path)
        
        # Suppress model warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            start_time = time.time()  # Record start time

            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if self.task == 'classification':
                acc_test = accuracy_score(y_test, y_pred)
                print(f'Accuracy on {file_name}: {acc_test}')
            elif self.task == 'regression':
                mse_test = mean_squared_error(y_test, y_pred)
                print(f'Mean Squared Error on {file_name}: {mse_test}')

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time

            print(f'Time elapsed for {file_name}: {elapsed_time}s\n\n')

def get_npz_files(directory):
    """
    Fetch all .npz files in the given directory.

    Parameters:
    - directory (str): Path to the directory to search for .npz files.

    Returns:
    - List[str]: A list of full paths to .npz files in the directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LGBM models on datasets.')
    parser.add_argument('--dataset_path', type=str, default='all',
                        help='Path to a specific dataset (default: all datasets)')
    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'all'], default='all',
                        help='Task type: classification or regression (default: classification and regression)')
    args = parser.parse_args()

    if args.task == 'classification' or args.task == 'all':
        if args.dataset_path == 'all':
            # Get all .npz files for classification
            file_paths = get_npz_files('Data/Classification')
        else:
            file_paths = [args.dataset_path]

        # Running classification tasks
        print("Running classification tasks:")
        lgbm_classification_model = LGBMClassificationModel()
        runner_classification = ModelRunner(lgbm_classification_model, task='classification')

        for file_path in file_paths:
            X_train, y_train, X_test, y_test = load_data(file_path)
            runner_classification.run_model(file_path, X_train, y_train, X_test, y_test)

    if args.task == 'regression' or args.task == 'all':
        if args.dataset_path == 'all':
            # Get all .npz files for regression
            file_paths = get_npz_files('Data/Regression')
        else:
            file_paths = [args.dataset_path]

        # Running regression tasks
        print("Running regression tasks:")
        lgbm_regression_model = LGBMRegressionModel()
        runner_regression = ModelRunner(lgbm_regression_model, task='regression')

        for file_path in file_paths:
            X_train, y_train, X_test, y_test = load_data(file_path)
            runner_regression.run_model(file_path, X_train, y_train, X_test, y_test)
