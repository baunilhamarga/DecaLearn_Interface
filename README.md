# DecaLearn Interface: Simple Dataset Scoring Interface

This repository provides a simple interface to score multiple datasets using a given model in either classification or regression tasks. The default model is **LightGBM**, and the default metrics are **accuracy (ACC)** for classification and **mean squared error (MSE)** for regression.

## Features

- **LightGBM** is used by default for both classification and regression.
- Automatically processes all datasets in a specified directory or a single dataset.
- Tracks performance metrics: ACC for classification and MSE for regression.
- Easily switch between classification and regression tasks.

## Step-by-Step Guide

### Step 1: Download the `Data` Folder
Before running the script, download the `Data` folder containing the datasets from [this link](https://cloud.turingusp.com.br/index.php/s/FyDTtaRc9NkMQyS) and place it in the root directory of the repository.

The folder should have two subdirectories:
- `Data/Classification`: For classification datasets.
- `Data/Regression`: For regression datasets.

Each dataset should be a `.npz` file containing the following keys:
- `X_train`: Training features.
- `y_train`: Training labels (for classification) or values (for regression).
- `X_test`: Test features.
- `y_test`: Test labels (for classification) or values (for regression).

### Step 2: Install Dependencies

Before running the script, ensure you have the required Python packages installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### Step 3: Running the Script

You can now run the script to score multiple datasets. The script accepts the following command-line arguments to control its behavior.

#### Available Commands

- `--dataset_path`: Path to a specific dataset (in `.npz` format). If set to "all" (default), the script processes all datasets in the `Data/Classification` or `Data/Regression` folders, depending on the task.
- `--task`: Type of task to perform. You can choose either:
  - `classification`
  - `regression`
  - `all` (default): Runs both classification and regression tasks on the corresponding datasets.

#### Example Commands

1. **Run Classification and Regression on All Datasets** (default behavior):
    ```bash
    python main.py
    ```

    This will automatically load all datasets from the `Data/Classification` folder and run the classification task using LightGBM.

2. **Run Regression on All Datasets**:
    ```bash
    python main.py --task regression
    ```

    This will load all datasets from the `Data/Regression` folder and run the regression task using LightGBM.

3. **Run Classification on a Specific Dataset**:
    ```bash
    python main.py --dataset_path Data/Classification/sample_dataset.npz --task classification
    ```

    This will run classification on the `sample_dataset.npz` file using LightGBM.

4. **Run Regression on a Specific Dataset**:
    ```bash
    python main.py --dataset_path Data/Regression/sample_dataset.npz --task regression
    ```

    This will run regression on the `sample_dataset.npz` file using LightGBM.

### Output
After running the script, you will see output in the terminal indicating:
- The accuracy for classification tasks (`ACC`).
- The mean squared error for regression tasks (`MSE`).
- The time taken to process each dataset.

Example output:
```
Running classification tasks:
Accuracy on dataset1.npz: 0.85
Time elapsed for dataset1.npz: 3.2s

Running regression tasks:
Mean Squared Error on dataset2.npz: 0.025
Time elapsed for dataset2.npz: 4.8s
```

### Step 4: Customize the Script
If you want to use a different model or metric, you can modify the script by replacing the `LGBMClassifier` or `LGBMRegressor` with your preferred model in the `LGBMClassificationModel` or `LGBMRegressionModel` classes, respectively.

---

## Conclusion

This simple interface allows users to quickly evaluate the performance of a model on multiple datasets without needing deep programming expertise. By following the steps provided, you can easily run classification or regression tasks with LightGBM and analyze your datasets with minimal effort.

If you have any questions or issues, feel free to open a ticket in the repository.
