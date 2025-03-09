# Wine Quality Prediction

## 1. Project Overview

### Project Name

Wine Quality Prediction

### Problem Statement

Wine quality assessment is crucial for both producers and consumers, as it directly impacts market value, customer satisfaction, and production efficiency. Traditional methods rely on human sensory evaluation, which is subjective, time-consuming, and inconsistent. This project aims to leverage machine learning techniques to predict wine quality based on physicochemical properties, providing a more objective, scalable, and data-driven approach to quality assessment. By implementing a structured ML pipeline with modular programming, this project focuses on automating the process, ensuring reproducibility, and integrating MLflow for experiment tracking and Flask for deployment.

### Data Processed

The project handles tabular data.

#### Schema (schema.yaml)

```yaml
COLUMNS:
  fixed acidity: float64
  volatile acidity: float64
  citric acid: float64
  residual sugar: float64
  chlorides: float64
  free sulfur dioxide: float64
  total sulfur dioxide: float64
  density: float64
  pH: float64
  sulphates: float64
  alcohol: float64
  quality: int64

TARGET_COLUMN:
  name: quality
```

### Type of Problem

This is a regression problem.

## 2. Project Structure

The folder structure is illustrated in the attached images.

### Main Files and Folders

- ``: Contains all the core scripts for data processing, training, and evaluation.
- ``: Stores raw and processed datasets.
- ``: Contains Jupyter notebooks for exploratory data analysis (EDA).
- ``: Holds configuration settings.
- ``: Defines the schema of the dataset.
- ``: Contains hyperparameter settings.
- ``: Executes different stages of the ML pipeline.

### `main.py` Functionality

The `main.py` file sequentially executes the following stages:

- Data Ingestion
- Data Validation
- Data Transformation
- Model Training
- Model Evaluation

Each stage is initiated using modular pipeline classes, and logs are maintained using `logger`.

### Configuration Files

#### `config.yaml`

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/winequality-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

data_validation:
  root_dir: artifacts/data_validation
  unzip_data_dir: artifacts/data_ingestion/winequality-red.csv
  STATUS_FILE: artifacts/data_validation/status.txt

data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/winequality-red.csv

model_trainer:
  root_dir: artifacts/model_trainer
  train_data_path: artifacts/data_transformation/train.csv
  test_data_path: artifacts/data_transformation/test.csv
  model_name: model.joblib

model_evaluation:
  root_dir: artifacts/model_evaluation
  test_data_path: artifacts/data_transformation/test.csv
  model_path: artifacts/model_trainer/model.joblib
  metric_file_name: artifacts/model_evaluation/metrics.json
```

#### `params.yaml`

```yaml
ElasticNet:
  alpha: 0.2
  l1_ratio: 0.1
```

## 3. Technical Details

### Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Dagshub
- Flask
- Flask-Cors
- Matplotlib
- TQDM
- Joblib
- PyYAML
- Python-Box
- Ensure

### ML Pipeline Workflow

1. **Data Ingestion**: Downloads and extracts the dataset.
2. **Data Validation**: Ensures schema consistency.
3. **Data Transformation**: Preprocesses data for training.
4. **Model Training**: Trains an ElasticNet model.
5. **Model Evaluation**: Evaluates model performance using RMSE, MAE, and R².

### Model Evaluation Metrics

```json
{
  "rmse": 0.722643197795955,
  "mae": 0.55180191525614,
  "r2": 0.2513072525867187
}
```

## 4. Installation & Setup

### System Requirements

- Python 3.8+
- OS: Windows/Linux/MacOS

### Installation Steps

```bash
# Clone the repository
git clone <repo-url>
cd wine-quality-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 5. Usage

### Running the Project

```bash
# Run the ML pipeline
python main.py

# Start Flask API
python app.py
```

### Expected Inputs and Outputs

- **Input**: CSV file containing wine quality features.
- **Output**: Predicted wine quality score.

## 6. Results & Model Performance

- The model performance is evaluated using RMSE, MAE, and R².
- Future improvements could involve hyperparameter tuning to optimize performance.

## 7. Contributions & Future Scope

### Future Improvements

- Enhance model performance through better feature engineering.
- Deploy the Flask API on a cloud platform.

### Contributing

- Fork the repository and submit pull requests for improvements.

## 8. License & Acknowledgments

### Dataset Source

- The dataset is sourced from [this repository](https://github.com/krishnaik06/datasets/raw/refs/heads/main/winequality-data.zip).

### License

- This project is licensed under the MIT License.
