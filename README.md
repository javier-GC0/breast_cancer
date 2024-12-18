## How to run the project in a virtual environment

### 1. Create the virtual environment (use `env` for convenience):
```
python.exe -m venv env --prompt <name_environment>
```
### 2. Activate the virtual environment:
```
.\env\Scripts\activate
```
### 3. Clone the repository into the desired directory:
```
git clone https://github.com/javier-GC0/breast_cancer
```
### 4. Install the project dependencies:
```
python.exe -m pip install .\breast_cancer\requirements.txt
```
### 5. Register the virtual environment in Jupyter:
```
python -m ipykernel install --user --name=breast_cancer --display-name "Python (breast_cancer)"
```
### 6. Start Jupyter Notebook:
```
jupyter notebook
```
### 7. Open the project notebook:
Navigate to the notebook file (`breast_cancer.ipynb`) and select the `Python (breast_cancer)` kernel.



## Project Structure

### config
- **kaggle.json**: Configuration file required for downloading the dataset from Kaggle. If you want to generate this file, go to your Kaggle profile, then navigate to **Settings** > **Account** > **API** > **Create New Token**.

### data
- **data.csv**: Dataset containing information about breast cancer, downloaded from Kaggle.

### src
- **funs.py**: This file contains helper functions used in the Jupyter notebook for better code organization and reusability. These functions include model evaluation, plotting metrics, and cross-validation for different models, among others.

### breast_cancer.ipynb
- Jupyter notebook containing the following sections:
    - **0. Imports**: Importing the necessary Python libraries and modules for data manipulation, visualization, and model training.
    - **1. Dataset Acquisition**: Downloading and loading the breast cancer dataset, which is stored in `data.csv`, and preparing it for analysis. Includes loading the dataset from Kaggle using the `kaggle.json` authentication file.
    - **2. Exploratory Data Analysis (EDA)**: Data cleaning, feature exploration and visualization of the breast cancer dataset. This section includes tasks such as handling missing values, checking data types, exploring distributions of features, and visualizing the relationship between different features.
    - **3. Predictive models**: Various machine learning models are trained and evaluated, comparing their performance based on some metrics.

    In **Section 1** of the notebook, there is a step to generate the `kaggle.json` file to authenticate the Kaggle API and download the dataset.

### requirements.txt
- This file lists all Python packages and the corresponding versions required to run the project. It ensures that all dependencies are installed correctly for consistent execution across environments.
