# Housing-Price
# Housing Price Prediction with Regression

## Project Description

This project aims to predict housing prices using regression techniques. It utilizes a dataset containing various housing characteristics to train linear regression and Random Forest models. The goal is to evaluate and compare the performance of these models in predicting housing prices.

## Project Structure

- `housing-price-prediction-regression.ipynb`: Jupyter Notebook containing code for data loading, preprocessing, feature engineering, model creation and evaluation (both linear regression and Random Forest), and cross-validation.

- `linear_regression_model.joblib`: Saved linear regression model after being trained with the data.

- `linear_regression_scaling_model.joblib`: Saved linear regression model after being trained with scaled data.

- `dataset/`: Folder containing the data file.

  - `housing.csv`: Dataset with housing features and prices.

- `.ipynb_checkpoints/`: Automatically generated folder by Jupyter Notebook for saving notebook checkpoints. Not necessary for the project’s use.

## Requirements

To run the Jupyter Notebook and the models, you need to have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `scipy`

You can install these dependencies using `pip`. For example:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib scipy
