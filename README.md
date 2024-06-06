# Heart Disease Prediction Project

## Overview

This project aims to predict the presence of heart disease using the Cleveland heart disease dataset. By leveraging various machine learning algorithms, we developed models to assist healthcare professionals in making informed diagnostic decisions. The project evaluates the performance of multiple algorithms and explores potential enhancements for future work.

## Key Findings

- **XGBoost**: Achieved the highest test accuracy of 86.96%, demonstrating strong predictive power and generalizability.
- **Random Forest**: Test accuracy of 83.70%, indicating robust performance but slight overfitting.
- **SVM**: Test accuracy of 82.97%, effectively finding optimal hyperplanes for classification.
- **KNN**: Test accuracy of 84.42%, capturing local patterns effectively.
- **Naive Bayes**: Test accuracy of 83.33%, providing a good balance between bias and variance.
- **Logistic Regression**: Test accuracy of 84.42%, offering straightforward and efficient predictions.

## Future Work

Potential enhancements for the predictive models include:
- **Ensemble Learning**: Implementing advanced methods like stacking to combine multiple models for improved performance.
- **Deep Learning**: Exploring CNNs and RNNs to capture complex patterns and temporal relationships.
- **Feature Engineering**: Developing sophisticated techniques to capture non-linear relationships.
- **Model Interpretability**: Utilizing SHAP values to align model predictions with medical expertise.
- **Class Imbalance Handling**: Applying methods such as SMOTE or cost-sensitive learning to improve performance on minority classes.
- **Integration of Additional Data**: Incorporating genetic, lifestyle, and longitudinal health data for more comprehensive models.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate   # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure the Cleveland heart disease dataset is available in the project directory.
2. Run the Jupyter notebooks to preprocess the data, train models, and evaluate performance:
    ```sh
    jupyter notebook
    ```
3. Follow the notebooks for detailed steps on data preprocessing, model training, and evaluation.

## Results

The project results are summarized in the final report and detailed in the Jupyter notebooks. Model performance metrics and visualizations are provided for each algorithm.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

