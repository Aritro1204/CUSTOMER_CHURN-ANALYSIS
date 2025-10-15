# CUSTOMER_CHURN-ANALYSIS

# 🏦 Customer Churn Prediction: ANN Deep Learning Model

This repository houses a complete machine learning workflow to predict **customer churn** for a bank using an **Artificial Neural Network (ANN)**. From data preprocessing to a live, interactive web application, this project is ready for deployment\! 🚀

-----

## 📁 Project Contents

| File / Folder | 🏷️ Type | ✨ Description |
| :--- | :--- | :--- |
| `experiments.ipynb` | Jupyter Notebook | **Model Training & Preprocessing:** The core script for data preparation, ANN building, training, and saving all artifacts. |
| `prediction.ipynb` | Jupyter Notebook | **Prediction Demo:** Demonstrates how to load the saved components (`.h5`, `.pkl`) and make predictions on new data. |
| `app.py` | Python Script | **Streamlit Web App:** An interactive front-end for real-time churn predictions. |
| `model.h5` | Saved Model | The trained Keras/TensorFlow **ANN Model**. |
| `scaler.pkl` | Pickle File | The **StandardScaler** object used to scale numerical features. |
| `label_encoder_gender.pkl` | Pickle File | The **LabelEncoder** object for the `Gender` column. |
| `onehot_encoder_geo.pkl` | Pickle File | The **OneHotEncoder** object for the `Geography` column. |
| `Churn_Modelling.csv` | Dataset (Required) | **Crucial:** The raw bank customer data used to train the model. *(Must be added to the directory\!)* |

-----

## 🛠️ Setup & Installation

### 1\. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# 🐍 Create & Activate Environment
python -m venv venv

# For macOS/Linux
source venv/bin/activate

# For Windows
.\venv\Scripts\activate
```

### 2\. Install Dependencies

Install all required Python libraries using `pip`:

```bash
pip install pandas scikit-learn tensorflow keras streamlit
```

### 3\. Data Check

Ensure the `Churn_Modelling.csv` file is placed in the root directory of this project before running any notebooks. 🛑

-----

## ⚙️ Workflow & Usage

### 🧠 Step 1: Model Training (`experiments.ipynb`)

This step executes the entire pipeline, from raw data to saved production-ready files.

1.  **Data Preprocessing:** Drops irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
2.  **Feature Engineering:** Encodes `Gender` (LabelEncoder) and `Geography` (OneHotEncoder).
3.  **Scaling:** Scales numerical features using **StandardScaler**.
4.  **Training:** Trains the ANN model using the processed data.
5.  **Saving:** Saves the final model (`model.h5`) and the necessary preprocessing objects (`.pkl` files).

### 🖥️ Step 2: Launch the Web App (`app.py`)

Once the model and all pickle files are saved, you can launch the interactive prediction interface:

1.  Make sure your virtual environment is activated.

2.  Run the Streamlit command:

    ```bash
    streamlit run app.py
    ```

3.  The application will open in your browser, allowing you to input customer metrics and receive an instant prediction on their likelihood to churn\! 📉📈

-----

## 🎯 Model Architecture

The core of the prediction engine is a Deep Learning model built with Keras/TensorFlow:

| Layer | Neurons (Output Shape) | Activation | Notes |
| :--- | :--- | :--- | :--- |
| **Input** | (12 features) | N/A | Receives the scaled and encoded input vector. |
| **Hidden 1** | 64 | `ReLU` | |
| **Hidden 2** | 32 | `ReLU` | |
| **Output** | 1 | `Sigmoid` | Outputs the probability of the customer exiting the bank (Churn). |

| Training Parameters | Value |
| :--- | :--- |
| **Optimizer** | Adam (`learning_rate=0.01`) |
| **Loss** | Binary Cross-Entropy |
| **Regularization** | Early Stopping (`patience=10`) |
