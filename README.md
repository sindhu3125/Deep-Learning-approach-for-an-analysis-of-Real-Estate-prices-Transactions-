
# 🏠 Real Estate Land Price and Construction Cost Prediction Using Deep Learning

## 📌 Project Overview

Real estate prices vary depending on location, land characteristics, property size, and other factors. Estimating land prices and construction costs accurately can help buyers, sellers, and property developers make informed decisions.

This project develops a **Real Estate Land Price and Construction Cost Prediction System** using Artificial Neural Networks (ANN). It uses two separate deep learning models to estimate land prices and construction costs based on relevant property features.

The project combines data preprocessing, neural network model development, model evaluation, and web application integration using Django to provide predictions through a user-friendly interface.

## 🎯 Objectives

- Analyze real estate data to identify factors affecting land prices and construction costs.
- Perform data preprocessing and exploratory data analysis (EDA).
- Encode categorical variables and scale numerical features.
- Develop separate Artificial Neural Network models for land price and construction cost prediction.
- Evaluate the performance of both models using regression metrics.
- Integrate trained models into a Django web application.
- Provide prediction history and location-based real estate agent details.

## 📊 Dataset

**Dataset:** Real Estate Land Price Dataset and Construction Cost Dataset

**Data Type:** Structured real estate data

**Prediction Targets:**
- Land Price — Price per square foot
- Construction Cost — Estimated construction price in lakhs

### Important Features

**Land Price Prediction**

| Feature | Description |
|---|---|
| State | State where the land is located |
| City / District | Location of the land |
| Plot Type | Type or category of land |
| Area | Land area |
| Price | Land price |
| Price per SqFt | Price per square foot |

**Construction Cost Prediction**

| Feature | Description |
|---|---|
| State | State where the property is located |
| City | Property location |
| BHK | Number of bedrooms, halls and kitchens |
| Size in SqFt | Property size |
| Year Built | Construction year |
| Age of Property | Property age |
| Price in Lakhs | Estimated property/construction price |

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Deep Learning:** TensorFlow, Keras
- **Data Preprocessing:** OneHotEncoder, StandardScaler
- **Model Evaluation:** Scikit-learn
- **Model Persistence:** Joblib, Keras model files
- **Web Application:** Django
- **Development Environment:** Jupyter Notebook, VS Code

## 🧠 Deep Learning Models

The project uses two separate Artificial Neural Network models.

### 1. Land Price Prediction Model

- Predicts land prices based on location and plot-related features.
- Uses categorical encoding and numerical feature scaling.
- Learns complex relationships between land characteristics and price.
- Achieved an **R² score of 91.31%**.

### 2. Construction Cost Prediction Model

- Predicts construction costs based on property-related features.
- Uses property size, BHK, location, and other available attributes.
- Uses a separate preprocessing pipeline and ANN model.
- Achieved an **R² score of 52.66%**.

### ANN Architecture

- Input Layer — receives preprocessed features.
- Hidden Layers — learn patterns and relationships between the input variables.
- ReLU Activation — introduces non-linearity.
- Output Layer — generates the predicted continuous value.
- Optimizer — Adam.
- Loss Function — Mean Squared Error (MSE).

## 🔬 Methodology

1. **Data Collection:** Gather land price and construction-related datasets.
2. **Data Preprocessing:** Clean the datasets, handle missing values where required, and prepare input features.
3. **Exploratory Data Analysis:** Analyze distributions, relationships, and location-based price patterns.
4. **Feature Engineering:** Select relevant features and prepare categorical and numerical variables.
5. **Categorical Encoding:** Apply OneHotEncoder to categorical features such as state, city, and plot type.
6. **Feature Scaling:** Apply StandardScaler to numerical features.
7. **Data Splitting:** Divide the data into training and testing sets using an 80:20 split.
8. **Model Development:** Build two separate ANN models using TensorFlow and Keras.
9. **Model Training:** Train the models using the training data and optimize weights using the Adam optimizer.
10. **Model Evaluation:** Evaluate prediction performance using regression metrics, including MAE, MSE, RMSE, and R².
11. **Model Persistence:** Save trained models and preprocessing objects for later use.
12. **Web Application Development:** Integrate the models into a Django application for user interaction and prediction.

## 📈 Model Performance

### Land Price Prediction

| Metric | Result |
|---|---:|
| R² Score | 91.31% |

### Construction Cost Prediction

| Metric | Result |
|---|---:|
| R² Score | 52.66% |

The land price prediction model achieved strong performance on the test dataset. The construction cost model was also evaluated, with its performance indicating opportunities for further improvement.

## 📏 Evaluation Metrics

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help assess prediction error and how well the models explain the variation in real estate prices.

## 🌐 Django Web Application Modules

- **User Authentication:** User registration and login.
- **Land Price Prediction:** Predict land price based on selected location and land details.
- **Construction Cost Prediction:** Estimate construction costs based on property details.
- **Prediction History:** Store and view previous prediction records.
- **Real Estate Agent Details:** Display agent contact details based on the selected location.

## 🚀 Project Structure

```text
real-estate-price-prediction/
│
├── dataset/
│   ├── land_dataset.csv
│   └── construction_dataset.csv
│
├── models/
│   ├── land_dl_model.keras
│   ├── construction_dl_model.keras
│   ├── land_preprocessor.joblib
│   └── construction_preprocessor.joblib
│
├── train_land_ann.py
├── train_construction_ann.py
│
├── manage.py
├── app/
│
├── templates/
├── static/
│
├── requirements.txt
└── README.md
```

*Note: The actual folder structure and filenames may differ based on the final GitHub repository.*

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd real-estate-price-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv myenv
```

### 4. Activate the Environment

**Windows:**

```bash
myenv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the Models

Train the land price model:

```bash
python train_land_ann.py
```

Train the construction cost model:

```bash
python train_construction_ann.py
```

### 7. Run the Django Application

```bash
python manage.py runserver
```

Open the application in your browser and use the available modules to predict land prices and construction costs.

## 🌟 Expected Outcome

The project provides a deep learning-based solution for estimating real estate land prices and construction costs using property and location-related information.

The system allows users to enter relevant property details, receive predicted prices, review their previous predictions, and access real estate agent contact details through a Django web interface.

## 🔮 Future Enhancements

- Improve construction cost prediction accuracy using additional relevant features.
- Incorporate more recent real estate datasets.
- Expand coverage to additional cities and regions.
- Add interactive geographical price visualization.
- Deploy the application to a cloud platform.
- Explore additional neural network architectures and optimization techniques.

## 👩‍💻 Author

**Sindhu Reddyvari**

- GitHub: https://github.com/sindhu3125
