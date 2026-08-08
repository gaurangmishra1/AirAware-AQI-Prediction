# AirAware

A Smart Air Quality Index (AQI) Prediction System developed during the **Infosys Springboard Internship 6.0**. The project uses machine learning, deep learning, and time-series forecasting techniques to analyze historical air-quality data and predict future AQI values.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* ARIMA
* Prophet
* LSTM
* Streamlit
* Matplotlib
* Google Colab
* VS Code

## Features

* AQI data preprocessing and cleaning
* Historical AQI analysis
* Time-series forecasting
* AQI prediction using multiple models
* ARIMA-based forecasting
* Prophet-based forecasting
* LSTM-based prediction
* Model performance evaluation
* Prediction visualization
* Interactive Streamlit dashboard

## Development Environment

The project was developed using two environments:

* **Google Colab** — used for data analysis, preprocessing, experimentation, model training, and evaluation.
* **VS Code** — used for organizing the project, application development, and running the Streamlit interface.

## Project Workflow

```text
AQI Dataset
    ↓
Data Collection
    ↓
Data Cleaning & Preprocessing
    ↓
Exploratory Data Analysis
    ↓
Time-Series Preparation
    ↓
Model Training
    ↓
ARIMA / Prophet / LSTM
    ↓
Model Evaluation
    ↓
AQI Prediction
    ↓
Streamlit Dashboard
```

## Prediction Models

### ARIMA

ARIMA (AutoRegressive Integrated Moving Average) is used for time-series forecasting based on historical AQI values.

```text
Historical AQI Data
        ↓
Time-Series Preparation
        ↓
ARIMA Model
        ↓
Future AQI Forecast
```

### Prophet

Prophet is used for time-series forecasting by modeling trends and temporal patterns present in the AQI data.

```text
Historical AQI Data
        ↓
Time-Series Preparation
        ↓
Prophet Model
        ↓
AQI Forecast
```

### LSTM

LSTM (Long Short-Term Memory) is a recurrent neural network architecture used to learn patterns from sequential AQI data and predict future AQI values.

```text
Historical AQI Data
        ↓
Data Preprocessing
        ↓
Normalization / Scaling
        ↓
Sequence Generation
        ↓
LSTM Model
        ↓
AQI Prediction
```

## Data Processing

The data-processing pipeline includes:

* Loading historical AQI data
* Handling missing values
* Data cleaning
* Date/time processing
* Exploratory data analysis
* Time-series preparation
* Feature preparation
* Data normalization/scaling
* Sequence generation for LSTM

## Model Evaluation

The implemented models were evaluated to understand their forecasting performance.

The project compares:

| Model   | Type                          |
| ------- | ----------------------------- |
| ARIMA   | Statistical time-series model |
| Prophet | Time-series forecasting model |
| LSTM    | Deep learning model           |

Prediction results can be analyzed using evaluation metrics and visual comparisons between actual and predicted AQI values.

## Streamlit Application

The Streamlit application provides an interactive interface for working with the AQI prediction system.

The application allows users to:

* View AQI data
* Analyze historical AQI trends
* Select prediction approaches
* View forecasting results
* Visualize AQI predictions

## Setup

Clone the repository:

```bash
git clone https://github.com/gaurangmishra1/AirAware-AQI-Prediction.git
```

Navigate to the project directory:

```bash
cd AirAware-AQI-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser and provide the AQI analysis and prediction interface.

## Google Colab

The data-science experimentation and model development were performed in **Google Colab**, including:

* Data preprocessing
* Exploratory data analysis
* Model experimentation
* ARIMA implementation
* Prophet implementation
* LSTM model development
* Model evaluation
* Prediction visualization

The trained/validated approach was then integrated into the project application.

## Testing / Evaluation

The following components were tested during development:

* AQI data preprocessing
* Missing-value handling
* Historical AQI analysis
* ARIMA forecasting
* Prophet forecasting
* LSTM prediction
* Model evaluation
* Prediction visualization
* Streamlit application functionality

## Assumptions

* Historical AQI data is used as the primary input for forecasting.
* AQI observations are treated as time-series data.
* Prediction quality depends on the quality and availability of historical data.
* Different forecasting models can perform differently depending on the characteristics of the dataset.
* LSTM requires properly prepared sequential data for training.

## Internship

This project was developed as part of the **Infosys Springboard Internship 6.0** in the **Data Science and Machine Learning** domain.

The project provided hands-on experience with:

* Python-based data analysis
* Data preprocessing
* Exploratory data analysis
* Time-series forecasting
* Machine learning
* Deep learning
* Model evaluation
* Data visualization
* Streamlit application development

## Demo Video

Demo Link: https://drive.google.com/file/d/1Wmqdo1N3TnrVNJPPOMOhKcrUFVl1QXS_/view?usp=drivesdk

## Author

**Gaurang Mishra**
