# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Generate sample EV dataset
- Preprocess the data
- Train the XGBoost model
- Save the model to `models/range_prediction_model.pkl`

### 4. Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## Project Structure

```
Range-Prediction-using-AI/
├── data/                          # Generated datasets
│   └── ev_data.csv
├── models/                        # Trained models
│   └── range_prediction_model.pkl
├── utils/                         # Utility modules
│   ├── data_loader.py            # Data loading and generation
│   ├── data_preprocessing.py     # Data preprocessing functions
│   └── model_training.py         # ML model training
├── train_model.py                # Model training script
├── streamlit_app.py              # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Features

### Range Predictor
- Input vehicle and environmental parameters
- Get real-time range predictions
- View range breakdown by factors

### Data Insights
- Explore dataset distributions
- Analyze feature correlations
- Download data for external analysis

### Model Performance
- View feature importance
- Understand model metrics
- Analyze prediction accuracy

### AI Assistant
- Ask questions about EV range
- Get explanations about the model
- Learn about factors affecting range

## Troubleshooting

### Module Not Found Error
If you see `ModuleNotFoundError`, ensure you've activated the virtual environment and installed dependencies:
```bash
pip install -r requirements.txt
```

### Model Not Found
If the app shows "Model not found", run the training script first:
```bash
python train_model.py
```

### Port Already in Use
If port 8501 is in use, specify a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```
