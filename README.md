# Range Prediction Using AI

## Project Overview

An intelligent end-to-end system for predicting the driving range of electric vehicles (EVs) using machine learning and AI techniques. This project provides accurate range estimates based on battery specifications, environmental conditions, vehicle characteristics, and driving patterns.

## Features

- **ML-Powered Predictions**: XGBoost-based model for accurate range estimation
- **Interactive Web Interface**: Built with Streamlit for real-time predictions
- **Data Visualization**: Comprehensive insights into factors affecting EV range
- **AI Chat Assistant**: Interactive help system for understanding predictions
- **Multi-Factor Analysis**: Considers battery specs, temperature, terrain, driving style, and more

## Technology Stack

- **Machine Learning**: scikit-learn, XGBoost, pandas, numpy
- **Frontend**: Streamlit
- **Visualization**: Plotly, matplotlib, seaborn
- **Language**: Python 3.8+

## Project Structure

```
Range-Prediction-using-AI/
├── data/                          # Generated datasets
│   └── ev_data.csv               # Sample EV data
├── models/                        # Trained ML models
│   └── range_prediction_model.pkl
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_loader.py            # Data generation and loading
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   └── model_training.py         # ML model training functions
├── train_model.py                # Model training script
├── streamlit_app.py              # Web application
├── requirements.txt              # Python dependencies
├── setup_instructions.md         # Detailed setup guide
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Range-Prediction-using-AI.git
cd Range-Prediction-using-AI
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python train_model.py
```

This will generate sample data, train the XGBoost model, and save it to the `models/` directory.

5. **Run the application**
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### 1. Range Predictor

Navigate to the "Range Predictor" page to:
- Input vehicle specifications (battery capacity, weight, age)
- Set environmental conditions (temperature, terrain)
- Choose driving parameters (speed, driving style)
- Get instant range predictions with visual breakdowns

### 2. Data Insights

Explore comprehensive data analysis:
- Distribution charts for key metrics
- Correlation analysis between features
- Interactive visualizations
- Raw data export capability

### 3. Model Performance

Understand the ML model:
- Feature importance rankings
- Performance metrics
- Model architecture details

### 4. AI Assistant

Interactive chatbot to answer questions about:
- How the prediction model works
- Factors affecting EV range
- Tips for maximizing range
- Dataset and feature explanations

## Key Features Explained

### Input Parameters

**Battery & Vehicle:**
- Battery Capacity (40-100 kWh)
- Efficiency (3.5-6.0 km/kWh)
- Vehicle Weight (1500-2500 kg)
- Vehicle Age (0-10 years)
- Charging Cycles (0-1000)

**Environmental:**
- Temperature (-10 to 40°C)
- Terrain Type (flat, mixed, hilly)

**Driving:**
- Average Speed (30-120 km/h)
- Driving Style (eco, normal, sport)

### Prediction Model

The system uses **XGBoost Regressor** with:
- 9 input features
- 100 estimators
- Optimized hyperparameters
- Cross-validated performance

### Data Generation

Sample dataset includes:
- 500 synthetic EV records
- Multiple manufacturers and models
- Realistic operating conditions
- Calculated range based on physics and efficiency factors

## Model Performance

The trained model achieves:
- High R² score indicating strong predictive power
- Low RMSE for accurate predictions
- Robust performance across different conditions
- Feature importance insights for interpretability

## Datasets Reference

This project is designed to work with:
- **Electric Vehicle Specifications Dataset 2025**: Real-world EV specs
- **EVIoT Predictive Maintenance Dataset**: Battery and maintenance data

For demonstration purposes, the system generates synthetic data that mimics real-world patterns.

## Code Structure

### utils/data_loader.py
- Generates sample EV datasets
- Handles data loading and saving
- Creates realistic EV operational data

### utils/data_preprocessing.py
- Data cleaning and validation
- Missing value handling
- Feature encoding
- Data normalization
- Feature engineering

### utils/model_training.py
- ML model initialization
- Training and evaluation
- Performance metrics calculation
- Feature importance analysis
- Model persistence

### train_model.py
- End-to-end training pipeline
- Data generation and preprocessing
- Model training and evaluation
- Results reporting

### streamlit_app.py
- Web application interface
- Interactive prediction tool
- Data visualization dashboards
- AI chat assistant
- Multi-page navigation

## Deployment

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Streamlit Community Cloud

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy the `streamlit_app.py` file

### Custom Port
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Troubleshooting

### Model Not Found Error
**Problem**: "Model not found" message in the app

**Solution**: Run the training script first
```bash
python train_model.py
```

### Module Import Errors
**Problem**: `ModuleNotFoundError` when running scripts

**Solution**: Ensure virtual environment is activated and dependencies installed
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Port Already in Use
**Problem**: Port 8501 is already in use

**Solution**: Use a different port
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- Integration with real-world EV datasets
- Historical trip data analysis
- Route optimization suggestions
- Battery degradation predictions
- Weather API integration
- Mobile responsive design improvements
- User authentication and saved predictions
- Advanced visualization options

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by real-world EV range prediction challenges
- Built with modern ML and web technologies
- Designed for educational and practical applications

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with Python, Machine Learning, and Streamlit**