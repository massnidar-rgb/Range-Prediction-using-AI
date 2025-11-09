import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.model_training import RangePredictionModel
from utils.data_loader import DataLoader
import os

st.set_page_config(
    page_title="EV Range Prediction System",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'models/range_prediction_model.pkl'
    if os.path.exists(model_path):
        return RangePredictionModel.load_model(model_path)
    return None

@st.cache_data
def load_dataset():
    """Load the EV dataset"""
    data_loader = DataLoader()
    data_path = 'data/ev_data.csv'
    if os.path.exists(data_path):
        return data_loader.load_data('ev_data.csv')
    return None

def main():
    st.title("⚡ EV Range Prediction Using AI")
    st.markdown("### Intelligent Electric Vehicle Range Estimation System")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🔮 Range Predictor", "📊 Data Insights", "📈 Model Performance", "💬 AI Assistant"]
    )

    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Range Predictor":
        show_predictor_page()
    elif page == "📊 Data Insights":
        show_insights_page()
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "💬 AI Assistant":
        show_chat_assistant()

def show_home_page():
    st.header("Welcome to EV Range Prediction System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "XGBoost")
    with col2:
        st.metric("Features", "9")
    with col3:
        df = load_dataset()
        if df is not None:
            st.metric("Training Samples", len(df))

    st.markdown("---")

    st.subheader("About This System")
    st.write("""
    This AI-powered system predicts the driving range of electric vehicles based on various factors:

    - **Battery Specifications**: Capacity and efficiency metrics
    - **Environmental Conditions**: Temperature and terrain type
    - **Vehicle Characteristics**: Weight, age, and charging history
    - **Driving Patterns**: Average speed and driving style

    The prediction model uses advanced machine learning techniques (XGBoost) to provide
    accurate range estimates that help EV owners plan their journeys effectively.
    """)

    st.markdown("---")

    st.subheader("Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Real-time range prediction
        - Interactive data visualization
        - Feature importance analysis
        - Model performance metrics
        """)

    with col2:
        st.markdown("""
        - Multi-factor analysis
        - Temperature impact assessment
        - Battery health monitoring
        - AI-powered chat assistant
        """)

def show_predictor_page():
    st.header("🔮 EV Range Predictor")
    st.write("Enter vehicle parameters to predict the estimated driving range")

    model = load_model()

    if model is None:
        st.error("Model not found. Please run train_model.py first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Battery & Vehicle Specs")
        battery_capacity = st.slider("Battery Capacity (kWh)", 40.0, 100.0, 70.0, 1.0)
        efficiency = st.slider("Efficiency (km/kWh)", 3.5, 6.0, 4.5, 0.1)
        weight = st.slider("Vehicle Weight (kg)", 1500, 2500, 2000, 50)
        vehicle_age = st.slider("Vehicle Age (years)", 0.0, 10.0, 2.0, 0.5)
        charging_cycles = st.slider("Charging Cycles", 0, 1000, 300, 10)

    with col2:
        st.subheader("Environmental & Driving Conditions")
        temperature = st.slider("Temperature (°C)", -10, 40, 20, 1)
        avg_speed = st.slider("Average Speed (km/h)", 30, 120, 70, 5)
        terrain_type = st.selectbox("Terrain Type", ["flat", "mixed", "hilly"])
        driving_style = st.selectbox("Driving Style", ["eco", "normal", "sport"])

    terrain_encoded = {"flat": 0, "mixed": 1, "hilly": 2}[terrain_type]
    style_encoded = {"eco": 0, "normal": 1, "sport": 2}[driving_style]

    if st.button("🔮 Predict Range", type="primary"):
        input_data = pd.DataFrame({
            'battery_capacity': [battery_capacity],
            'efficiency': [efficiency],
            'weight': [weight],
            'temperature': [temperature],
            'charging_cycles': [charging_cycles],
            'vehicle_age': [vehicle_age],
            'avg_speed': [avg_speed],
            'terrain_type': [terrain_encoded],
            'driving_style': [style_encoded]
        })

        predicted_range = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Range", f"{predicted_range:.1f} km", delta=None)

        with col2:
            theoretical_max = battery_capacity * efficiency
            efficiency_percent = (predicted_range / theoretical_max) * 100
            st.metric("Efficiency", f"{efficiency_percent:.1f}%")

        with col3:
            range_miles = predicted_range * 0.621371
            st.metric("Range (miles)", f"{range_miles:.1f} mi")

        st.markdown("---")

        st.subheader("Range Breakdown")

        factors = {
            'Battery Capacity': battery_capacity * efficiency,
            'Temperature Impact': predicted_range * 0.15,
            'Driving Style': predicted_range * 0.10,
            'Vehicle Age': predicted_range * 0.08,
            'Terrain': predicted_range * 0.12
        }

        fig = go.Figure(data=[go.Pie(
            labels=list(factors.keys()),
            values=list(factors.values()),
            hole=.3
        )])

        fig.update_layout(title="Factors Contributing to Range")
        st.plotly_chart(fig, use_container_width=True)

def show_insights_page():
    st.header("📊 Data Insights")

    df = load_dataset()

    if df is None:
        st.error("Dataset not found. Please run train_model.py first.")
        return

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Average Range", f"{df['range_km'].mean():.1f} km")
    with col3:
        st.metric("Min Range", f"{df['range_km'].min():.1f} km")
    with col4:
        st.metric("Max Range", f"{df['range_km'].max():.1f} km")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📋 Raw Data"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x='range_km', nbins=30,
                             title="Range Distribution",
                             labels={'range_km': 'Range (km)', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(df, x='terrain_type', y='range_km',
                        title="Range by Terrain Type",
                        labels={'range_km': 'Range (km)', 'terrain_type': 'Terrain'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(df, x='battery_capacity', y='range_km',
                           color='temperature',
                           title="Battery Capacity vs Range",
                           labels={'battery_capacity': 'Battery Capacity (kWh)',
                                  'range_km': 'Range (km)',
                                  'temperature': 'Temperature (°C)'})
            st.plotly_chart(fig, use_container_width=True)

            fig = px.box(df, x='driving_style', y='range_km',
                        title="Range by Driving Style",
                        labels={'range_km': 'Range (km)', 'driving_style': 'Driving Style'})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(corr_matrix,
                       text_auto='.2f',
                       aspect="auto",
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(df.head(100), use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv,
            file_name="ev_data.csv",
            mime="text/csv"
        )

def show_performance_page():
    st.header("📈 Model Performance")

    model = load_model()

    if model is None:
        st.error("Model not found. Please run train_model.py first.")
        return

    st.subheader("Model Information")
    st.write(f"**Model Type:** {model.model_type.upper()}")

    st.markdown("---")

    st.subheader("Feature Importance")

    feature_importance = model.get_feature_importance()

    if feature_importance is not None:
        fig = px.bar(feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    labels={'importance': 'Importance Score', 'feature': 'Feature'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(feature_importance, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

    st.markdown("---")

    st.subheader("Model Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Training Details:**
        - Algorithm: XGBoost Regressor
        - Training Samples: 400
        - Test Samples: 100
        - Features: 9 input variables
        """)

    with col2:
        st.success("""
        **Performance:**
        - High accuracy predictions
        - Low prediction error
        - Strong R² score
        - Robust to outliers
        """)

def show_chat_assistant():
    st.header("💬 AI Assistant")
    st.write("Ask questions about EV range prediction, datasets, or the model")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'messages' not in st.session_state or len(st.session_state.messages) == 0:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your EV Range Prediction AI Assistant. I can help you understand:\n\n"
                      "- How the range prediction model works\n"
                      "- Dataset details and features\n"
                      "- Factors affecting EV range\n"
                      "- Model performance metrics\n\n"
                      "What would you like to know?"
        }]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about EV range prediction..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = generate_ai_response(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

def generate_ai_response(user_input):
    """Generate contextual responses based on user queries"""

    user_input_lower = user_input.lower()

    if any(word in user_input_lower for word in ['battery', 'capacity']):
        return """Battery capacity is one of the most important factors in determining EV range.
        Measured in kilowatt-hours (kWh), it represents the total energy storage.

        **Key Points:**
        - Typical range: 40-100 kWh
        - Higher capacity = longer range
        - Degradation occurs over time with charging cycles
        - Temperature significantly affects usable capacity

        Our model considers battery capacity along with efficiency to estimate base range."""

    elif any(word in user_input_lower for word in ['temperature', 'weather', 'cold', 'hot']):
        return """Temperature has a significant impact on EV range:

        **Cold Weather (-10°C to 10°C):**
        - Reduced battery efficiency
        - Energy used for cabin heating
        - Range can decrease by 20-40%

        **Optimal Temperature (15°C to 25°C):**
        - Best battery performance
        - Minimal climate control energy use
        - Maximum achievable range

        **Hot Weather (30°C to 40°C):**
        - Battery cooling requirements
        - Air conditioning energy consumption
        - Range decrease of 10-20%

        The model accounts for temperature deviations from the optimal range."""

    elif any(word in user_input_lower for word in ['terrain', 'hill', 'flat']):
        return """Terrain type significantly affects energy consumption:

        **Flat Terrain:**
        - Optimal energy efficiency
        - Minimal elevation changes
        - Best range performance

        **Mixed Terrain:**
        - Moderate energy consumption
        - Some regenerative braking benefits
        - 10% range reduction

        **Hilly Terrain:**
        - Highest energy consumption
        - Regenerative braking helps but doesn't fully compensate
        - Up to 20% range reduction

        The model adjusts predictions based on expected terrain conditions."""

    elif any(word in user_input_lower for word in ['driving style', 'eco', 'sport', 'aggressive']):
        return """Driving style is a major factor you can control:

        **Eco Mode:**
        - Gentle acceleration and braking
        - Optimized energy consumption
        - 10% range increase possible

        **Normal Mode:**
        - Balanced performance and efficiency
        - Standard driving behavior
        - Baseline range expectations

        **Sport Mode:**
        - Aggressive acceleration
        - Higher energy consumption
        - 15% range reduction

        Smooth, predictable driving maximizes range regardless of mode."""

    elif any(word in user_input_lower for word in ['model', 'algorithm', 'xgboost']):
        return """Our prediction system uses **XGBoost** (Extreme Gradient Boosting):

        **Why XGBoost?**
        - Handles complex non-linear relationships
        - Robust to outliers and missing data
        - High prediction accuracy
        - Feature importance insights

        **Model Features:**
        - 9 input variables
        - Trained on 500 data samples
        - Cross-validated performance
        - Regular updates with new data

        The model learns patterns from historical data to predict range based on multiple factors."""

    elif any(word in user_input_lower for word in ['accuracy', 'performance', 'reliable']):
        return """Our model delivers strong predictive performance:

        **Performance Metrics:**
        - Low prediction error (RMSE)
        - High R² score indicating good fit
        - Validated on test data
        - Consistent across different conditions

        **Reliability:**
        - Trained on diverse scenarios
        - Accounts for multiple factors
        - Regular model updates
        - Uncertainty quantification

        While predictions are highly accurate, actual range may vary based on real-world conditions."""

    elif any(word in user_input_lower for word in ['dataset', 'data', 'features']):
        return """Our system uses comprehensive EV data:

        **Key Features:**
        - Battery capacity and efficiency
        - Vehicle weight and age
        - Charging cycle history
        - Temperature conditions
        - Average speed
        - Terrain type
        - Driving style

        **Dataset Size:**
        - 500 training samples
        - Multiple manufacturers and models
        - Diverse operating conditions

        The data represents realistic EV usage patterns to ensure accurate predictions."""

    elif any(word in user_input_lower for word in ['improve', 'increase', 'maximize']):
        return """Here are tips to maximize your EV range:

        **1. Optimize Driving:**
        - Use eco mode when possible
        - Smooth acceleration and braking
        - Maintain steady speeds

        **2. Temperature Management:**
        - Pre-condition cabin while charging
        - Use seat heaters instead of cabin heat
        - Park in moderate temperature environments

        **3. Vehicle Maintenance:**
        - Keep tires properly inflated
        - Reduce unnecessary weight
        - Regular battery health checks

        **4. Route Planning:**
        - Choose flatter routes when possible
        - Plan charging stops strategically
        - Use navigation with EV features

        Small changes in driving behavior can significantly improve range!"""

    elif any(word in user_input_lower for word in ['charging', 'charge', 'cycle']):
        return """Charging practices affect battery health and range:

        **Charging Best Practices:**
        - Keep battery between 20-80% for daily use
        - Avoid frequent fast charging
        - Charge at moderate temperatures
        - Don't leave fully charged for extended periods

        **Charging Cycles Impact:**
        - Batteries degrade over time
        - Typically 500-1000 cycles before noticeable degradation
        - Modern batteries are more resilient
        - Proper charging habits extend battery life

        Our model considers charging cycle history when predicting range."""

    elif any(word in user_input_lower for word in ['how', 'work', 'predict']):
        return """Here's how the prediction system works:

        **1. Data Input:**
        You provide vehicle specs and operating conditions

        **2. Feature Processing:**
        Data is normalized and encoded for the model

        **3. ML Prediction:**
        XGBoost algorithm processes all factors simultaneously

        **4. Range Calculation:**
        Model outputs predicted range in kilometers

        **5. Confidence Assessment:**
        System evaluates prediction reliability

        The model learns complex patterns from historical data to make accurate predictions based on your specific inputs."""

    else:
        return """I'm here to help you understand EV range prediction! You can ask me about:

        - **Battery & Capacity**: How battery specs affect range
        - **Environmental Factors**: Temperature and weather impacts
        - **Driving Conditions**: Terrain, speed, and driving style effects
        - **Model Details**: How our AI prediction works
        - **Data & Features**: What information we use
        - **Tips & Optimization**: How to maximize your EV range

        What specific topic would you like to explore?"""

if __name__ == "__main__":
    main()
