import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from utils.data_preprocessing import DataPreprocessor
from utils.model_training import RangePredictionModel
import os

def main():
    print("=" * 60)
    print("EV Range Prediction - Model Training Pipeline")
    print("=" * 60)

    data_loader = DataLoader(data_dir='data')

    print("\n[1/5] Creating sample EV dataset...")
    df = data_loader.create_sample_ev_data()
    print(f"   Dataset created: {len(df)} samples")

    filepath = data_loader.save_data(df, 'ev_data.csv')
    print(f"   Saved to: {filepath}")

    print("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()

    df_cleaned = preprocessor.clean_data(df)
    print(f"   Data cleaned: {len(df_cleaned)} rows")

    categorical_columns = ['vehicle_model', 'manufacturer', 'terrain_type', 'driving_style']
    df_encoded = preprocessor.encode_categorical(df_cleaned, categorical_columns)

    print("\n[3/5] Preparing features...")
    feature_columns = ['battery_capacity', 'efficiency', 'weight', 'temperature',
                      'charging_cycles', 'vehicle_age', 'avg_speed',
                      'terrain_type', 'driving_style']

    X = df_encoded[feature_columns]
    y = df_encoded['range_km']

    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(
        pd.concat([X, y], axis=1),
        target_column='range_km',
        test_size=0.2,
        random_state=42
    )

    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    print("\n[4/5] Training XGBoost model...")
    model = RangePredictionModel(model_type='xgboost')
    model.train(X_train, y_train)
    print("   Model training completed!")

    print("\n[5/5] Evaluating model performance...")
    metrics, predictions = model.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.4f}")

    os.makedirs('models', exist_ok=True)
    model.save_model('models/range_prediction_model.pkl')
    print("\n   Model saved to: models/range_prediction_model.pkl")

    print("\n[Feature Importance]")
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        print(feature_importance.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
