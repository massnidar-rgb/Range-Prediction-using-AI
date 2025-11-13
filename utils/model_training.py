import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pickle
import os

class RangePredictionModel:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None

        if model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train, y_train):
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        }

        return metrics, predictions

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_

            if self.feature_names:
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values(by='importance', ascending=False)

                return feature_importance_df
            else:
                return pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importance))],
                    'importance': importance
                }).sort_values(by='importance', ascending=False)

        return None

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']

        return instance
