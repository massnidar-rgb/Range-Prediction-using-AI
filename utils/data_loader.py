import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def create_sample_ev_data(self, n_samples=500):
        np.random.seed(42)

        manufacturers = ['Tesla', 'Nissan', 'Chevrolet', 'BMW', 'Audi', 'Hyundai', 'Kia']
        models = ['Model 3', 'Leaf', 'Bolt', 'i3', 'e-tron', 'Kona', 'EV6']
        terrain_types = ['flat', 'mixed', 'hilly']
        driving_styles = ['eco', 'normal', 'sport']

        data = {
            'vehicle_id': range(1, n_samples + 1),
            'manufacturer': np.random.choice(manufacturers, n_samples),
            'vehicle_model': np.random.choice(models, n_samples),
            'battery_capacity': np.random.uniform(40, 100, n_samples),
            'efficiency': np.random.uniform(3.5, 6.0, n_samples),
            'weight': np.random.uniform(1500, 2500, n_samples),
            'temperature': np.random.uniform(-10, 40, n_samples),
            'charging_cycles': np.random.randint(0, 1000, n_samples),
            'vehicle_age': np.random.uniform(0, 10, n_samples),
            'avg_speed': np.random.uniform(30, 120, n_samples),
            'terrain_type': np.random.choice(terrain_types, n_samples),
            'driving_style': np.random.choice(driving_styles, n_samples)
        }

        df = pd.DataFrame(data)

        base_range = df['battery_capacity'] * df['efficiency']

        temp_factor = 1 - np.abs(df['temperature'] - 20) / 100

        terrain_factor = df['terrain_type'].map({'flat': 1.0, 'mixed': 0.9, 'hilly': 0.8})

        style_factor = df['driving_style'].map({'eco': 1.1, 'normal': 1.0, 'sport': 0.85})

        age_factor = 1 - (df['vehicle_age'] * 0.02)

        speed_factor = 1 - np.abs(df['avg_speed'] - 70) / 200

        weight_factor = 1 - (df['weight'] - 1500) / 5000

        df['range_km'] = (base_range * temp_factor * terrain_factor *
                         style_factor * age_factor * speed_factor * weight_factor)

        df['range_km'] = df['range_km'].clip(lower=100)

        noise = np.random.normal(0, 10, n_samples)
        df['range_km'] = df['range_km'] + noise

        return df

    def load_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"File not found: {filepath}")

    def save_data(self, df, filename):
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
