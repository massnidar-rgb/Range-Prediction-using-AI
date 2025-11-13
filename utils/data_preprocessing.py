import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def clean_data(self, df):
        df_cleaned = df.copy()

        df_cleaned = df_cleaned.dropna()

        df_cleaned = df_cleaned.drop_duplicates()

        return df_cleaned

    def handle_missing_values(self, df, strategy='mean'):
        df_filled = df.copy()

        if strategy == 'mean':
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_columns] = df_filled[numeric_columns].fillna(
                df_filled[numeric_columns].mean()
            )
        elif strategy == 'median':
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_columns] = df_filled[numeric_columns].fillna(
                df_filled[numeric_columns].median()
            )
        elif strategy == 'mode':
            for column in df_filled.columns:
                df_filled[column].fillna(df_filled[column].mode()[0], inplace=True)

        return df_filled

    def encode_categorical(self, df, columns):
        df_encoded = df.copy()

        for column in columns:
            if column in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                self.label_encoders[column] = le

        return df_encoded

    def normalize_features(self, X_train, X_test=None):
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def prepare_data_for_modeling(self, df, target_column, test_size=0.2, random_state=42):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test
