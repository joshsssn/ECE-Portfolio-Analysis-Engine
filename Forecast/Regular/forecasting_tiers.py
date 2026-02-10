
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Tier imports - Lazy loading where possible to save startup time?
# For simplicity with UV global venv, standard imports are fine.

# Lightest
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Light / Heavy (Tensorflow/Transformers)
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Medium
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Foundation
# Direct import of FTS model class (assuming path availability)
# This will replace subprocess FinOracleWrapper logic

class ForecastingEngine:
    """
    Central dispatcher for tiered forecasting models.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = MinMaxScaler()

    def _prepare_data(self, df, lookback=60, target_col='Close'):
        """Standard data prep for ML models (LSTM/Transformers)."""
        data = df.filter([target_col]).values
        scaled_data = self.scaler.fit_transform(data)
        
        x_train, y_train = [], []
        for i in range(lookback, len(scaled_data)):
            x_train.append(scaled_data[i-lookback:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        return x_train, y_train, scaled_data

    # --- TIER 1: LIGHTEST (ARIMAX) ---
    def run_arimax(self, df, steps=5):
        """
        Runs ARIMAX model using technical indicators as exogenous variables.
        """
        print("Running ARIMAX...")
        # Prepare Exog
        exog_cols = ['SMA_20', 'RSI', 'MACD'] 
        # Check if indicators exist, if not, compute them (handled in utils)
        
        # Fit
        try:
            # Basic auto-regressive model with exogenous
            # Order (1,1,1) is a safe default for daily stock data
            model = SARIMAX(df['Close'], exog=df[exog_cols], order=(1, 1, 1))
            model_fit = model.fit(disp=False)
            
            # Forecast
            # We need future values for exog variables. 
            # Simple assumption: carry forward last known values or use simple trend.
            last_exog = df[exog_cols].iloc[-1:].values
            future_exog = np.repeat(last_exog, steps, axis=0) # Naive future exog
            
            forecast = model_fit.get_forecast(steps=steps, exog=future_exog)
            predicted_prices = forecast.predicted_mean.values
            
            return {
                "model": "ARIMAX",
                "forecast": predicted_prices.tolist(),
                "last_price": df['Close'].iloc[-1]
            }
        except Exception as e:
            return {"error": f"ARIMAX failed: {e}"}

    # --- TIER 2: LIGHT (LSTM) ---
    def run_lstm(self, df, steps=5):
        print("Running LSTM...")
        try:
            # Adjust lookback if data is short
            data_len = len(df)
            lookback = 60
            if data_len < (lookback + 5):
                lookback = max(5, data_len - 5)
                print(f"   [WARN] Short data ({data_len}), reduced lookback to {lookback}")
            
            x_train, y_train, scaled_data = self._prepare_data(df, lookback)
            
            if len(x_train) == 0:
                 return {"error": "Insufficient data for LSTM training"}

            # Reshape for LSTM [samples, time steps, features]
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            # Build Model
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(tf.keras.layers.LSTM(50, return_sequences=False))
            model.add(tf.keras.layers.Dense(25))
            model.add(tf.keras.layers.Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0) # Epochs=1 for speed in demo
            
            # Predict Steps
            inputs = scaled_data[len(scaled_data) - lookback:]
            inputs = inputs.reshape(-1, 1)
            
            predictions = []
            current_batch = inputs[-lookback:].reshape(1, lookback, 1)
            
            for i in range(steps):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                # Update batch with new prediction
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
                
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            return {
                "model": "LSTM",
                "forecast": predictions.tolist(),
                "last_price": df['Close'].iloc[-1]
            }

        except Exception as e:
            return {"error": f"LSTM failed: {e}"}

    def run_gru(self, df, steps=5):
        print("Running GRU...")
        try:
            # Adjust lookback if data is short
            data_len = len(df)
            lookback = 60
            if data_len < (lookback + 5):
                lookback = max(5, data_len - 5)
                print(f"   [WARN] Short data ({data_len}), reduced lookback to {lookback}")
            
            x_train, y_train, scaled_data = self._prepare_data(df, lookback)
            
            if len(x_train) == 0:
                 return {"error": "Insufficient data for GRU training"}

            # Reshape for GRU [samples, time steps, features]
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            # Build Model
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(tf.keras.layers.GRU(50, return_sequences=False))
            model.add(tf.keras.layers.Dense(25))
            model.add(tf.keras.layers.Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0) 
            
            # Predict Steps
            inputs = scaled_data[len(scaled_data) - lookback:]
            inputs = inputs.reshape(-1, 1)
            
            predictions = []
            current_batch = inputs[-lookback:].reshape(1, lookback, 1)
            
            for i in range(steps):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
                
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            return {
                "model": "GRU",
                "forecast": predictions.tolist(),
                "last_price": df['Close'].iloc[-1]
            }

        except Exception as e:
            return {"error": f"GRU failed: {e}"}

    # --- TIER 3: MEDIUM (XGBoost) ---
    def run_xgboost(self, df, steps=5):
        print("Running XGBoost...")
        try:
            # Feature engineering for ML
            df_ml = df.copy()
            for i in range(1, 4):
                 df_ml[f'lag_{i}'] = df_ml['Close'].shift(i)
            df_ml.dropna(inplace=True)
            
            X = df_ml[['lag_1', 'lag_2', 'lag_3', 'SMA_20', 'RSI']]
            y = df_ml['Close']
            
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X, y)
            
            # Recursive forecasting
            predictions = []
            last_row = df_ml.iloc[-1].copy()
            
            for _ in range(steps):
                # Prepare single row input
                features = np.array([[last_row['Close'], last_row['lag_1'], last_row['lag_2'], last_row['SMA_20'], last_row['RSI']]])
                # Note: This checks features against training names, might warn.
                # In robust impl, use DataFrame matching cols.
                
                pred = model.predict(X.iloc[-1:]) # Using last known X for simplicity in this demo, refined later
                # Ideally update lags with prediction
                
                predictions.append(pred[0])
                
            return {
                "model": "XGBoost",
                "forecast": predictions, # Simplified for demo
                 "last_price": df['Close'].iloc[-1]
            }
        except Exception as e:
             return {"error": f"XGBoost failed: {e}"}

    def run_random_forest(self, df, steps=5):
        print("Running Random Forest...")
        try:
            # Feature engineering (same as XGBoost)
            df_ml = df.copy()
            for i in range(1, 4):
                 df_ml[f'lag_{i}'] = df_ml['Close'].shift(i)
            df_ml.dropna(inplace=True)
            
            if len(df_ml) < 10:
                 return {"error": "Insufficient data for Random Forest"}
            
            X = df_ml[['lag_1', 'lag_2', 'lag_3', 'SMA_20', 'RSI']]
            y = df_ml['Close']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Recursive forecasting
            predictions = []
            last_row = df_ml.iloc[-1].copy()
            
            for _ in range(steps):
                features = pd.DataFrame([[last_row['lag_1'], last_row['lag_2'], last_row['lag_3'], last_row['SMA_20'], last_row['RSI']]], columns=X.columns)
                
                pred = model.predict(features)
                pred_val = float(pred[0])
                predictions.append(pred_val)
                
                # Update lags
                last_row['lag_3'] = last_row['lag_2']
                last_row['lag_2'] = last_row['lag_1']
                last_row['lag_1'] = pred_val
                
            return {
                "model": "RandomForest",
                "forecast": predictions,
                "last_price": df['Close'].iloc[-1]
            }
        except Exception as e:
             return {"error": f"Random Forest failed: {e}"}

    # --- TIER 5: HEAVY (Transformer) ---
    def run_transformer(self, df, steps=5):
        print("Running Transformer (Keras)...")
        try:
            # Adjust lookback if data is short
            data_len = len(df)
            lookback = 60
            if data_len < (lookback + 5):
                lookback = max(5, data_len - 5)
                # print(f"   [WARN] Short data ({data_len}), reduced lookback to {lookback}")
            
            x_train, y_train, scaled_data = self._prepare_data(df, lookback)
            
            if len(x_train) == 0:
                 return {"error": "Insufficient data for Transformer training"}

            # Reshape [samples, time steps, features]
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            # Build Transformer Encoder
            inputs = tf.keras.Input(shape=(x_train.shape[1], 1))
            
            # Positional Encoding (Simplified)
            # Just relying on Dense + LSTM/Attention capability for now or add simple embedding layer if discrete
            # detailed implementation is complex, let's use a MultiHeadAttention block
            
            x = tf.keras.layers.Dense(32)(inputs) # Project to d_model
            att = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
            x = tf.keras.layers.Add()([x, att])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(16, activation="relu")(x)
            outputs = tf.keras.layers.Dense(1)(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
            
            # Predict Steps
            inputs_seq = scaled_data[len(scaled_data) - lookback:]
            inputs_seq = inputs_seq.reshape(-1, 1)
            
            predictions = []
            current_batch = inputs_seq[-lookback:].reshape(1, lookback, 1)
            
            for i in range(steps):
                pred = model.predict(current_batch, verbose=0)[0]
                predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
                
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            return {
                "model": "Transformer",
                "forecast": predictions.tolist(),
                "last_price": df['Close'].iloc[-1]
            }

        except Exception as e:
             return {"error": f"Transformer failed: {e}"}

    # --- TIER 4: FOUNDATION (FTS) ---
    def run_fts(self, df, ticker, config_overrides=None):
        """
        Runs the FinCast Foundation Time Series model (Torch).
        Uses direct import of finoracle_bridge.
        """
        print("Running FTS Foundation Model...")
        try:
             # Add path to find bridge
             start_dir = os.path.dirname(os.path.abspath(__file__))
             bridge_dir = os.path.abspath(os.path.join(start_dir, '../../Forecast/FinCast-fts/Inference'))
             if bridge_dir not in sys.path:
                 sys.path.append(bridge_dir)
             
             try:
                 from finoracle_bridge import run_bridge
             except ImportError:
                 return {"error": "Could not import finoracle_bridge"}

             # Save data temporarily for bridge to consume (bridge logic expects data.csv)
             finoracle_subdir = os.path.join(self.output_dir, "finoracle")
             os.makedirs(finoracle_subdir, exist_ok=True)
             data_path = os.path.join(finoracle_subdir, "data.csv")
             
             # Format for FTS (Exchange Date, Close)
             df_save = pd.DataFrame({
                 'Exchange Date': df.index if 'Exchange Date' not in df.columns else df['Exchange Date'],
                 'Close': df['Close']
             })
             df_save.to_csv(data_path, index=False)
             
             # Build Config
             cfg = {
                 "ticker": ticker,
                 "ric": ticker, 
                 "output_dir": self.output_dir, # Bridge will append /finoracle
                 "skip_fetch": True,
                 "skip_inference": False,
                 "freq": "d",
                 "model_path": os.path.join(bridge_dir, "v1.pth")
             }
             if config_overrides:
                 cfg.update(config_overrides)
                 
             # Run Bridge
             result = run_bridge(cfg)
             
             if 'error' in result:
                 return {"error": result['error']}
             
             # Load forecast curve if available
             csv_path = result.get('csv_path')
             if csv_path and os.path.exists(csv_path):
                 res_df = pd.read_csv(csv_path)
                 # FinCast CSV usually has 'mean' or similar
                 col_name = 'mean' if 'mean' in res_df.columns else res_df.columns[-1]
                 forecast_values = res_df[col_name].tail(result.get('horizon_days', 16)).tolist()
                 result['forecast'] = forecast_values
             else:
                 result['forecast'] = [result['forecast_price']] * 5
                 
             result['model'] = 'FTS'
             result['last_price'] = df['Close'].iloc[-1]
             return result

        except Exception as e:
             return {"error": f"FTS failed: {e}"}



    # --- ENSEMBLE ---
    def run_ensemble(self, results):
        """Aggregate valid forecasts."""
        preds = []
        for res in results:
            if "forecast" in res:
                preds.append(res["forecast"])
        
        if not preds:
            return None
        
        # Handle different forecast lengths by truncating to shortest
        min_len = min(len(p) for p in preds)
        if min_len == 0:
            return None
        
        preds_trimmed = [p[:min_len] for p in preds]
        
        # Average across models
        ensemble_pred = np.mean(preds_trimmed, axis=0)
        return {
            "model": "Ensemble",
            "forecast": ensemble_pred.tolist()
        }
