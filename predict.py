import os
import json
import math
import sys

class FloodPredictionEngine:
    def __init__(self, model_path="model_inference.json", scaler_path=None):
        """
        Pure Python implementation of the Flood Prediction Engine.
        Loads from the unified JSON file.
        """
        try:
            # If legacy paths are passed explicitly but model_inference.json is present, prefer model_inference.json
            if (model_path == "floods.save" or model_path == "model_inference.json") and os.path.exists("model_inference.json"):
                model_path = "model_inference.json"
                
            if model_path == "model_inference.json":
                with open(model_path, 'r') as f:
                    data = json.load(f)
                self.feature_names = data['features']
                self.threshold = data['threshold']
                self.base_score = data['base_score']
                self.base_margin = math.log(self.base_score / (1.0 - self.base_score))
                self.scaler_mean = data['scaler_mean']
                self.scaler_scale = data['scaler_scale']
                self.trees = data['trees']
                self.loaded = True
                print("Prediction engine loaded successfully from JSON.")
            else:
                # If JSON is not found, try importing ML packages and falling back to joblib (useful for legacy compatibility)
                print(f"JSON model file {model_path} not found. Attempting legacy joblib fallback...")
                import joblib
                
                legacy_model_path = model_path
                legacy_scaler_path = "scaler.pkl" if scaler_path is None else scaler_path
                
                self.model_data = joblib.load(legacy_model_path)
                self.model = self.model_data['model']
                self.threshold = self.model_data['threshold']
                self.feature_names = self.model_data['features']
                self.scaler = joblib.load(legacy_scaler_path)
                
                self.legacy_mode = True
                self.loaded = True
                print("Legacy prediction engine loaded successfully using joblib.")
        except Exception as e:
            self.loaded = False
            self.error_message = str(e)
            print(f"Error loading prediction engine: {e}")

    def predict(self, input_features):
        """
        Accepts a dictionary of inputs, preprocesses them, and makes a prediction.
        input_features keys should match feature names.
        """
        if not self.loaded:
            return {"error": f"Model not loaded: {self.error_message}"}
            
        try:
            # Check if we are running in legacy mode
            if hasattr(self, 'legacy_mode') and self.legacy_mode:
                import pandas as pd
                input_df = pd.DataFrame([input_features])
                input_df = input_df[self.feature_names]
                input_scaled = self.scaler.transform(input_df)
                prob = self.model.predict_proba(input_scaled)[0, 1]
            else:
                # Pure Python prediction
                # 1. Scale inputs
                scaled_inputs = []
                for feat, mean, scale in zip(self.feature_names, self.scaler_mean, self.scaler_scale):
                    val = float(input_features.get(feat, 0.0))
                    scaled_inputs.append((val - mean) / scale)
                
                # 2. Evaluate trees
                sum_weights = 0.0
                for tree in self.trees:
                    node = 0
                    while True:
                        left = tree['left_children'][node]
                        right = tree['right_children'][node]
                        if left == -1 and right == -1:
                            sum_weights += tree['base_weights'][node]
                            break
                        
                        split_idx = tree['split_indices'][node]
                        split_val = tree['split_conditions'][node]
                        val = scaled_inputs[split_idx]
                        
                        if val < split_val:
                            node = left
                        else:
                            node = right
                            
                # 3. Compute probability using sigmoid
                margin = self.base_margin + sum_weights
                prob = 1.0 / (1.0 + math.exp(-margin))
                
            # Make prediction based on the tuned threshold
            prediction = 1 if prob >= self.threshold else 0
            
            # Formulate recommendation and status
            if prediction == 1:
                status = "High Flood Risk"
                confidence = prob * 100
                recommendations = [
                    "Issue immediate flood warnings for low-lying areas.",
                    "Initiate evacuation protocols for high-vulnerable regions.",
                    "Pre-position disaster response supplies and personnel.",
                    "Establish communication channels for emergency coordination."
                ]
            else:
                status = "No Flood Risk"
                confidence = (1 - prob) * 100
                recommendations = [
                    "Maintain standard meteorological monitoring.",
                    "Keep drainage systems clear of debris.",
                    "Check emergency equipment and supplies readiness.",
                    "Review disaster plan protocols periodically."
                ]
                
            return {
                "prediction": prediction,
                "status": status,
                "confidence": round(confidence, 2),
                "probability": round(prob, 4),
                "threshold": round(self.threshold, 4),
                "recommendations": recommendations
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

def main():
    print("--- Flood Prediction CLI Engine ---")
    engine = FloodPredictionEngine()
    
    if not engine.loaded:
        print("Could not initialize engine. Exiting.")
        sys.exit(1)
        
    print("\nPlease enter meteorological measurements:")
    
    try:
        inputs = {}
        inputs['annual_rainfall'] = float(input("Annual Rainfall (mm, e.g., 2500): "))
        inputs['seasonal_rainfall'] = float(input("Seasonal Rainfall (mm, e.g., 1500): "))
        inputs['monthly_rainfall'] = float(input("Peak Monthly Rainfall (mm, e.g., 600): "))
        inputs['cloud_visibility'] = float(input("Cloud Visibility (km, e.g., 4.5): "))
        inputs['temperature'] = float(input("Temperature (°C, e.g., 27.5): "))
        inputs['humidity'] = float(input("Humidity (%, e.g., 85): "))
        inputs['atmospheric_pressure'] = float(input("Atmospheric Pressure (hPa, e.g., 1008): "))
        inputs['wind_speed'] = float(input("Wind Speed (km/h, e.g., 32): "))
        inputs['river_level'] = float(input("River Level (m, e.g., 8.2): "))
        inputs['cloud_cover'] = float(input("Cloud Cover (%, e.g., 80): "))
        
        result = engine.predict(inputs)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
            
        print("\n" + "="*40)
        print("          PREDICTION RESULTS          ")
        print("="*40)
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Model Probability Score: {result['probability']} (Threshold: {result['threshold']})")
        print("\nRecommended Actions:")
        for rec in result['recommendations']:
            print(f" - {rec}")
        print("="*40)
        
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        sys.exit(1)

if __name__ == "__main__":
    main()
