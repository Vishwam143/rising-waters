import sys
import joblib
import numpy as np
import pandas as pd

class FloodPredictionEngine:
    def __init__(self, model_path="floods.save", scaler_path="scaler.pkl"):
        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.threshold = self.model_data['threshold']
            self.feature_names = self.model_data['features']
            self.scaler = joblib.load(scaler_path)
            self.loaded = True
            print("Prediction engine loaded successfully.")
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
            # Create a dataframe with features in correct order
            input_df = pd.DataFrame([input_features])
            input_df = input_df[self.feature_names]
            
            # Apply preprocessing (scaling)
            input_scaled = self.scaler.transform(input_df)
            
            # Generate probability
            prob = self.model.predict_proba(input_scaled)[0, 1]
            
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
