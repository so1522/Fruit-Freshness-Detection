import os
import sys
import joblib
import numpy as np

# Ensure root directory and current directory are in sys.path regardless of execution path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from backend.feature_extractor import extract_features_from_bytes, extract_features_from_path
except ModuleNotFoundError:
    from feature_extractor import extract_features_from_bytes, extract_features_from_path

FRUIT_TIPS = {
    "fresh apple": {
        "title": "Fresh Apple detected!",
        "freshness_status": "Fresh & Healthy",
        "storage": "Store in a cool, dry place or in the refrigerator crisper drawer (1-4°C / 34-40°F).",
        "shelf_life": "5 - 7 weeks in refrigerator",
        "recommendation": "Safe for direct consumption, juices, fruit salads, or baking.",
    },
    "fresh banana": {
        "title": "Fresh Banana detected!",
        "freshness_status": "Fresh & Healthy",
        "storage": "Keep at room temperature away from direct sunlight. Do not refrigerate unpeeled unless ripe.",
        "shelf_life": "5 - 7 days at room temperature",
        "recommendation": "Perfect for immediate eating, smoothies, or breakfast toppings.",
    },
    "fresh orange": {
        "title": "Fresh Orange detected!",
        "freshness_status": "Fresh & Healthy",
        "storage": "Store at room temperature for up to a week, or refrigerate for longer freshness.",
        "shelf_life": "2 - 3 weeks in refrigerator",
        "recommendation": "Rich in Vitamin C. Safe to consume raw or prepare fresh juice.",
    },
    "rottenapples": {
        "title": "Rotten Apple detected!",
        "freshness_status": "Decayed / Spoiled",
        "storage": "Isolate immediately from other fresh fruits to prevent ethylene gas spread.",
        "shelf_life": "Expired",
        "recommendation": "Do not consume raw. Discard safely or use for organic composting.",
    },
    "rottenbanana": {
        "title": "Rotten Banana detected!",
        "freshness_status": "Decayed / Overripe",
        "storage": "Isolate from other produce. Dark soft patches indicate structural decay.",
        "shelf_life": "Expired",
        "recommendation": "Unsafe for raw eating. If mildly bruised without mold, can be used for banana bread; otherwise compost.",
    },
    "rottenoranges": {
        "title": "Rotten Orange detected!",
        "freshness_status": "Decayed / Moldy",
        "storage": "Discard immediately. Mold spores on citrus spread quickly to adjacent fruits.",
        "shelf_life": "Expired",
        "recommendation": "Do not consume or juice. Mold toxins (mycotoxins) may be present.",
    },
}


class ModelLoader:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(ROOT_DIR, "saved_models")

        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.best_model_name = "XGBoost"
        self._load_models()

    def _load_models(self):
        try:
            model_path = os.path.join(self.model_dir, "best_model.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
            name_path = os.path.join(self.model_dir, "best_model_name.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            
            if os.path.exists(name_path):
                self.best_model_name = joblib.load(name_path)

            print(f"[ModelLoader] Successfully loaded '{self.best_model_name}' model from {self.model_dir}")
        except Exception as e:
            print(f"[ModelLoader Error] Failed to load models: {e}")
            raise e

    def predict_bytes(self, img_bytes):
        feature_vector, metrics = extract_features_from_bytes(img_bytes)
        return self._predict_feature_vector(feature_vector, metrics)

    def predict_path(self, image_path):
        feature_vector, metrics = extract_features_from_path(image_path)
        return self._predict_feature_vector(feature_vector, metrics)

    def _predict_feature_vector(self, feature_vector, metrics):
        feat_input = feature_vector.reshape(1, -1)

        # Scale features if model requires scaling
        if self.best_model_name in ["KNN", "SVM", "Logistic Regression"]:
            feat_input = self.scaler.transform(feat_input)

        prediction_idx = self.model.predict(feat_input)[0]
        predicted_raw = str(self.label_encoder.inverse_transform([prediction_idx])[0])

        # Get probabilities
        probabilities = {}
        confidence = 100.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_input)[0]
            confidence = float(np.max(probs) * 100.0)
            for cls_name, prob in zip(self.label_encoder.classes_, probs):
                probabilities[str(cls_name)] = round(float(prob * 100.0), 2)
        else:
            probabilities[predicted_raw] = 100.0

        # Standardize key names & labels
        raw_key = predicted_raw.lower()
        if raw_key in ["freshapple", "fresh apple"]:
            std_key = "fresh apple"
            display_name = "Fresh Apple"
            fruit_type = "Apple"
            is_fresh = True
        elif raw_key in ["freshbanana", "fresh banana"]:
            std_key = "fresh banana"
            display_name = "Fresh Banana"
            fruit_type = "Banana"
            is_fresh = True
        elif raw_key in ["freshorange", "freshoranges", "fresh orange"]:
            std_key = "fresh orange"
            display_name = "Fresh Orange"
            fruit_type = "Orange"
            is_fresh = True
        elif "apple" in raw_key:
            std_key = "rottenapples"
            display_name = "Rotten Apple"
            fruit_type = "Apple"
            is_fresh = False
        elif "banana" in raw_key:
            std_key = "rottenbanana"
            display_name = "Rotten Banana"
            fruit_type = "Banana"
            is_fresh = False
        elif "orange" in raw_key:
            std_key = "rottenoranges"
            display_name = "Rotten Orange"
            fruit_type = "Orange"
            is_fresh = False
        else:
            std_key = raw_key
            display_name = raw_key.title()
            fruit_type = "Fruit"
            is_fresh = "fresh" in raw_key

        advice = FRUIT_TIPS.get(std_key, {
            "title": f"{display_name} detected",
            "freshness_status": "Fresh" if is_fresh else "Rotten",
            "storage": "Store appropriately.",
            "shelf_life": "N/A",
            "recommendation": "Inspect manually before consumption."
        })

        formatted_probabilities = {
            "Fresh Apple": probabilities.get("fresh apple", 0.0),
            "Fresh Banana": probabilities.get("fresh banana", 0.0),
            "Fresh Orange": probabilities.get("fresh orange", 0.0),
            "Rotten Apple": probabilities.get("rottenapples", 0.0),
            "Rotten Banana": probabilities.get("rottenbanana", 0.0),
            "Rotten Orange": probabilities.get("rottenoranges", 0.0),
        }

        return {
            "prediction": std_key,
            "display_name": display_name,
            "fruit_type": fruit_type,
            "is_fresh": is_fresh,
            "confidence": round(confidence, 2),
            "probabilities": formatted_probabilities,
            "raw_probabilities": probabilities,
            "advice": advice,
            "metrics": metrics,
            "model_used": str(self.best_model_name),
        }


# Global singleton instance
_model_loader_instance = None

def get_model_loader():
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader()
    return _model_loader_instance


if __name__ == "__main__":
    loader = get_model_loader()
    print("ModelLoader initialized successfully!")
