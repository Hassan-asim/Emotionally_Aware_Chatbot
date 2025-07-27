import joblib

# Load the model and vectorizer with allow_pickle=True
try:
    model = joblib.load('model.pkl', mmap_mode=None)
    vectorizer = joblib.load('vectorizer.pkl', mmap_mode=None)
    
    # Save them back with the current version
    joblib.dump(model, 'model_fixed.pkl', compress=3)
    joblib.dump(vectorizer, 'vectorizer_fixed.pkl', compress=3)
    print("Successfully converted the model files!")
except Exception as e:
    print(f"Error during conversion: {str(e)}")
