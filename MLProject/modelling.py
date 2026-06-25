import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Memulai Workflow CI: Load data...")
    df = pd.read_csv("adult_clean.csv")
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Biarkan MLflow yang mengatur Run ID secara otomatis
    mlflow.sklearn.autolog()
    
    print("Melatih model RandomForest...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(model, "model")
    print("Model berhasil dilatih dan disimpan!")

if __name__ == "__main__":
    main()