from prefect import flow, task
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

@task
def load_data():
    df = pd.read_csv("data/processed/cleaned_reviews.csv")

    # Critical fix: remove NaN reviews
    df = df.dropna(subset=["clean_review", "sentiment"])

    return df

@task
def train_model(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1 = f1_score(y_test, preds)
        mlflow.log_param("model", model_name)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

@flow(name="Flipkart Sentiment Training Pipeline")
def training_flow():
    mlflow.set_experiment("Flipkart_Sentiment_Analysis")

    df = load_data()

    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    train_model(
        "Logistic_Regression",
        LogisticRegression(max_iter=1000),
        X_train, X_test, y_train, y_test
    )

    train_model(
        "Linear_SVM",
        LinearSVC(),
        X_train, X_test, y_train, y_test
    )

if __name__ == "__main__":
    training_flow()
