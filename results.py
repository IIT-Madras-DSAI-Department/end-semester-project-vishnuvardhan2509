import numpy as np
import pandas as pd
import time

from algorithms import (
    LinearRegressionOVR,
    LogisticRegressionOVR,
    KMeansClassifier,
    GaussianClassifier,
    EnsembleClassifier,
)


def load_csv(path):
    df = pd.read_csv(path)
    y = df.iloc[:, 0].to_numpy().astype(int)
    X = df.iloc[:, 1:].to_numpy().astype(np.float32) / 255.0
    return X, y

def f1_macro(y_true, y_pred):
    f1s = []
    for c in range(10):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        f1s.append(f1)
    return np.mean(f1s)

def predict_single_image(model, img_vector):
    img = np.array(img_vector).astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    img = img.reshape(1, -1)
    return int(model.predict(img)[0])


# Main program
if __name__ == "__main__":
    print("Loading dataset...")
    X_train, y_train = load_csv(r"C:/Users/vishn/OneDrive\Desktop/MNIST_train.csv")
    X_val, y_val = load_csv(r"C:/Users/vishn/OneDrive\Desktop/MNIST_validation.csv")
    print("Train:", X_train.shape, " Validation:", X_val.shape)

    results = {}

    # Train Linear Regression
    start = time.time()
    lr_model = LinearRegressionOVR()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_val)
    results["Linear Regression"] = (f1_macro(y_val, lr_pred), time.time() - start)

    # Train Logistic Regression
    start = time.time()
    log_model = LogisticRegressionOVR(lr=0.8, iters=150)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_val)
    results["Logistic Regression"] = (f1_macro(y_val, log_pred), time.time() - start)

    # Train KMeans
    start = time.time()
    km_model = KMeansClassifier(k=60, iters=15)
    km_model.fit(X_train, y_train)
    km_pred = km_model.predict(X_val)
    results["KMeans"] = (f1_macro(y_val, km_pred), time.time() - start)

    # Train Gaussian model
    start = time.time()
    gauss_model = GaussianClassifier()
    gauss_model.fit(X_train, y_train)
    gauss_pred = gauss_model.predict(X_val)
    results["Gaussian"] = (f1_macro(y_val, gauss_pred), time.time() - start)

    # Ensemble
    ensemble = EnsembleClassifier(
        models=[lr_model, log_model, km_model, gauss_model],
        weights=[1.2, 2.0, 0.6, 0.8]
    )
    start = time.time()
    ens_pred = ensemble.predict(X_val)
    ens_time = time.time() - start
    ens_f1 = f1_macro(y_val, ens_pred)

    print("\n MODEL RESULTS :")
    for model, (f1, t) in results.items():
        print(f"{model:25}  F1={f1:.4f}   Time={t:.2f}s")
    print("\n ENSEMBLE RESULT :")
    print(f"Ensemble F1 Score: {ens_f1:.4f}   Time={ens_time:.2f}s")
    # Example prediction
    test_img = X_val[5]  
    pred = predict_single_image(ensemble, test_img)

    print("\nSingle Image Prediction:")
    print("Predicted: ", pred)
