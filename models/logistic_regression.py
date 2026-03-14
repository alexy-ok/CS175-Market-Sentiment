from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, classification_report

def run_logistic_regression(texts, labels):

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=0.3,
        random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42
    )


    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)


    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train_vec, y_train)

    val_preds = model.predict(X_val_vec)
    print("Validation MAE:", mean_absolute_error(y_val, val_preds))
    print(classification_report(y_val, val_preds))

    test_preds = model.predict(X_test_vec)
    print("Test MAE:", mean_absolute_error(y_test, test_preds))
    print(classification_report(y_test, test_preds))
    

    