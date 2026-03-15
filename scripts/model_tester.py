from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report


def compute_mae(y_true, y_pred):
    """
    Computes Mean Absolute Error for ordinal sentiment labels (0–4).
    """
    return mean_absolute_error(y_true, y_pred)


def compute_accuracy(y_true, y_pred):
    """
    Computes simple classification accuracy.
    """
    return accuracy_score(y_true, y_pred)


def compute_report(y_true, y_pred):
    """
    Returns detailed precision/recall/f1 report.
    """
    return classification_report(y_true, y_pred)


def evaluate_model(model_name, y_true, y_pred):
    """
    Runs all evaluation metrics and prints results.
    """

    mae = compute_mae(y_true, y_pred)
    acc = compute_accuracy(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"Evaluation: {model_name}")
    print("=" * 60)

    print(f"Accuracy: {acc:.3f}")
    print(f"MAE: {mae:.3f}")

    print("\nClassification Report:")
    print(compute_report(y_true, y_pred))

    return {
        "model": model_name,
        "accuracy": acc,
        "mae": mae
    }
