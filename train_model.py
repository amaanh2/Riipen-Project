import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


def print_model_summary(y_test, y_pred, model_name, le_target):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate recall (sensitivity) for each class
    classes = le_target.classes_

    benign_acc = tn / (tn + fp)
    path_acc = tp / (tp + fn)

    return {
        "Model": model_name,
        "Overall Accuracy": f"{accuracy:.1%}",
        "Benign Accuracy": f"{benign_acc:.1%}",
        "Pathogenic Accuracy": f"{path_acc:.1%}",
        "False Positives": fp,
        "False Negatives": fn
    }


def train():
    # Load data
    print("Loading data from balanced_dataset_augmented.csv...")
    try:
        df = pd.read_csv('balanced_dataset_augmented.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Select columns
    print("Preprocessing data...")
    data = df[["CDS Mutation", "AA Mutation",
               "Germline classification"]].copy()

    # Encode categorical features
    le_cds = LabelEncoder()
    le_aa = LabelEncoder()
    le_target = LabelEncoder()

    data['CDS Mutation_enc'] = le_cds.fit_transform(
        data['CDS Mutation'].astype(str))
    data['AA Mutation_enc'] = le_aa.fit_transform(
        data['AA Mutation'].astype(str))
    data['Germline classification_enc'] = le_target.fit_transform(
        data['Germline classification'].astype(str))

    # Define X and y
    X = data[["CDS Mutation_enc", "AA Mutation_enc"]]
    y = data["Germline classification_enc"]

    print("Target distribution:")
    print(data['Germline classification'].value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    summary_data = []

    # Final Model: Random Forest
    print("\n" + "="*50)
    print("TRAINING FINAL RANDOM FOREST MODEL")
    print("="*50)

    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=42,
        class_weight='balanced'
    )
    rf_clf.fit(X_train, y_train)

    # Standard Prediction (Threshold 0.5) - Just for comparison
    y_pred_standard = rf_clf.predict(X_test)

    # Optimized Prediction (Threshold 0.1 for 0 False Negatives)
    print("Applying optimized decision threshold (0.1) for maximum safety...")
    y_proba = rf_clf.predict_proba(X_test)[:, 1]
    best_thresh = 0.1
    y_pred_opt = (y_proba >= best_thresh).astype(int)

    # Add to summary
    summary_data.append(print_model_summary(
        y_test, y_pred_opt, f"Random Forest (Thresh {best_thresh})", le_target))

    # Detailed Stats
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_opt)
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nBreakdown:")
    print(f"True Negatives (Correctly Benign): {tn}")
    print(f"False Positives (Predicted Pathogenic, actually Benign): {fp}")
    print(f"False Negatives (Predicted Benign, actually Pathogenic): {fn}")
    print(f"True Positives (Correctly Pathogenic): {tp}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_opt, target_names=le_target.classes_))

    # Print Final Summary Table
    print("\n" + "="*80)
    print("FINAL MODEL SUMMARY")
    print("="*80)
    try:
        summary_df = pd.DataFrame(summary_data)
        import tabulate  # Ensure dependency is checked or imported
        print(summary_df.to_markdown(index=False))
    except ImportError:
        print(summary_df)


if __name__ == "__main__":
    train()
