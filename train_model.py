import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

def train():
    # Load data
    print("Loading data from dataset.xlsx...")
    try:
        df = pd.read_excel('dataset.xlsx')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Select columns
    print("Preprocessing data...")
    # Keep original dataframe intact for clarity
    data = df[["CDS Mutation", "AA Mutation", "Germline classification"]].copy()
    
    # Check for nulls (optional based on notebook, but good practice)
    # data = data.dropna() 

    # Encode categorical features
    # Using separate encoders for each column is best practice
    le_cds = LabelEncoder()
    le_aa = LabelEncoder()
    le_target = LabelEncoder()
    
    data['CDS Mutation_enc'] = le_cds.fit_transform(data['CDS Mutation'].astype(str))
    data['AA Mutation_enc'] = le_aa.fit_transform(data['AA Mutation'].astype(str))
    data['Germline classification_enc'] = le_target.fit_transform(data['Germline classification'].astype(str))
    
    # Define X and y
    X = data[["CDS Mutation_enc", "AA Mutation_enc"]]
    y = data["Germline classification_enc"]
    
    print("Target distribution:")
    print(data['Germline classification'].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Model 1: Decision Tree
    print("\n--- Decision Tree ---")
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred_dt))
    
    # Model 2: Random Forest
    print("\n--- Random Forest ---")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))

    # Model 3: Naive Bayes
    print("\n--- Gaussian Naive Bayes ---")
    # CategoricalNB can be tricky with LabelEncoder if indices are not perfectly continuous or if test set has unseen values
    # GaussianNB is often more robust for simple encoded data, or we need to ensure categories are perfectly mapped.
    # Trying GaussianNB first as a robust alternative since CategoricalNB crashed in the notebook.
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb, target_names=le_target.classes_, zero_division=0))

    # Model 4: CatBoost
    print("\n--- CatBoost ---")
    # Using params from notebook
    cat_clf = CatBoostClassifier(
        iterations=300, 
        depth=6, 
        learning_rate=0.1, 
        loss_function='Logloss',
        verbose=0, 
        class_weights=[1, 31]
    )
    
    # CatBoost can handle categorical features directly if we tell it
    # Here we are using the integer encoded versions
    cat_features = [0, 1] 
    cat_clf.fit(X_train, y_train, cat_features=cat_features)
    
    y_pred_cat = cat_clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred_cat))
    print("\nClassification Report (CatBoost):")
    print(classification_report(y_test, y_pred_cat, target_names=le_target.classes_))

if __name__ == "__main__":
    train()

