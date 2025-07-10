import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def adaboost(df_train_encoded, df_val_encoded, target_column, n_estimators=250, random_state=42):
    """
    Esegue classificazione AdaBoost con valutazione e grafico della confusion matrix.

    Args:
        df_train_encoded (pd.DataFrame): Dataset di training preprocessato e codificato.
        df_val_encoded (pd.DataFrame): Dataset di validazione preprocessato e codificato.
        target_column (str): Nome della colonna target da predire.
        n_estimators (int): Numero di estimatori per AdaBoost. Default = 100.

    Returns:
        model: Modello addestrato (AdaBoostClassifier)
    """

    # 1. Split in X / y
    X_train = df_train_encoded.drop(columns=[target_column])
    y_train = df_train_encoded[target_column]

    X_val = df_val_encoded.drop(columns=[target_column])
    y_val = df_val_encoded[target_column]

    # 2. Inizializza modello AdaBoost (versione >= 1.2 → usa "estimator")
    base_model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    model = AdaBoostClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        learning_rate=1.0,
        random_state=random_state
    )

    y_train = y_train.astype(bool)
    y_val = y_val.astype(bool)

    # 3. Allenamento
    model.fit(X_train, y_train)

    # 4. Predizione sul validation
    y_pred = model.predict(X_val)

    # 5. Valutazione
    print("\n=== Risultati AdaBoost ===")
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:\n", classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n", cm)

    # 6. Grafico Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix (AdaBoost, n_estimators={n_estimators})")
    plt.grid(False)
    plt.show()

    return model