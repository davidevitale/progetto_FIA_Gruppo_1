import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def random_forest(df_train_encoded, df_val_encoded, target_column, n_estimators=500, max_depth=4, random_state=42):
    """
    Esegue classificazione Random Forest con valutazione e grafico della confusion matrix.

    Args:
        df_train_encoded (pd.DataFrame): Dataset di training preprocessato e codificato.
        df_val_encoded (pd.DataFrame): Dataset di validazione preprocessato e codificato.
        target_column (str): Nome della colonna target da predire.
        n_estimators (int): Numero di alberi nella foresta. Default = 100.
        max_depth (int): Profondità massima degli alberi. Default = 8.
        random_state (int): Semenza per riproducibilità. Default = 42.

    Returns:
        model: Modello addestrato (RandomForestClassifier)
    """

    # 1. Split in X / y
    X_train = df_train_encoded.drop(columns=[target_column])
    y_train = df_train_encoded[target_column]

    X_val = df_val_encoded.drop(columns=[target_column])
    y_val = df_val_encoded[target_column]

    y_train = y_train.astype(bool)
    y_val = y_val.astype(bool)

    # 2. Inizializza modello Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    # 3. Allenamento
    model.fit(X_train, y_train)

    # 4. Predizione sul validation
    y_pred = model.predict(X_val)

    # 5. Valutazione
    print("\n=== Risultati Random Forest ===")
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:\n", classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n", cm)

    # 6. Grafico Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Greens', values_format='d')
    plt.title(f"Confusion Matrix (Random Forest, n_estimators={n_estimators}, max_depth={max_depth})")
    plt.grid(False)
    plt.show()

    return model
