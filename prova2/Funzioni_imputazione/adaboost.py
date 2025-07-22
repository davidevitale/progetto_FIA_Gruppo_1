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

def adaboost(
    df_train_encoded,
    df_val_encoded,
    df_test_encoded,
    target_column,
    id_column,
    submission_filename,
    n_estimators=500,
):
    """
    Esegue classificazione AdaBoost, valutazione, grafico confusion matrix e salvataggio della submission.

    Args:
        df_train_encoded (pd.DataFrame): Dataset di training preprocessato e codificato.
        df_val_encoded (pd.DataFrame): Dataset di validazione preprocessato e codificato.
        df_test_encoded (pd.DataFrame): Dataset di test preprocessato e codificato (senza target).
        target_column (str): Nome della colonna target da predire.
        id_column (str): Nome della colonna identificativa da usare nella submission.
        submission_filename (str): Nome del file CSV di output.
        n_estimators (int): Numero di estimatori per AdaBoost. Default = 250.

    Returns:
        model: Modello addestrato (AdaBoostClassifier)
    """

    # 1. Split in X / y per train e val (rimuovo target e id)
    cols_to_drop = [target_column, id_column]

    X_train = df_train_encoded.drop(columns=cols_to_drop)
    y_train = df_train_encoded[target_column]

    X_val = df_val_encoded.drop(columns=cols_to_drop)
    y_val = df_val_encoded[target_column]

    # 2. Inizializza modello AdaBoost
    base_model = DecisionTreeClassifier(max_depth=3)
    model = AdaBoostClassifier(
        estimator=base_model,
        n_estimators=n_estimators,
        learning_rate=0.2,
    )

    # 3. Allenamento
    model.fit(X_train, y_train)

    # 4. Predizione sul validation
    y_pred = model.predict(X_val)

    # 5. Valutazione
    print("\n=== Risultati AdaBoost ===")
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n", cm)

    # 6. Grafico Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix (AdaBoost, n_estimators={n_estimators})")
    plt.grid(False)
    plt.show()

    # 7. Predizione sul test
    X_test = df_test_encoded.drop(columns=cols_to_drop)
    y_test_pred = model.predict(X_test)

    # 8. Submission con SOLO id_column e target_column (predizioni) convertendo a bool
    submission = pd.DataFrame({
        id_column: df_test_encoded[id_column],
        target_column: y_test_pred.astype(bool)
    })

    submission.to_csv(submission_filename, index=False)
    print(f"Sample submission salvato in: {submission_filename}")

    return model
