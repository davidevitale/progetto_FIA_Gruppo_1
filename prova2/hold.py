import pandas as pd
from sklearn.model_selection import train_test_split

def holdout_split(
    input_path,
    train_path,
    val_path,
    test_size=0.2,
    stratify_col='Transported',
    random_state=42
):
    # === Caricamento del dataset ===
    df = pd.read_csv(input_path)

    # === Split holdout ===
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col] if stratify_col in df.columns else None,
        random_state=random_state
    )

    # === Salvataggio dei file ===
    train_df.to_excel(train_path, index=False)
    val_df.to_excel(val_path, index=False)

    print(f"Train salvato in: {train_path} ({train_df.shape[0]} righe)")
    print(f"Validation salvato in: {val_path} ({val_df.shape[0]} righe)")