import pandas as pd
from sklearn.model_selection import train_test_split

def holdout_split(
    input_path = 'C:/Users/dvita/Desktop/TITANIC/train.xlsx',
    train_path = 'C:/Users/dvita/Desktop/TITANIC/train_holdout.xlsx',
    val_path = 'C:/Users/dvita/Desktop/TITANIC/val_holdout.xlsx',
    test_size=0.2,
    stratify_col='Transported',
    random_state=42
):
    df = pd.read_excel(input_path)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col] if stratify_col in df.columns else None,
        random_state=random_state
    )
    train_df.to_excel(train_path, index=False)
    val_df.to_excel(val_path, index=False)
    print(f"Train saved to {train_path} ({train_df.shape[0]} rows)")
    print(f"Validation saved to {val_path} ({val_df.shape[0]} rows)")

# def holdout_split(
#     test_size=0.2,
#     stratify_col='Transported',
#     random_state=42
# ):
#     input_path = input("Inserisci il percorso del file di input (.xlsx): ")
#     train_path = input("Inserisci il path di dove salvare il file di training (.xlsx): ")
#     val_path = input("Inserisci il path di dove salvare il file di validation (.xlsx): ")
    
#     df = pd.read_excel(input_path)
#     train_df, val_df = train_test_split(
#         df,
#         test_size=test_size,
#         stratify=df[stratify_col] if stratify_col in df.columns else None,
#         random_state=random_state
#     )
#     train_df.to_excel(train_path, index=False)
#     val_df.to_excel(val_path, index=False)
#     print(f"Train saved to {train_path} ({train_df.shape[0]} rows)")
#     print(f"Validation saved to {val_path} ({val_df.shape[0]} rows)")

