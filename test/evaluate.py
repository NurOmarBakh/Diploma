# evaluate.py

import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def tokenize(text: str) -> set[str]:
    """
    Простая токенизация: всё в нижний регистр, берём только буквы/цифры.
    """
    return set(re.findall(r"\w+", text.lower()))

def evaluate(test_path: str, pred_path: str):
    # 1) Загрузка данных
    df_true = pd.read_csv(test_path, sep='\t', header=0, usecols=['Вопрос', 'Ответ'])
    df_true = df_true.rename(columns={'Ответ': 'true_answer'})

    df_pred = pd.read_csv(pred_path, sep='\t', header=0, usecols=['Вопрос', 'Ответ'])
    df_pred = df_pred.rename(columns={'Ответ': 'predicted_answer'})

    # 2) Merge по вопросу
    df = pd.merge(df_true, df_pred, on='Вопрос', how='inner')

    # 3) Подготовка списков результатов
    exact_matches = []
    precisions = []
    recalls = []

    for idx, row in df.iterrows():
        q = row['Вопрос']
        true = str(row['true_answer'])
        pred = str(row['predicted_answer'])

        # Exact match
        is_exact = (pred.strip().lower() == true.strip().lower())
        exact_matches.append(int(is_exact))

        # Token-level precision/recall
        t_true = tokenize(true)
        t_pred = tokenize(pred)

        precision = len(t_true & t_pred) / len(t_pred) if t_pred else 0.0
        recall    = len(t_true & t_pred) / len(t_true) if t_true else 0.0

        precisions.append(precision)
        recalls.append(recall)

        print(f"[{idx+1}/{len(df)}] Q: {q}")
        print(f"    → True : {true}")
        print(f"    → Pred : {pred}")
        print(f"    Exact : {is_exact}, Precision: {precision:.3f}, Recall: {recall:.3f}\n")

    # 4) Итоговые метрики
    accuracy = accuracy_score([1]*len(exact_matches), exact_matches)
    p_avg = np.mean(precisions)
    r_avg = np.mean(recalls)
    f1 = (2 * p_avg * r_avg / (p_avg + r_avg)) if (p_avg + r_avg) > 0 else 0.0

    print("=== Evaluation results ===")
    print(f"Exact Match Accuracy : {accuracy:.4f}")
    print(f"Token Precision      : {p_avg:.4f}")
    print(f"Token Recall         : {r_avg:.4f}")
    print(f"Token F1-score       : {f1:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_model_txt.py test.txt ytest2.txt")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
