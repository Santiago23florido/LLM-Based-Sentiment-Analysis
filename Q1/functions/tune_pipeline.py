import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

def tune_pipeline(vectorizer, model, param_grid, X_train, y_train, model_name, output_dir="outputs", refit_metric="f1_macro"):
    
    os.makedirs(output_dir, exist_ok=True)

    pipe = Pipeline([
        ("vec", vectorizer),
        ("model", model)
    ])

    scoring = {
        "accuracy": "accuracy",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    # 1. Export the best model (Entire Pipeline: Vectorizer + Model)
    model_filename = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_best.joblib")
    joblib.dump(search.best_estimator_, model_filename)
    print(f"--- Best model saved to: {model_filename}")

    # 2. Save Tuning Report
    # Extract results and sort them by the ranking of your chosen metric
    results_df = pd.DataFrame(search.cv_results_)
    rank_column = f"rank_test_{refit_metric}"
    
    # Sort so the best model (Rank 1) is at the top
    results_df = results_df.sort_values(by=rank_column)
    
    report_filename = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_tuning_report.csv")
    results_df.to_csv(report_filename, index=False)
    print(f"--- Performance report saved to: {report_filename}")

    return search
