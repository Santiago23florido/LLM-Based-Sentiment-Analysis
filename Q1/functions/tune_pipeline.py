from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

def tune_pipeline(vectorizer, model, param_grid, X_train, y_train, refit_metric="f1_macro"):
    
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
    return search
