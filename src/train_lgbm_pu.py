import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def train_lgbm_pu(folds, params):

    results = []

    for i, fold in enumerate(folds):

        print(f"\nTraining Fold {i}")

        X_train = fold["X_train"]
        y_train = fold["y_train"]

        X_test = fold["X_test"]
        y_test = fold["y_test"]

        train_data = lgb.Dataset(X_train, label=y_train)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500
        )

        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, preds)

        print("AUC:", auc)

        results.append(auc)

    print("\nMean AUC:", sum(results)/len(results))

    return results