import joblib

from neptune.integrations.sklearn import (
    create_classifier_summary,
)
from neptune_sklearn_charts import (
    create_classification_report_chart,
    create_confusion_matrix_chart,
    create_roc_auc_chart,
    create_precision_recall_chart,
    create_class_prediction_error_chart,
)


def log_metrics(
    nt_run,
    trained_model,
    encoder,
    X_train,
    X_test,
    y_train,
    y_test,
    train_metric,
    eval_metric,
):
    # from neptune.integrations.tensorflow_keras import NeptuneCallback
    # print('\n' + '-' * 30 + 'Neptune' + '-' * 30 + '\n')
    # nt_run = init_neptune(config['model']['path'])
    # nt_run["pickled-model"] = npt_utils.get_pickled_model(trained_model)
    train_acc, train_pre, train_rec, train_f1 = train_metric
    eval_acc, eval_pre, eval_rec, eval_f1 = eval_metric
    nt_run["metrics/train"] = {
        "train_acc": train_acc,
        "train_pre": train_pre,
        "train_rec": train_rec,
        "train_f1": train_f1,
    }
    nt_run["metrics/test"] = {
        "test_acc": eval_acc,
        "test_pre": eval_pre,
        "test_rec": eval_rec,
        "test_f1": eval_f1,
    }
    # nt_run["metrics/confusion-matrix"] = npt_utils.create_confusion_matrix_chart(
    #     trained_model, X_train, X_test, y_train, y_test
    # )

    nt_run.sync()
    with joblib.parallel_config(backend="threading", n_jobs=-1):
        print("metric: classifier summary")
        nt_run["metrics/summary"] = create_classifier_summary(
            trained_model,
            X_train,
            X_test,
            y_train,
            y_test,
            log_charts=False,
        )

        print("metric: classification report")
        classification_report = create_classification_report_chart(
            trained_model, X_train, X_test, y_train, y_test, classes=encoder.classes_
        )
        if classification_report:
            nt_run["metrics/summary"][
                "diagnostics_charts/classification_report"
            ] = classification_report

        print("metric: confusion matrix")
        confusion_matrix = create_confusion_matrix_chart(
            trained_model, X_train, X_test, y_train, y_test, classes=encoder.classes_
        )
        if confusion_matrix:
            nt_run["metrics/summary"][
                "diagnostics_charts/confusion_matrix"
            ] = confusion_matrix

        print("metric: roc auc")
        roc_auc = create_roc_auc_chart(
            trained_model, X_train, X_test, y_train, y_test, classes=encoder.classes_
        )
        if roc_auc:
            nt_run["metrics/summary"]["diagnostics_charts/ROC_AUC"] = roc_auc

        print("metric: precision recall")
        precision_recall = create_precision_recall_chart(
            trained_model, X_train, X_test, y_train, y_test, classes=encoder.classes_
        )
        if precision_recall:
            nt_run["metrics/summary"][
                "diagnostics_charts/precision_recall"
            ] = precision_recall

        print("metric: class prediction error")
        class_prediction_error = create_class_prediction_error_chart(
            trained_model, X_train, X_test, y_train, y_test, classes=encoder.classes_
        )
        if class_prediction_error:
            nt_run["metrics/summary"][
                "diagnostics_charts/class_prediction_error"
            ] = class_prediction_error

    nt_run.sync(wait=True)
