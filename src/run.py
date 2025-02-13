"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT

Project: ENViSEC - Artificial Intelligence-enabled Cybersecurity for Future Smart Environments 
(funded from the European Union’s Horizon 2020, NGI-POINTER under grant agreement No 871528).
@Programmer: Guru Bhandari
@File - to run different ML models for training, testing and prediction. 
"""

import os
import pickle
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from sklearn import metrics

# import tensorflow_addons as tfa

from keras.api.wrappers import SKLearnClassifier

import src.metrics as metrics2
from src.utility import (
    init_neptune,
    load_config,
    load_data,
    plot_history,
    utilize_gpu,
    get_dataset_name,
)

print("\n\n\nimport done\n\n\n")


Split = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]


def split_data(data: pd.DataFrame) -> Split:
    """split the dataset into train and test"""
    clf_type = config["model"]["type"]  # 'binary' or 'multiclass'
    label_cols = ["multi_label"]
    data = data.reset_index(drop=True)

    if "label" in data.columns:
        label_cols.append("label")

    if clf_type == "binary":
        y = data["label"]

    elif clf_type == "multiclass":
        y = data["multi_label"]
        # Transforming non numerical labels into numerical labels
        encoder = preprocessing.LabelEncoder()
        encoder.fit(y)
        classes_file = Path(config["data_dir"]) / config["predict"]["classes_file"]
        np.save(classes_file, encoder.classes_)
        # encoding train labels
        y = encoder.transform(y)
        y = pd.Series(y)

    X = data.drop(label_cols, axis=1)
    # 80% for training and 20% for eval and test,
    # random_state for reproducibility
    X_train, X_eval_test, y_train, y_eval_test = train_test_split(
        X, y, test_size=0.2, random_state=config["model"]["seed"]
    )
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_eval_test, y_eval_test, test_size=0.50, random_state=config["model"]["seed"]
    )
    return (
        X_train,
        X_eval,
        X_test,
        pd.Series(y_train),
        pd.Series(y_eval),
        pd.Series(y_test),
    )


def filter_minority_classes(df, minor_threshold):
    """filter out the minority classes with threshold value."""
    if config["debug"]:
        minor_threshold = 10
    count_dict = dict(df["multi_label"].value_counts())
    selected_classes = [x[0] for x in count_dict.items() if x[1] > minor_threshold]
    df = df[df["multi_label"].isin(selected_classes)]
    return df


def sampling_strategy(X, y, n_samples, t="majority"):
    # undersample if the majority class is greater than mean value
    # reference: https://towardsdatascience.com/how-to-deal-with-imbalanced-multiclass-datasets-in-python-fe0bb3f2b669
    target_classes = ""
    sampling_strategy = {}

    if t == "majority":
        print("\nUndersampling the majority classes...")
        target_classes = y.value_counts() > n_samples
    elif t == "minority":
        print("\nOversampling the majority classes...")
        target_classes = y.value_counts() < n_samples

    tc = target_classes[target_classes == True].index

    for target in tc:
        sampling_strategy[target] = n_samples

    # not applying undersampling to 'benign' class
    # if 'benign' in sampling_strategy:
    sampling_strategy = {i: sampling_strategy[i] for i in sampling_strategy if i != 7}

    print("\t", sampling_strategy)
    return sampling_strategy


def apply_balancer(X, y):
    """
    apply class balancer(s) to equalize the number of samples into difference labels.
    """
    print("-" * 30)
    print("Applying oversampling class balancer SMOTE...")
    print("-" * 30)
    count = pd.Series(Counter(y))
    n_samples = count.median().astype(np.int64) * 3
    print("labels's count: \n", count)

    under_sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy(X, pd.Series(y), n_samples, t="majority"),
        random_state=41,
    )
    X_under, y_under = under_sampler.fit_resample(X, y)

    over_sampler = SMOTE(
        sampling_strategy=sampling_strategy(
            X_under, pd.Series(y_under), n_samples, t="minority"
        ),
        random_state=41,
    )
    X_bal, y_bal = over_sampler.fit_resample(X_under, y_under)
    print("Labels count: ", Counter(y_bal))
    return X_bal, y_bal


def sklearnise_keras(keras_model, warm_start=False):
    # wrap model in keras scikit learn wrapper to use yellowbrick
    return SKLearnClassifier(
        keras_model,
        warm_start=warm_start,
        fit_kwargs={
            "batch_size": config["dnn"]["batch"],
            "epochs": config["dnn"]["epochs"],
            "verbose": config["dnn"]["verbose"],
        },
    )


def dynamic_model(X, y):
    from keras.api.optimizers import Adam, SGD
    from src.models import create_DNN, create_LSTM

    input_size = X.shape[1]
    output_size = y.shape[1] if len(y.shape) > 1 else 1
    print(f"{input_size=}, {output_size=}")
    metrics = ["accuracy", "Precision", "Recall"]

    optim = None
    if config["dnn"]["optimizer"] == "adam":
        optim = Adam(learning_rate=config["dnn"]["lr"])
    elif config["dnn"]["optimizer"] == "sgd":
        optim = SGD(learning_rate=config["dnn"]["lr"])
    else:
        optim = config["dnn"]["optimizer"]

    # Setup the model
    model = None
    match config["model"]["name"].lower():
        case "dnn":
            model = create_DNN(input_size, output_size)
        case "lstm":
            model = create_LSTM(input_size, output_size, config["dnn"]["dropout"])
        case _:
            raise ValueError("Model type required")

    model.compile(optimizer=optim, loss=config["dnn"]["loss"], metrics=metrics)
    # Lets save our current model state so we can reload it later
    model.save_weights(str(config["model"]["path"] / "pre-fit.weights.h5"))

    model.summary()

    return model


def load_dnn():
    model = sklearnise_keras(dynamic_model)

    return model


def train_dnn(nt_run, model, X_train, y_train, X_test, y_test):
    """train the ML model"""
    import tensorflow as tf

    # model_path = config['result_dir'] + config['model']['name'] \
    #              + '-' + str(config['dnn']['epochs']) + '/'

    output_size = len(set(list(y_train)))

    print()
    print(f"{output_size=}")
    print(f"{y_train=}")
    print(f"{y_test=}")
    print(f"{X_train.shape=}")

    X_train = X_train.values.astype(float)
    X_test = X_test.values.astype(float)

    # y_train_ndry = tf.keras.utils.to_categorical(x=y_train, num_classes=output_size)
    y_test_ndry = tf.keras.utils.to_categorical(x=y_test, num_classes=output_size)

    model.fit_kwargs["validation_data"] = (X_test, y_test_ndry)
    model.fit_kwargs["callbacks"] = tf_callbacks

    # TODO correct categorizing of val_labels
    # print()
    # print(f"{y_train_ndry[:2]=}")
    # print(f"{y_test_ndry[:2]=}")
    print()
    print(f"{X_train[:2]=}")
    print("-" * 70)
    # multi-level class weights
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights)}
    print("Class_weights: ", class_weights)

    # fit the model
    model.fit(X_train, y_train)
    # print(model.summary())

    # return model, model.history_


def score_predict(model, X, y):
    """
    evaluate the model
    returns performance scores of the model
    """
    y_pred = model.predict(X)
    acc = metrics.accuracy_score(y, y_pred)
    pre = metrics.precision_score(y, y_pred, average="macro")
    rec = metrics.recall_score(y, y_pred, average="macro")
    f1 = metrics.f1_score(y, y_pred, average="macro")
    if config["model"]["name"].lower() != "dnn":
        loss = metrics.log_loss(y, model.predict_proba(X))
        print("Loss: ", loss)

    return acc, pre, rec, f1


def train_shallow(nt_run, model, X_train, y_train, X_test, y_test):
    """training of non-dnn models"""
    model.fit(X_train, y_train)
    # print("+" * 70)
    # train_metrics = score_predict(model, X_train, y_train)
    # print("\nTrain metrics (acc, pre, rec): ", train_metrics)
    # eval_metrics = score_predict(model, X_test, y_test)
    # print("\nTest metrics (acc, pre, rec): ", eval_metrics)
    # print("+" * 70)

    # y_pred = model.predict(X_test)
    # # encoder = preprocessing.LabelEncoder() # need to check
    # # y_pred_label = list(encoder.inverse_transform(y_pred))
    # # y_test_label = list(encoder.inverse_transform(y_test))

    # classes_file = Path(config["data_dir"]) / config["predict"]["classes_file"]
    # encoder = preprocessing.LabelEncoder()  # need to check
    # encoder.classes_ = np.load(classes_file, allow_pickle=True)

    # # y_pred_index = [np.argmax(y, axis=None, out=None) for y in y_pred]
    # # y_pred_label = list(encoder.inverse_transform(y_pred_index))
    # # y_test_index = [np.argmax(y, axis=None, out=None) for y in y_test]
    # # y_test_label = list(encoder.inverse_transform(y_test_index))
    # y_pred_label = list(encoder.inverse_transform(y_pred))
    # y_test_label = list(encoder.inverse_transform(y_test))

    # clf_matrix = metrics.confusion_matrix(y_true=y_test_label, y_pred=y_pred_label)
    # clf_report = metrics.classification_report(
    #     y_true=y_test_label, y_pred=y_pred_label, zero_division=0
    # )  # output_dict=True

    # return clf_matrix, clf_report


def model_train(nt_run, data: Split):
    """Train and test the model using the training data"""
    model_name = config["model"]["name"]
    # print('y_labels on total data: ', set(list(data.multi_label)))
    # X_train, X_test, y_train, y_test = split_data(data, config)
    X_train, X_eval, _, y_train, y_eval, _ = data

    # apply class balancer(s) if that is enabled at config.yaml
    if config["apply_balancer"]:
        X_train, y_train = apply_balancer(X=X_train, y=y_train)

    if config["model"]["use_neptune"]:
        metrics2.upload_df(nt_run, "X_train_balance", X_train.head(20))
        metrics2.upload_df(nt_run, "y_train_balance", y_train.head(20))

    match model_name.lower():
        case "svm" | "svc":
            model = SVC(kernel="linear", probability=True, verbose=True)
        case "lsvm" | "lsvc":
            model = LinearSVC(verbose=True)
            model = CalibratedClassifierCV(model)
        case "rf" | "randomforest":
            model = RandomForestClassifier(n_jobs=-1, verbose=1)
        case "dt" | "decisiontree":
            model = DecisionTreeClassifier()
        case "gb" | "gradientboosting":
            model = GradientBoostingClassifier(n_iter_no_change=10, verbose=1)
        case "hgb" | "histgradientboosting":
            model = HistGradientBoostingClassifier(verbose=1)
        case "nb" | "naivebayes":
            model = GaussianNB()
        case "basic-dnn" | "dnn":
            model = load_dnn()
        case _:
            print(
                "\nModelSelectionError, please select the right model! Not found: ",
                model_name,
            )
            exit(1)

    # plot_curves(config, trained_model, X_train, y_train, cv=3, return_times=True)
    print("\n\n" + "#" * 70)
    print("Training model: ", model_name)
    print("#" * 70)
    if model_name.lower() != "dnn":
        train_shallow(nt_run, model, X_train, y_train, X_eval, y_eval)
        # clf_matrix, clf_report = train_shallow(
        #     nt_run, model, X_train, y_train, X_eval, y_eval
        # )
        # print("\n Classification Matrix (max): \n", clf_matrix)
        # print("\n Classification Report (max): \n", clf_report)
    else:
        train_dnn(nt_run, model, X_train, y_train, X_eval, y_eval)
        # model, history = train_dnn(nt_run, model, X_train, y_train, X_eval, y_eval)
        # # Plot the training history
        # plot_history(config["model"]["path"] + "learnig_curves", history)

    return model


def test_model(model, X_test, y_test, output_size):
    """test the trained model with testing data."""
    import tensorflow as tf

    print("\n" + "-" * 35 + "Testing" + "-" * 35)

    # Generate generalization metrics

    X_test = X_test.values.astype(float)
    y_test = tf.keras.utils.to_categorical(x=y_test, num_classes=output_size)

    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {score[0]} / Test accuracy: {score[1]}")
    print(f"{score=}")
    print("\n" + "-" * 35 + "Testing Completed" + "-" * 35 + "\n")


if __name__ == "__main__":
    # Command Line Arguments:
    parser = argparse.ArgumentParser(
        description="AI-enabled Cybersecurity for malware detection..."
    )
    parser.add_argument("--model", type=str, help="Name of the model to train/test.")
    parser.add_argument("--data", type=str, help="Data file for train/test.")
    paras = parser.parse_args()

    # Config File Arguments:
    config = load_config("config.yaml")
    # data_file = config['data_dir'] + config['data_file']
    data_file = paras.data if paras.data else config["data_dir"] + config["data_file"]

    minor_threshold = int(config["minority_threshold"])

    # pick model name from command line arg otherwise from config file.
    config["model"]["name"] = paras.model if paras.model else config["model"]["name"]

    match config["model"]["name"].lower():
        case "basic-dnn" | "dnn":
            config["model"]["path"] = Path(config["result_dir"]) / (
                f"{config['model']['name']}-{config['dnn']['epochs']}-"
                + f"{config['dnn']['batch']}-{config['dnn']['lr']}-{Path(data_file).stem}"
            )
        case _:
            config["model"]["path"] = Path(config["result_dir"]) / (
                f"{config['model']['name']}-{Path(data_file).stem}"
            )

    model_file = config["model"]["path"] / "model-final.h5"
    print(f"\n\nModel path: {config['model']['path']}")

    if config["model"]["use_neptune"]:
        print("\n" + "-" * 30 + "Neptune" + "-" * 30 + "\n")
        tags = [
            config["model"]["name"].lower(),
            f"data={Path(data_file).stem}",
            f"balancer={config['apply_balancer']}",
        ]
        match config["model"]["name"].lower():
            case "basic-dnn" | "dnn":
                tags.extend(
                    [
                        f"epochs={config['dnn']['epochs']}",
                        f"batch={config['dnn']['batch']}",
                        f"learning_rate={config['dnn']['lr']}",
                    ]
                )
        nt_run = init_neptune(tags)
    else:
        nt_run = None

    if config["debug"]:
        config["model"]["path"] = config["model"]["path"].rsplit("/", 1)[0] + "-debug/"

    if config["train"]:
        Path(config["model"]["path"]).mkdir(parents=True, exist_ok=True)

    # ==============
    # MONKEYPATCHING
    # ==============

    from keras.api.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

    tf_callbacks = [
        EarlyStopping(
            patience=config["dnn"]["patience"],
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(config["model"]["path"] / "checkpoint_model.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        TensorBoard(log_dir=str(config["model"]["path"] / "logs")),
    ]

    if config["model"]["use_neptune"]:
        from neptune.integrations.tensorflow_keras import NeptuneCallback

        # print('\n' + '-' * 30 + 'Neptune' + '-' * 30 + '\n')
        # nt_run = init_neptune(config['model']['path'])
        neptune_cbk = NeptuneCallback(run=nt_run, base_namespace="metrics/keras")
        tf_callbacks.append(neptune_cbk)

    # monkeypatch SKLearnClassifier to not save fit_kwargs["callbacks"]
    old_getstate = SKLearnClassifier.__getstate__

    def __getstate__(self, *k, **kw):
        old_getstate(self, *k, **kw)
        state = self.__dict__.copy()
        if "fit_kwargs" in state and "callbacks" in state["fit_kwargs"]:
            del state["fit_kwargs"]["callbacks"]
        return state

    old_setstate = SKLearnClassifier.__setstate__

    def __setstate__(self, state, *k, **kw):
        old_setstate(self, state, *k, **kw)
        self.__dict__.update(state)
        state["fit_kwargs"]["callbacks"] = tf_callbacks

    SKLearnClassifier.__getstate__ = __getstate__
    SKLearnClassifier.__setstate__ = __setstate__

    # add predict_proba to SKLearnClassifier
    def predict_proba(self, X):
        import sklearn
        from keras.src.wrappers.fixes import _validate_data

        sklearn.base.check_is_fitted(self)
        X = _validate_data(self, X, reset=False)
        return self.model_.predict(X)

    SKLearnClassifier.predict_proba = predict_proba

    # ====
    # MAIN
    # ====

    # loading the data from the processed csv file for training/testing
    df = load_data(data_file)
    if "label" in df.columns:
        df = df.drop(columns=["label"], axis=1)
    if config["model"]["use_neptune"]:
        metrics2.upload_df(nt_run, "data", df.head(20))

    if minor_threshold >= 0:
        df = filter_minority_classes(df, minor_threshold)

    X_train, X_eval, X_test, y_train, y_eval, y_test = data = split_data(df)

    if config["model"]["use_neptune"]:
        metrics2.upload_df(nt_run, "X_train", X_train.head(20))
        metrics2.upload_df(nt_run, "y_train", y_train.head(20))
        metrics2.upload_df(nt_run, "X_eval", X_eval.head(20))
        metrics2.upload_df(nt_run, "y_eval", y_eval.head(20))
        metrics2.upload_df(nt_run, "X_test", X_test.head(20))
        metrics2.upload_df(nt_run, "y_test", y_test.head(20))

    if config["train"]:
        # use_gpu = False
        if config["model"]["name"] == "dnn":
            utilize_gpu()
            print("\n" + "-" * 35 + "Training" + "-" * 35)
            trained_model = model_train(nt_run, data)
        else:
            import joblib

            with joblib.parallel_config(backend="threading", n_jobs=-1):
                trained_model = model_train(nt_run, data)

        # save the trained model as a pickle file
        if config["model"]["save"]:
            try:
                # if config["model"]["name"] == "dnn":
                #     trained_model.model_.save(model_file)
                # else:
                #     pickle.dump(trained_model, open(model_file, "wb"))
                trained_model.fit_kwargs["callbacks"] = []
                pickle.dump(trained_model, open(model_file, "wb"))
                trained_model.fit_kwargs["callbacks"] = tf_callbacks
                print("The final trained model is saved at: ", model_file)
            except pickle.PicklingError:
                print("Failed to save the trained model")

        print("\n" + "-" * 35 + "Training Completed" + "-" * 35 + "\n")
    else:
        from keras.api.models import load_model

        print("Used the trained model saved at: ", model_file)
        # if config["model"]["name"] == "dnn":
        #     assert (
        #         model_file.exists()
        #     ), f"The trained model does not exist for the prediction at: {model_file}"
        #     trained_model = load_model(model_file)
        #     trained_model = sklearnise_keras(trained_model)
        #     # NOTE: initialize doesnt exist for this new keras wrapper
        # else:
        #     trained_model = pickle.load(open(model_file, "rb"))

        # I think the keras wrapper doesnt support saving only the keras model
        # and reinitialising the wrapper
        trained_model = pickle.load(open(model_file, "rb"))

    print("+" * 70)
    train_metrics = score_predict(trained_model, X_train, y_train)
    print("\nTrain metrics (acc, pre, rec): ", train_metrics)
    eval_metrics = score_predict(trained_model, X_test, y_test)
    print("\nTest metrics (acc, pre, rec): ", eval_metrics)
    print("+" * 70)

    classes_file = Path(config["data_dir"]) / config["predict"]["classes_file"]
    encoder = preprocessing.LabelEncoder()  # need to check
    encoder.classes_ = np.load(classes_file, allow_pickle=True)

    if config["model"]["use_neptune"]:
        if config["model"]["name"] == "dnn":
            n_jobs = 1
        else:
            n_jobs = -1
        metrics2.log_metrics(
            nt_run,
            trained_model,
            encoder,
            X_train,
            X_eval,
            y_train,
            y_eval,
            train_metrics,
            eval_metrics,
            n_jobs=n_jobs,
        )

    if config["test"]:
        output_size = len(set(list(y_train)))

        # if config["model"]["name"] == "dnn":
        #     test_model(trained_model, X_test, y_test, output_size)
        # else:
        result = trained_model.score(X_test, y_test)
        print("result: ", result)
        print("\n" + "-" * 35 + "Testing Completed" + "-" * 35 + "\n")
