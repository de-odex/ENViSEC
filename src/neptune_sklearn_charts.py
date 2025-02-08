#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import matplotlib.pyplot as plt
import pandas as pd
from scikitplot.estimators import plot_learning_curve
from scikitplot.metrics import plot_precision_recall
from sklearn.base import (
    is_classifier,
    is_regressor,
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from yellowbrick.classifier import (
    ROCAUC,
    ClassificationReport,
    ClassPredictionError,
    ConfusionMatrix,
    PrecisionRecallCurve,
)
from yellowbrick.cluster import (
    KElbowVisualizer,
    SilhouetteVisualizer,
)
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.regressor import (
    CooksDistance,
    PredictionError,
    ResidualsPlot,
)

from neptune_sklearn.impl.version import __version__

try:
    from neptune.types import (
        File,
        FileSeries,
    )
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new.types import (
        File,
        FileSeries,
    )
    from neptune.new.utils import stringify_unsupported

from copy import deepcopy
from warnings import warn


def create_learning_curve_chart(regressor, X_train, y_train):
    """Creates learning curve chart.

    Args:
        regressor (`regressor`): Fitted scikit-learn regressor object.
        X_train (`ndarray`): Training data matrix.
        y_train (`ndarray`): The regression target for training.

    Returns:

        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.new.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/learning_curve"] = npt_utils.create_learning_curve_chart(rfr, X_train, y_train)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        plot_learning_curve(regressor, X_train, y_train, ax=ax)

        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log learning curve chart. Error: {e}")

    return chart


def create_feature_importance_chart(regressor, X_train, y_train):
    """Creates feature importance chart.

    Args:
        regressor (`regressor`): Fitted scikit-learn regressor object.
        X_train (`ndarray`): Training data matrix.
        y_train (`ndarray`): The regression target for training.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/feature_importance"] = npt_utils.create_feature_importance_chart(rfr, X_train, y_train)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = FeatureImportances(deepcopy(regressor), is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.finalize()

        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log feature importance chart. Error: {e}")

    return chart


def create_residuals_chart(regressor, X_train, X_test, y_train, y_test):
    """Creates residuals chart.

    Args:
        regressor (`regressor`): Fitted scikit-learn regressor object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The regression target for training.
        y_test (`ndarray`): The regression target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.new.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/residuals"] = npt_utils.create_residuals_chart(rfr, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ResidualsPlot(regressor, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log residuals chart. Error: {e}")

    return chart


def create_prediction_error_chart(regressor, X_train, X_test, y_train, y_test):
    """Creates prediction error chart.

    Args:
        regressor (`regressor`): Fitted sklearn regressor object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The regression target for training.
        y_test (`ndarray`): The regression target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.new.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["prediction_error"] = npt_utils.create_prediction_error_chart(rfr, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = PredictionError(regressor, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log prediction error chart. Error: {e}")

    return chart


def _monkey_draw(self):
    """
    Monkey patches `yellowbrick.regressor.CooksDistance.draw()`
    to remove unsupported matplotlib argument `use_line_collection`.

    Draws a stem plot where each stem is the Cook's Distance of the instance at the
    index specified by the x axis. Optionaly draws a threshold line.
    """
    # Draw a stem plot with the influence for each instance
    _, _, baseline = self.ax.stem(
        self.distance_,
        linefmt=self.linefmt,
        markerfmt=self.markerfmt,
        # use_line_collection=True
    )

    # No padding on either side of the instance index
    self.ax.set_xlim(0, len(self.distance_))

    # Draw the threshold for most influential points
    if self.draw_threshold:
        label = r"{:0.2f}% > $I_t$ ($I_t=\frac {{4}} {{n}}$)".format(
            self.outlier_percentage_
        )
        self.ax.axhline(
            self.influence_threshold_,
            ls="--",
            label=label,
            c=baseline.get_color(),
            lw=baseline.get_linewidth(),
        )

    return self.ax


CooksDistance.draw = _monkey_draw


def create_cooks_distance_chart(regressor, X_train, y_train):
    """Creates cooks distance chart.

    Args:
        regressor (`regressor`): Fitted sklearn regressor object
        X_train (`ndarray`): Training data matrix
        y_train (`ndarray`): The regression target for training

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/cooks_distance"] = npt_utils.create_cooks_distance_chart(rfr, X_train, y_train)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = CooksDistance(ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log cooks distance chart. Error: {e}")

    return chart


def create_classification_report_chart(
    classifier, X_train, X_test, y_train, y_test, classes=None
):
    """Creates classification report chart.

    Args:
        classifier (`classifier`): Fitted sklearn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/classification_report"] = npt_utils.create_classification_report_chart(
            rfc, X_train, X_test, y_train, y_test
        )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ClassificationReport(
            classifier, support=True, is_fitted=True, ax=ax, classes=classes
        )
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log Classification Report chart. Error: {e}")

    return chart


def create_confusion_matrix_chart(
    classifier, X_train, X_test, y_train, y_test, classes=None
):
    """Creates confusion matrix.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/confusion_matrix"] = npt_utils.create_confusion_matrix_chart(
            rfc, X_train, X_test, y_train, y_test
        )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ConfusionMatrix(classifier, is_fitted=True, ax=ax, classes=classes)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log Confusion Matrix chart. Error: {e}")

    return chart


def create_roc_auc_chart(classifier, X_train, X_test, y_train, y_test, classes=None):
    """Creates ROC-AUC chart.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/roc_auc"] = npt_utils.create_roc_auc_chart(rfc, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ROCAUC(classifier, is_fitted=True, ax=ax, classes=classes)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log ROC-AUC chart. Error {e}")

    return chart


def create_precision_recall_chart(
    classifier, X_train, X_test, y_train, y_test, y_pred_proba=None, classes=None
):
    """Creates precision-recall chart.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_test (`ndarray`): Testing data matrix.
        y_test (`ndarray`): The classification target for testing.
        y_pred_proba (`ndarray`, optional): Classifier predictions probabilities on test data.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/precision_recall"] = npt_utils.create_precision_recall_chart(rfc, X_test, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    # if y_pred_proba is None:
    #     try:
    #         y_pred_proba = classifier.predict_proba(X_test)
    #     except Exception as e:
    #         warn(
    #             f"""Did not log Precision-Recall chart: this classifier does not provide predictions probabilities.
    #             Error {e}"""
    #         )
    #         return chart

    # try:
    #     fig, ax = plt.subplots()
    #     plot_precision_recall(y_test, y_pred_proba, ax=ax)
    #     chart = File.as_image(fig)
    #     plt.close(fig)
    # except Exception as e:
    #     warn(f"Did not log Precision-Recall chart. Error {e}")

    try:
        fig, ax = plt.subplots()
        visualizer = PrecisionRecallCurve(
            classifier,
            # is_fitted=True,
            ax=ax,
            classes=classes,
            iso_f1_curves=True,
            micro=False,
            per_class=True,
        )
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log Precision-Recall chart. Error {e}")

    return chart


def create_class_prediction_error_chart(
    classifier, X_train, X_test, y_train, y_test, classes=None
):
    """Creates class prediction error chart.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/class_prediction_error"] = npt_utils.create_class_prediction_error_chart(
            rfc, X_train, X_test, y_train, y_test
        )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ClassPredictionError(
            classifier, is_fitted=True, ax=ax, classes=classes
        )
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log Class Prediction Error chart. Error {e}")

    return chart


def get_cluster_labels(model, X, nrows=1000, **kwargs):
    """Logs the index of the cluster label each sample belongs to.

    Args:
        model (`KMeans`): KMeans object.
        X (`ndarray`): Training instances to cluster.
        nrows (`int`, optional): Number of rows to log.
        kwargs: KMeans parameters.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        km = KMeans(n_init=11, max_iter=270)
        X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

        run = neptune.init_run()
        run["kmeans/cluster_labels"] = npt_utils.get_cluster_labels(km, X)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(model, KMeans), "Model should be sklearn KMeans instance."
    assert isinstance(nrows, int), "nrows should be integer, {} was passed".format(
        type(nrows)
    )

    model.set_params(**kwargs)
    labels = model.fit_predict(X)
    df = pd.DataFrame(data={"cluster_labels": labels})
    df = df.head(n=nrows)

    return File.as_html(df)


def create_kelbow_chart(model, X, **kwargs):
    """Creates K-elbow chart for KMeans clusterer.

    Args:
        model (`KMeans`): KMeans object.
        X (`ndarray`): Training instances to cluster.
        kwargs: KMeans parameters.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        km = KMeans(n_init=11, max_iter=270)
        X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

        run = neptune.init_run()
        run["kmeans/kelbow"] = npt_utils.create_kelbow_chart(km, X)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(model, KMeans), "Model should be sklearn KMeans instance."

    chart = None

    model.set_params(**kwargs)

    if "n_clusters" in kwargs:
        k = kwargs["n_clusters"]
    else:
        k = 10

    try:
        fig, ax = plt.subplots()
        visualizer = KElbowVisualizer(model, k=k, ax=ax)
        visualizer.fit(X)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        warn(f"Did not log KMeans elbow chart. Error {e}")

    return chart


def create_silhouette_chart(model, X, **kwargs):
    """Creates silhouette coefficients charts for KMeans clusterer.

    Charts are computed for j = 2, 3, ..., n_clusters.

    Args:
        model (`KMeans`): KMeans object.
        X (`ndarray`): Training instances to cluster.
        kwargs: KMeans parameters.

    Returns:
        `neptune.types.FileSeries` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        km = KMeans(n_init=11, max_iter=270)
        X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

        run = neptune.init_run()
        run["kmeans/silhouette"] = npt_utils.create_silhouette_chart(km, X, n_clusters=12)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(model, KMeans), "Model should be sklearn KMeans instance."

    charts = []

    model.set_params(**kwargs)

    n_clusters = model.get_params()["n_clusters"]

    for j in range(2, n_clusters + 1):
        model.set_params(**{"n_clusters": j})
        model.fit(X)

        try:
            fig, ax = plt.subplots()
            visualizer = SilhouetteVisualizer(model, is_fitted=True, ax=ax)
            visualizer.fit(X)
            visualizer.finalize()
            charts.append(File.as_image(fig))
            plt.close(fig)
        except Exception as e:
            warn(f"Did not log Silhouette Coefficients chart. Error {e}")

    return FileSeries(charts)
