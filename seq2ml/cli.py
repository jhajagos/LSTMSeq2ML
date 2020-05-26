"""Command-line interface to train and evaulate models."""

import json
import os
import pprint
import sys

import click
import h5py
import sklearn.metrics
import tensorflow as tf

from .models import get_model

tfk = tf.keras


class JSONParamType(click.ParamType):
    name = "json"

    def convert(self, value, param, ctx):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            self.fail("%s is not valid JSON" % value, param, ctx)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--filepath", "-f", type=click.Path(exists=True), required=True)
@click.option("--n-cut", "-n", type=int, required=True, default=25)
def popular(*, filepath, n_cut):
    """Print the most popular labels in the dataset."""

    with h5py.File(filepath, mode="r") as f:
        x_train_len = f["/data/processed/train/sequence/core_array"].shape[0]
        y_train_all = f["/data/processed/train/target/core_array"][:]
        y_train_labels = f["/data/processed/train/target/column_annotations"][:]

    # Count the occurrence of the labels in the training set, and display the most
    # common ones.
    target_labels = [str(s, "utf-8") for s in y_train_labels.flatten().tolist()]
    sum_targets = y_train_all.sum(0).tolist()
    sum_target_with_labels = [
        (target_labels[i].split("|")[-1], sum_targets[i], sum_targets[i] / x_train_len)
        for i, _ in enumerate(target_labels)
    ]
    sum_target_with_labels.sort(key=lambda x: -1 * x[1])

    print("The top {} variables in the training set.".format(n_cut))
    _s = "{:<45} {:>10} {:>10}".format("Label", "N", "Ratio")
    print()
    print(_s)
    print("-" * len(_s))
    for row in sum_target_with_labels[:n_cut]:
        print("{:<45} {:>10} {:>10.3f}".format(*row))


@cli.command()
@click.option("--filepath", "-f", type=click.Path(exists=True), required=True)
@click.option("--target-name", "-t", required=True)
@click.option("--model-name", "-m", required=True)
@click.option("--batch-size", "-b", default=48, show_default=True)
@click.option("--epochs", "-e", default=1, show_default=True)
@click.option("--learning-rate", "-l", default=1e-3, show_default=True)
@click.option("--evaluate/--no-evaluate", default=True, show_default=True)
@click.option("--model-kwds", type=JSONParamType(), show_default=True)
def train(
    *,
    filepath,
    target_name,
    model_name,
    batch_size,
    epochs,
    learning_rate,
    evaluate,
    model_kwds
):
    """Train an recurrent network."""

    print("Training to predict '{}'.".format(target_name), flush=True)

    # Try to get the model first so if it is not available, we error quickly.
    model_fn = get_model(model_name)
    model = model_fn() if model_kwds is None else model_fn(**model_kwds)

    # Find the target index for the labels.
    with h5py.File(filepath, mode="r") as f:
        y_train_labels = f["/data/processed/train/target/column_annotations"][:]

    target_prefix = "static_condition_hierarchy_condition_concept_name|"
    target_index_name = target_prefix + target_name
    target_labels = [str(s, "utf-8") for s in y_train_labels.flatten().tolist()]
    target_index = target_labels.index(target_index_name)

    # Load train/test data.
    with h5py.File(filepath, mode="r") as f:
        x_train = f["/data/processed/train/sequence/core_array"][:]
        y_train = f["/data/processed/train/target/core_array"][:, target_index]
        x_test = f["/data/processed/test/sequence/core_array"][:]
        y_test = f["/data/processed/test/target/core_array"][:, target_index]

    model.compile(
        loss=tfk.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tfk.optimizers.Adam(learning_rate),
        metrics=["accuracy"],
    )

    target_name_label = "_".join(target_name.lower().split())

    # TODO: add callbacks
    callbacks = []

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        # validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    if evaluate:
        y_pred = model.predict(x_test).ravel()
        y_pred = tf.sigmoid(y_pred).numpy()

        threshold = 0.5
        y_pred_classes = (y_pred > threshold).astype("int32")

        results_dict = {
            "meta": {"python_program": os.path.abspath(__file__)},
            "model": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            },
            "target": {
                "target_name": target_name,
                "target_name_label": target_name_label,
            },
            "data": {
                "input_filename": filepath,
                "training_size_n": x_train.shape[0],
                "max_time_steps_n": x_train.shape[1],
                "features_n": x_train.shape[2],
                "total_positive_cases_training_set": y_train.sum(),
            },
            "test": {
                "total_positive_cases_test_set": y_test.sum(),
                "sum_predicted_test_set": y_pred_classes.sum(),
                "sum_of_probabilities_test_set": y_pred.sum(),
                "model_auc_score": sklearn.metrics.roc_auc_score(y_test, y_pred),
                "f1_score": sklearn.metrics.f1_score(y_pred_classes, y_test),
                "average_precision_recall": sklearn.metrics.average_precision_score(
                    y_pred_classes, y_test
                ),
                "classification_report": sklearn.metrics.classification_report(
                    y_pred_classes, y_test
                ),
            },
        }

        pprint.pprint(results_dict)

        # TODO: save results to JSON.