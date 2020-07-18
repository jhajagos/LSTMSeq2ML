"""Command-line interface to train and evaulate models."""

import contextlib
import csv
import datetime
import json
import os
from pathlib import Path
import pprint
import re
import tempfile

import click
import h5py
import numpy as np
import sklearn.metrics
import tensorflow as tf

from .models import get_model_fn

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
@click.option(
    "--keyword",
    "-k",
    multiple=True,
    help="Case-insensitive, can be used multiple times.",
)
@click.option("--regex", "-r", help="Regular expression.")
@click.option(
    "--n-cut",
    "-n",
    type=int,
    default=25,
    show_default=True,
    help="Show this many matches maximum.",
)
@click.option(
    "--split-char",
    "-s",
    default="|",
    show_default=True,
    help="Character used to split prefix and label.",
)
def search(*, filepath, keyword, regex, n_cut, split_char):
    """Search target labels in the dataset using keywords or regular expression.

    Multiple keywords or a regular expression can be supplied. If multiple keywords are
    supplied, the union of matching labels is returned. Case-insensitive.

    Examples:

        seq2ml search -f path/to/data.hdf5 -k ventilator -k alveolar
        seq2ml search -f path/to/data.hdf5 -r '(\bVentilator\b|\bAlveolar\b)'
    """

    # keyword is a tuple; renamed here to be more descriptive.
    keywords = keyword

    if not (keywords or regex) or (keywords and regex):
        raise click.UsageError(
            "-k/--keyword or -r/--regex must be supplied, but not both."
        )

    # Check that the regular expression is valid.
    if regex:
        try:
            prog = re.compile(regex, flags=re.IGNORECASE)
        except re.error as err:
            raise click.BadParameter(err, param_hint="-r/--regex")

    with h5py.File(filepath, mode="r") as f:
        y_train_labels = f["/data/processed/train/target/column_annotations"][:]

    y_train_labels = y_train_labels.flatten().tolist()
    y_train_labels = list(map(bytes.decode, y_train_labels))

    matches = []

    if keywords:
        keywords_lower = list(map(str.lower, keywords))
        y_train_labels_lower = list(map(str.lower, y_train_labels))
        for jj, label in enumerate(y_train_labels_lower):
            for keyword in keywords_lower:
                if keyword in label:
                    # Add the original label (not the lower-case one).
                    match = y_train_labels[jj]
                    if split_char:
                        match = match.split(split_char)[-1]
                    matches.append(match)
        header = "keyword(s) '{}'".format("', '".join(keywords))

    elif regex:
        for label in y_train_labels:
            if prog.search(label) is not None:
                match = label
                if split_char:
                    match = match.split(split_char)[-1]
                matches.append(match)
        header = "regular expression '{}'".format(regex)

    print("\nThe first {} matches for {}".format(n_cut, header))
    print()
    print("Label")
    print("-----")
    print("\n".join(matches[:n_cut]))
    print()


@cli.command()
@click.option("--filepath", "-f", type=click.Path(exists=True), required=True)
@click.option("--n-cut", "-n", type=int, default=25, show_default=True)
def popular(*, filepath, n_cut):
    """Print the most popular labels in the dataset."""

    print("Loading hdf5 data")
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
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Default is temporary directory",
    required=False,
)
@click.option("--batch-size", "-b", default=48, show_default=True)
@click.option("--epochs", "-e", default=1, show_default=True)
@click.option("--learning-rate", "-l", default=1e-3, show_default=True)
@click.option("--save-history/--no-save-history", default=True, show_default=True)
@click.option("--evaluate/--no-evaluate", default=True, show_default=True)
@click.option("--model-kwds", type=JSONParamType(), default=None, show_default=True)
@click.option("--early-stopping/--no-early-stopping", default=True, show_default=True)
@click.option(
    "--early-stopping-kwds", type=JSONParamType(), default=None, show_default=True
)
@click.option(
    "--model-checkpoint/--no-model-checkpoint", default=True, show_default=True
)
@click.option(
    "--model-checkpoint-kwds", type=JSONParamType(), default=None, show_default=True
)
def train(
    *,
    filepath,
    target_name,
    model_name,
    output_dir,
    batch_size,
    epochs,
    learning_rate,
    save_history,
    evaluate,
    model_kwds,
    early_stopping,
    early_stopping_kwds,
    model_checkpoint,
    model_checkpoint_kwds
):
    """Train a recurrent network."""

    # Get the timestamp for use later when saving outputs.
    timestamp = get_timestamp()

    click.secho("Instantiating model", fg="yellow")
    # Try to get the model first so if it is not available, we error quickly.
    model_fn = get_model_fn(model_name)

    # MirroredStrategy for multi-gpu single-machine training.
    strategy = tf.distribute.MirroredStrategy()

    # cuDNN GRU implementation is not used with MirroredStrategy, so use a mock context
    # if using a single GPU. See https://github.com/tensorflow/tensorflow/issues/40421
    if strategy.num_replicas_in_sync < 2:
        strategy = NullStrategy()

    with strategy.scope():
        model = model_fn() if model_kwds is None else model_fn(**model_kwds)

    click.secho("Loading data", fg="yellow")
    # Find the target index for the labels.
    with h5py.File(filepath, mode="r") as f:
        y_train_labels = f["/data/processed/train/target/column_annotations"][:]
    y_train_labels = list(map(bytes.decode, y_train_labels.flatten().tolist()))

    # Search for target in list.
    indices = []
    for i, label in enumerate(y_train_labels):
        # TODO: is this necessary? This accounts for instances where the target name
        # can match multiple labels, but the user gave the exact target name
        # (minus prefix).
        if target_name == label.split("|")[-1]:
            indices = [i]
            break
        elif target_name.lower() in label.lower():
            indices.append(i)
    if not indices:
        raise ValueError("no labels found for target name '{}'".format(target_name))
    elif len(indices) > 1:
        raise ValueError(
            "{} labels found for target name '{}'".format(len(indices), target_name)
        )

    target_index = indices[0]
    target_name = y_train_labels[target_index]
    click.secho("Training to predict '{}'.".format(target_name), fg="yellow")

    filepath = os.path.abspath(filepath)

    # Load train/test data.
    with h5py.File(filepath, mode="r") as f:
        x_train = f["/data/processed/train/sequence/core_array"][:]
        y_train = f["/data/processed/train/target/core_array"][:, target_index]
        x_test = f["/data/processed/test/sequence/core_array"][:]
        y_test = f["/data/processed/test/target/core_array"][:, target_index]

    click.secho("Compiling model", fg="yellow")
    with strategy.scope():
        model.compile(
            loss=tfk.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tfk.optimizers.Adam(learning_rate),
            metrics=["accuracy"],
        )

    target_name_label = "_".join(target_name.lower().split())

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="seq2ml_{}_".format(timestamp))
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        raise ValueError("output directory '{}' is not a directory".format(output_dir))

    click.secho("Saving outputs to {}".format(output_dir), fg="yellow")
    output_dir = Path(output_dir)

    callbacks = []
    if early_stopping:
        default_early_stopping_kwds = dict(
            monitor="val_loss",
            min_delta=0,
            patience=2,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        default_early_stopping_kwds.update(early_stopping_kwds or {})
        click.secho("Using callback EarlyStopping(**{})".format(default_early_stopping_kwds), fg="yellow")
        callbacks.append(tfk.callbacks.EarlyStopping(**default_early_stopping_kwds))
    if model_checkpoint:
        (output_dir / "weights").mkdir(parents=True, exist_ok=True)
        default_model_checkpoint_kwds = dict(
            filepath=str(output_dir / "weights" / "weights.{epoch:03d}-{val_loss:.4f}.hdf5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        default_model_checkpoint_kwds.update(model_checkpoint_kwds or {})
        click.secho(
            "Using callback ModelCheckpoint(**{})".format(default_model_checkpoint_kwds),
            fg="yellow"
        )
        callbacks.append(tfk.callbacks.ModelCheckpoint(**default_model_checkpoint_kwds))

    click.secho("Training model", fg="yellow")
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    if save_history:
        path = output_dir / "{}_history.json".format(timestamp)
        click.secho("Saving training history to {}".format(path), fg="yellow")
        with open(path, "w") as f:
            json.dump(str(history.history), f)

    results_dict = {
        "meta": {"python_program": os.path.abspath(__file__)},
        "model": {
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "name": str(model_name),
            "keyword_args": {} if model_kwds is None else model_kwds,
        },
        "target": {
            "target_name": target_name,
            "target_name_label": target_name_label,
        },
        "data": {
            "input_filename": os.path.abspath(filepath),
            "training_size_n": int(x_train.shape[0]),
            "max_time_steps_n": int(x_train.shape[1]),
            "features_n": int(x_train.shape[2]),
            "total_positive_cases_training_set": int(y_train.sum()),
        }
    }
    if model_checkpoint:
        files = list((output_dir / "weights").glob("weights*.hdf5"))
        files = list(map(str, files))
        files.sort()
        results_dict["model"]["checkpoints"] = files

    if evaluate:
        click.secho("Evaluating model", fg="yellow")

        y_pred = model.predict(x_test).ravel()
        y_pred = tf.sigmoid(y_pred).numpy()

        threshold = 0.5
        y_pred_classes = (y_pred > threshold).astype("int32")

        results_dict.update({
            "test": {
                "total_positive_cases_test_set": int(y_test.sum()),
                "ratio_positive_cases_test_set": y_test.mean().round(4).astype(float),
                "sum_predicted_test_set": y_pred_classes.sum().astype(float),
                "sum_of_probabilities_test_set": y_pred.sum().astype(float),
                "model_auc_score": sklearn.metrics.roc_auc_score(y_test, y_pred).astype(
                    float
                ),
                "f1_score": sklearn.metrics.f1_score(y_pred_classes, y_test).astype(
                    float
                ),
                "average_precision_recall": sklearn.metrics.average_precision_score(
                    y_pred_classes, y_test
                ).astype(float),
                "classification_report": sklearn.metrics.classification_report(
                    y_pred_classes, y_test
                ),
            },
        })

        # Save predictions to CSV and HDF5.
        with h5py.File(filepath, mode="r") as f:
            identifiers = f["/data/processed/test/identifiers/core_array"][:]
            name_identifiers = (
                f["/data/processed/test/identifiers/column_annotations"][:]
                .flatten()
                .tolist()
            )
        _save_predictions(
            y_test=y_test,
            y_pred=y_pred,
            y_pred_classes=y_pred_classes,
            target_index=target_index,
            identifiers=identifiers,
            name_identifiers=name_identifiers,
            csv_filename=output_dir / "predictions.csv",
            hdf5_filename=output_dir / "predictions.hdf5",
        )

    path = output_dir / "{}_results.json".format(timestamp)
    click.secho("Saving evaluation results to {}".format(path), fg="yellow")
    with open(path, "w") as f:
        json.dump(results_dict, f)

    pprint.pprint(results_dict)

    click.secho("Done. Outputs are saved to {}".format(output_dir), fg="green")


def _save_predictions(
    y_test,
    y_pred,
    y_pred_classes,
    target_index,
    identifiers,
    name_identifiers,
    csv_filename,
    hdf5_filename,
):
    """Save model predictions to CSV and HDF5."""
    # new_sort_order = np.lexsort([np.negative(y_pred)])
    new_sort_order = np.argsort(np.negative(y_pred))
    identifiers = identifiers[new_sort_order]
    y_pred_sorted = y_pred[new_sort_order]
    y_test_sorted = y_test[new_sort_order]
    name_identifiers = [bytes.decode(x) for x in name_identifiers]

    csv_filename = str(csv_filename)
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(name_identifiers + ["position", "probability", "ground_truth"])
        for i in range(identifiers.shape[0]):
            writer.writerow(
                identifiers[i].tolist()
                + [new_sort_order[i], y_pred_sorted[i], y_test_sorted[i]]
            )

    if hdf5_filename.exists():
        raise FileExistsError("{} already exists.".format(hdf5_filename))
    with h5py.File(str(hdf5_filename), "w") as f:
        f.create_dataset(name="/thresholded", data=y_pred_classes, compression="lzf")
        f.create_dataset(name="/probabilities", data=y_pred, compression="lzf")


def get_timestamp():
    """Return a UTC timestamp string like "20200526-164805-UTC"."""
    dt = datetime.datetime.utcnow()
    return dt.strftime("%Y%m%d-%H%M%S-UTC")


# Source copied from
# https://github.com/python/cpython/blob/811e040b6e0241339545c2f055db8259b408802f/Lib/contextlib.py#L686-L704
class nullcontext(contextlib.AbstractContextManager):
    """Context manager that does no additional processing.
    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:
    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


class NullStrategy:
    """A mock of TensorFlow Strategy.scope() that does nothing."""

    scope = nullcontext
