import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Masking, SimpleRNN
from tensorflow.keras.utils import multi_gpu_model

from sklearn.metrics import roc_curve, roc_auc_score, f1_score, classification_report, average_precision_score
import numpy as np
import h5py
import pprint
import argparse
import datetime
import os
import json
import csv


"""
This does a a canonical LSTM model 

The CuDNNLSTM is much faster but it does not support Masking

"""


def main(input_file_name, target_name, output_directory="./", n_cut=25, prediction_threshold=0.5, epochs=10, batch_size=100,
         learning_rate=1e-3, learning_rate_decay=1e-4,
         target_prefix="static_condition_hierarchy_condition_concept_name|",
         cut_sequence=None, number_of_gpus=1
         ):

    target_index_name = target_prefix + target_name

    f5 = h5py.File(input_file_name)

    target_name_label = "_".join(target_name.lower().split())

    f5_train = f5["/data/processed/train/sequence/core_array"]
    f5_target = f5["/data/processed/train/target/core_array"]
    f5_target_labels = f5["/data/processed/train/target/column_annotations"]

    print("Dimensions of training set:")
    print(f5_train.shape)

    training_size_n, time_steps_n, features_n = f5_train.shape

    print("Dimension of full target:")
    print(f5_target.shape)

    target_labels = f5_target_labels[...].ravel().tolist()
    target_labels = [str(s, "utf8") for s in target_labels]

    # Get a count of the most frequent labels
    sum_targets = np.sum(f5_target[...], axis=0).tolist()
    sum_target_with_labels = [(target_labels[i].split("|")[-1], sum_targets[i], sum_targets[i] / f5_train.shape[0])
                              for i in range(len(target_labels))]

    sum_target_with_labels.sort(key=lambda x: -1 * x[1])

    # Let us generate a list of most frequent features to target for prediction
    print("The top %s variables in the training set" % n_cut)
    pprint.pprint(sum_target_with_labels[0:n_cut])
    print("")
    print("Predicting: '%s'" % target_name)

    target_index = target_labels.index(target_index_name)

    f5_test = f5["/data/processed/test/sequence/core_array"]
    f5_test_target = f5["/data/processed/test/target/core_array"]

    print("Dimensions of test set:")
    print(f5_test.shape)
    testing_size_n, _, _ = f5_test.shape

    print("Dimensions of target:")
    print(f5_test_target.shape)

    if cut_sequence is None:
        f5_train_array = f5_train[...]
    else:
        f5_train_array = f5_train[:, 0:cut_sequence, :]

    f5_train_array[np.isnan(f5_train_array)] = 0

    #print(np.sum(f5_train_array))

    if cut_sequence is None:
        f5_test_array = f5_test[...]
    else:
        f5_test_array = f5_test[:, 0:cut_sequence, :]

    f5_test_array[np.isnan(f5_test_array)] = 0

    start_date_time = datetime.datetime.utcnow()

    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # for device in gpu_devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    #TODO: Make models plugin
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=f5_train_array.shape[1:]))
    model.add(GRU(256, activation="tanh", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(256, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)

    if number_of_gpus > 1:
        model = multi_gpu_model(model, gpus=number_of_gpus)
    else:
        pass

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    best_model_file_name = os.path.join(output_directory, "best_model.h5")

    es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)
    mc_callback = ModelCheckpoint(best_model_file_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Fit the model to the data

    # This can be refactored as explicit casts are not needed
    model.fit(np.array(f5_train_array, dtype="float32"), np.array(f5_target[:, target_index], dtype="int32"),
              epochs=epochs,
              validation_data=(np.array(f5_test_array, dtype="float32"),
              np.array(f5_test_target[:, target_index], dtype="int32")), callbacks=[es_callback, mc_callback],
              batch_size=batch_size)

    model.load_weights(best_model_file_name)

    end_date_time = datetime.datetime.utcnow()
    end_label_time_stamp = end_date_time.strftime("%Y%m%d_%H%M")
    start_time_stamp = start_date_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time_stamp = end_date_time.strftime("%Y-%m-%d %H:%M:%S")

    time_lapse = end_date_time - start_date_time

    coefficients_filename = target_name_label + "_" + end_label_time_stamp + "_coeffs.hdf5"
    model_file_name = os.path.join(output_directory, coefficients_filename)
    model.save(model_file_name)

    if number_of_gpus > 1:
        model = tf.keras.models.load_model(model_file_name)

    # Make a probability prediction
    y_pred_keras = model.predict_proba(np.array(f5_test_array[...], dtype="float32")).ravel()

    # Diagnostics
    results_dict = {"meta": {}}
    python_program_run = os.path.abspath(__file__)
    meta_dict = results_dict["meta"]
    meta_dict["python_program_run"] = python_program_run
    meta_dict["start_date_time_utc"] = start_time_stamp
    meta_dict["end_date_time_utc"] = end_time_stamp
    meta_dict["time_lapse"] = str(time_lapse)

    results_dict["model"] = {}
    model_dict = results_dict["model"]

    model_dict["learning_rate"] = learning_rate
    model_dict["learning_rate_decay"] = learning_rate_decay
    model_dict["batch_size"] = batch_size
    model_dict["epochs"] = epochs

    results_dict["target"] = {}
    results_dict["target"]["target_name"] = target_name
    results_dict["target"]["target_name_label"] = target_name_label

    results_dict["data"] = {}
    results_dict["data"]["input_file_name"] = os.path.abspath(input_file_name)

    results_dict["data"]["training_size_n"] = int(training_size_n)
    results_dict["data"]["features_n"] = int(features_n)
    results_dict["data"]["max_time_steps_n"] = int(time_steps_n)

    print("Diagnostics of predictions")

    show_first_n = n_cut

    print("Actual test predictions %s:" % show_first_n)
    print(f5_test_target[0:show_first_n, target_index])

    target_threshold_predictions = np.array(y_pred_keras > prediction_threshold, dtype="float32")
    print("Setting the prediction threshold at %s" % prediction_threshold)
    print(np.array(target_threshold_predictions[0:show_first_n], dtype="int32"))

    print("Probability of first %s predictions from the model:" % show_first_n)
    print(y_pred_keras[0:show_first_n])

    total_positive_cases_test_set = int(np.sum(f5_test_target[:, target_index]))
    print("Total positive cases in test set:")
    print(total_positive_cases_test_set)

    results_dict["test"] = {}
    test_dict = results_dict["test"]
    test_dict["total_positive_cases_test_set"] = int(total_positive_cases_test_set)

    total_positive_cases_training_set = int(np.sum(f5_target[:, target_index]))
    results_dict["data"]["total_positive_cases_training_set"] = total_positive_cases_training_set

    sum_predicted_test_set = int(np.sum(target_threshold_predictions))
    print("Total predicted positive cases in test set:")
    print(sum_predicted_test_set)
    test_dict["sum_predicted_test_set"] = sum_predicted_test_set

    sum_of_probabilities_test_set = np.sum(y_pred_keras)
    print("Total sum of probabilities")
    print(sum_of_probabilities_test_set)
    test_dict["sum_of_probabilities_test_set"] = float(sum_of_probabilities_test_set)

    model_auc_score = roc_auc_score(np.array(f5_test_target[:, target_index], dtype="int32"), y_pred_keras)
    print("Computed AUC of the ROC:")
    print(model_auc_score)
    test_dict["model_auc_score"] = model_auc_score

    f1 = f1_score(np.array(target_threshold_predictions, dtype="int32"), np.array(f5_test_target[:, target_index], dtype="int32"))
    print("F1 score:")
    print(f1)
    test_dict["f1_score"] = f1

    avg_prec_rec = average_precision_score(np.array(target_threshold_predictions, dtype="int32"),
                                           np.array(f5_test_target[:, target_index], dtype="int32"))
    print("Average precision score:")
    print(avg_prec_rec)
    test_dict["average_precision_recall"] = avg_prec_rec

    print("")
    print("Classification report")

    cr_report = classification_report(np.array(target_threshold_predictions, dtype="int32"),
                                np.array(f5_test_target[:, target_index], dtype="int32"))

    print(cr_report)
    test_dict["classification_report"] = cr_report

    prediction_results_base_name = os.path.join(output_directory, "predicted_" + target_name_label + "_" + end_label_time_stamp)
    prediction_results_hdf5_file_name = prediction_results_base_name + ".hdf5"
    prediction_results_csv_file_name = prediction_results_base_name + ".csv"

    new_sort_order = np.lexsort((-1 * (y_pred_keras),))
    
    identifiers = f5["/data/processed/test/identifiers/core_array"][...]
    identifiers = identifiers[new_sort_order]
    sorted_predicted_prob = y_pred_keras[new_sort_order]

    target_sorted = f5_test_target[:, target_index].ravel()[new_sort_order]

    name_identifiers = f5["/data/processed/test/identifiers/column_annotations"][...].ravel().tolist()
    name_identifiers = [str(x, "utf8") for x in name_identifiers]
    with open(prediction_results_csv_file_name, mode="w", newline="") as fw:
        csv_writer = csv.writer(fw)
        csv_writer.writerow(name_identifiers + ["position", "probability","ground_truth"])
        
        for i in range(identifiers.shape[0]):
            csv_writer.writerow(identifiers[i,:].tolist() + [new_sort_order[i]] + [sorted_predicted_prob[i].tolist()] + [target_sorted[i]])

    with h5py.File(prediction_results_hdf5_file_name, "w") as f5w:
        ds = f5w["/"].create_dataset(name="proba", shape=target_threshold_predictions.shape,
                                     dtype=target_threshold_predictions.dtype)
        ds[...] = target_threshold_predictions[...]

    results_dict_file_name = os.path.join(output_directory, target_name_label + "_" + end_label_time_stamp + ".json")

    try:
        with open(results_dict_file_name, mode="w") as fw:
            json.dump(results_dict, fw, indent=4, sort_keys=True)
    except TypeError:
        pprint.pprint(results_dict)
        raise


if __name__ == "__main__":

    arg_parse_obj = argparse.ArgumentParser("Train a simple LSTM for EHR lab values sequences")
    arg_parse_obj.add_argument("-f", "--hdf5-file-name", dest="hdf5_file_name",
                               default="./processed_ohdsi_sequences.hdf5")
    arg_parse_obj.add_argument("-t", "--target", dest="target", default="Acute renal failure syndrome")
    arg_parse_obj.add_argument("-n", "--n-cut-off", dest="n_cut_off", default="25")
    arg_parse_obj.add_argument("-o", "--output-directory", dest="output_directory", default="./")
    arg_parse_obj.add_argument("-p", "--prediction-threshold", dest="prediction_threshold", default="0.5")
    arg_parse_obj.add_argument("-e", "--epochs", dest="epochs", default="10")
    arg_parse_obj.add_argument("-b", "--batch-size", dest="batch_size", default="50")
    arg_parse_obj.add_argument("-l", "--learning-rate", dest="learning_rate", default="1e-3")
    arg_parse_obj.add_argument("-d", "--learning-rate-decay", dest="learning_rate_decay", default="1e-4")
    arg_parse_obj.add_argument("-s", "--cut-sequence", dest="cut_sequence", default=None)
    arg_parse_obj.add_argument("-g", "--number-of-gpus", dest="number_of_gpus", default="1")

    arg_obj = arg_parse_obj.parse_args()

    if arg_obj.cut_sequence is None:
        cut_sequence_value = arg_obj.cut_sequence
    else:
        cut_sequence_value = int(arg_obj.cut_sequence)

    main(arg_obj.hdf5_file_name, arg_obj.target, n_cut=int(arg_obj.n_cut_off), output_directory=arg_obj.output_directory,
         prediction_threshold=float(arg_obj.prediction_threshold), epochs=int(arg_obj.epochs),
         batch_size=int(arg_obj.batch_size), learning_rate=float(arg_obj.learning_rate),
         learning_rate_decay=float(arg_obj.learning_rate_decay),
         cut_sequence=cut_sequence_value,
         number_of_gpus=int(arg_obj.number_of_gpus)
         )
