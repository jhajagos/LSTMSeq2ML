import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking, SimpleRNN
import numpy as np
import h5py
import pprint
import argparse
import datetime
import os


"""
This does a a canonical LSTM model 

The CuDNNLSTM is much faster but it does not support Masking

"""


def main(input_file_name, target_name, output_directory="./", n_cut=25, prediction_threshold=0.5, epochs=10, batch_size=100,
         learning_rate=1e-3, learning_rate_decay=1e-4):

    target_index_name = "static_condition_condition_concept_name|" + target_name

    f5 = h5py.File(input_file_name)

    target_name_label = "_".join(target_name.lower().split())

    f5_train = f5["/data/processed/train/sequence/core_array"]
    f5_target = f5["/data/processed/train/target/core_array"]
    f5_target_labels = f5["/data/processed/train/target/column_annotations"]

    print("Dimensions of training set:")
    print(f5_train.shape)
    print("Dimension of full target:")
    print(f5_target.shape)

    target_labels = f5_target_labels[...].ravel().tolist()
    target_labels = [str(s, "utf8") for s in target_labels]

    # Get a count of the most frequent labels
    sum_targets = np.sum(f5_target[...], axis=0).tolist()
    sum_target_with_labels = [(target_labels[i].split("|")[-1], sum_targets[i], sum_targets[i] / f5_train.shape[0]) for i in range(len(target_labels))]
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
    print("Dimensions of target:")
    print(f5_test_target.shape)

    f5_train_array = f5_train[...]
    f5_train_array[np.isnan(f5_train_array)] = 0

    print(np.sum(f5_train_array))

    f5_test_array = f5_test[...]
    f5_test_array[np.isnan(f5_test_array)] = 0

    #TODO: Make models plugin
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=f5_train_array.shape[1:]))
    model.add(LSTM(256, activation="tanh", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    best_model_file_name = os.path.join(output_directory, "best_model.h5")

    es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True)
    mc_callback = ModelCheckpoint(best_model_file_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # fit model

    # This can be refactored as explicit casts are not needed
    model.fit(np.array(f5_train_array, dtype="float32"), np.array(f5_target[:, target_index], dtype="int32"), epochs,
              validation_data=(np.array(f5_test_array, dtype="float32"),
              np.array(f5_test_target[:, target_index], dtype="int32")), callbacks=[es_callback, mc_callback],
              batch_size=batch_size)

    model.load_weights(best_model_file_name)

    time_stamp = datetime.date.utcnow().strftime("%Y%m%d_%H%M")

    coefficients_filename = target_name_label + "_" + time_stamp + "_coeffs.hdf5"
    model.save(os.path.joins(output_directory, coefficients_filename))

    # Make a probability prediction
    y_pred_keras = model.predict_proba(np.array(f5_test[...], dtype="float32")).ravel()

    print("Diagnostics of predictions")

    show_first_n = n_cut

    print("Actual test predictions %s:" % show_first_n)
    print(f5_test_target[0:show_first_n, target_index])

    target_threshold_predictions = np.array(y_pred_keras > prediction_threshold, dtype="float32")
    print("Setting the prediction threshold at %s" % prediction_threshold)
    print(np.array(target_threshold_predictions[0:show_first_n], dtype="int32"))

    print("Probability of first %s predictions from the model:" % show_first_n)
    print(y_pred_keras[0:show_first_n])

    print("Total positive cases in test set:")
    print(np.sum(f5_test_target[:, target_index]))

    print("Total predicted positive cases in test set:")
    print(np.sum(target_threshold_predictions))

    print("Total sum of probabilities")
    print(np.sum(y_pred_keras))

    from sklearn.metrics import roc_curve, roc_auc_score, f1_score, classification_report, average_precision_score

    model_auc_score = roc_auc_score(np.array(f5_test_target[:, target_index], dtype="int32"), y_pred_keras)
    print("Computed AUC of the ROC:")
    print(model_auc_score)

    f1 = f1_score(np.array(target_threshold_predictions, dtype="int32"), np.array(f5_test_target[:, target_index], dtype="int32"))
    print("F1 score:")
    print(f1)

    avg_prec_rec = average_precision_score(np.array(target_threshold_predictions, dtype="int32"), np.array(f5_test_target[:, target_index], dtype="int32"))
    print("Average precision score:")
    print(avg_prec_rec)

    print("")
    print("Classification report")

    print(classification_report(np.array(target_threshold_predictions, dtype="int32"),
                                np.array(f5_test_target[:, target_index], dtype="int32")))


if __name__ == "__main__":

    arg_parse_obj = argparse.ArgumentParser("Train a simple LSTM for EHR lab values sequences")
    arg_parse_obj.add_argument("-f", "--hdf5-file-name", dest="hdf5_file_name",
                               default="./processed_ohdsi_sequences.hdf5")

    arg_parse_obj.add_argument("-t", "--target", dest="target", default="Acute renal failure syndrome")

    arg_parse_obj.add_argument("-n", "--n-cut-off", dest="n_cut_off", default="25")

    #TODO: Add command line switches

    # def main(input_file_name, target_name, n_cut=25, prediction_threshold=0.5, epochs=10, batch_size=100,
    #          learning_rate=1e-3, learning_rate_decay=1e-4):


    arg_obj = arg_parse_obj.parse_args()

    main(arg_obj.hdf5_file_name, arg_obj.target, int(arg_obj.n_cut_off))
