import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, CuDNNLSTM, Masking, SimpleRNN
from tensorflow.keras.optimizers import SGD
import numpy as np
import h5py

"""

"""


def main(input_file_name):

    f5 = h5py.File(input_file_name)

    f5_train = f5["/data/processed/train/sequence/core_array"]
    f5_target = f5["/data/processed/train/target/core_array"]
    f5_target_labels = f5["/data/processed/train/target/column_annotations"]

    target_labels = f5_target_labels[...].ravel().tolist()
    target_labels = [str(s, "utf8") for s in target_labels]

    sum_targets = np.sum(f5_target[...], axis=0).tolist()

    sum_target_with_labels = [(target_labels[i].split("|")[-1], sum_targets[i]) for i in range(len(target_labels))]

    sum_target_with_labels.sort(key=lambda x: -1 * x[1])

    print(sum_target_with_labels[0:50])

    # raise(RuntimeError)
    # Let us generate a list of most frequent features to target for prediction

    target_index = target_labels.index("static_condition_condition_concept_name|Acute renal failure syndrome")
    # print(target_labels)
    # target_index = target_labels.index("static_condition_condition_concept_name|Anxiety disorder")
    # target_index = target_labels.index("static_condition_condition_concept_name|Essential hypertension")
    # target_index = target_labels.index("static_condition_condition_concept_name|Type 2 diabetes mellitus without complication")
    # target_index = target_labels.index("static_condition_condition_concept_name|Respiratory failure")
    # target_index = target_labels.index("static_condition_condition_concept_name|Heart failure")
    # target_index = target_labels.index("static_condition_condition_concept_name|Chronic kidney disease")
    # target_index = target_labels.index("static_condition_condition_concept_name|Hyperkalemia")
    # target_index = target_labels.index("static_condition_condition_concept_name|Sepsis")

    f5_test = f5["/data/processed/test/sequence/core_array"]
    f5_test_target = f5["/data/processed/test/target/core_array"]

    f5_train_array = f5_train[...]
    f5_train_array[np.isnan(f5_train_array)] = 0

    print(np.sum(f5_train_array))

    f5_test_array = f5_test[...]
    f5_test_array[np.isnan(f5_test_array)] = 0

    model = Sequential()
    # model.add(SimpleRNN(128, activation="tanh", return_sequences=True))
    # #model.add(Dropout(0.10))
    #
    # model.add(SimpleRNN(64, activation="tanh"))
    # #model.add(Dropout(0.10))
    #
    # model.add(Dense(64, activation="sigmoid"))
    # #model.add(Dropout(0.10))

    #model.add(LSTM(128, activation="tanh", input_shape=(100,182)))

    model.add(Masking(mask_value=0.0, input_shape=f5_train_array.shape[1:]))

    model.add(LSTM(128, activation="tanh", return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    # configure sgd with gradient norm clipping
    # opt = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)

    print(f5_train.shape[1:])

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit(np.array(f5_train_array, dtype="float32"), np.array(f5_target[:, target_index], dtype="int32"), epochs=3,
              validation_data=(np.array(f5_test_array, dtype="float32"),
                                                                                   np.array(f5_test_target[:,target_index],dtype="int32")))
    y_pred_keras = model.predict_proba(np.array(f5_test[...], dtype="float32")).ravel()

    # y_pred_keras = model.predict_proba(f5_test[...]).ravel()

    # fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.array(f5_test_target[:, target_index],dtype="int32"), y_pred_keras)

    model.save("coeffs.hdf5")

    print(y_pred_keras[0:60])
    print(f5_test_target[0:60, target_index])
    print(np.sum(f5_test_target[0:60, target_index]))
    print(np.array(y_pred_keras[0:60] > 0.5, dtype="float32"))
    print(np.sum(y_pred_keras[0:60]))

    from sklearn.metrics import roc_curve, roc_auc_score
    print(roc_auc_score(np.array(f5_test_target[:, target_index],dtype="int32"), y_pred_keras))


if __name__ == "__main__":
    main("Y:\\healthfacts\\ts\\processed_ohdsi_sequences.hdf5")