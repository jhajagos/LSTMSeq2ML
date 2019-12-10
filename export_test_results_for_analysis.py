import h5py
import csv
import argparse
import numpy as np
import json
import tensorflow as tf
import os

"""Export CSV results"""


def convert_binary_string_array(string_array):
    """Convert b'string' to 'python3 string'"""
    return [str(c, "utf8", errors="replace") for c in string_array.ravel().tolist()]


def generate_row(header, row_dict):
    row_to_write = [""] * len(header)
    for field_name in row_dict:
        field_position = header.index(field_name)
        row_to_write[field_position] = row_dict[field_name]

    return row_to_write


def main(processed_hdf5_file_name, csv_export_file_name, original_hdf5_file_name=None, hdf5_model_file_name=None,
         export_tracks=True, gpu_acceleration=False, write_n_rows=10):

    if not gpu_acceleration:
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    if hdf5_model_file_name is not None:
        relative_hdf5_model_file_name = os.path.split(hdf5_model_file_name)[-1]
        model_to_apply = tf.keras.models.load_model(hdf5_model_file_name)

    with h5py.File(processed_hdf5_file_name, "r") as f5:

        with open(csv_export_file_name, mode="w", newline="") as fw:
            csv_writer = csv.writer(fw)
            name_identifiers = convert_binary_string_array(f5["/data/processed/test/identifiers/column_annotations"][...])

            id_pos = name_identifiers.index("id")
            identifier_pos = name_identifiers.index("identifier_id")

            sequence_track_names = convert_binary_string_array(f5["/data/processed/test/sequence/column_annotations"][...])

            quantile_data_labels = convert_binary_string_array(f5["/data/samples/labels"][...])
            quantile_values = f5["/data/summary/quantiles/values"][...].ravel().tolist()
            quantiles_computed = f5["/data/summary/quantiles/computed"][...]

            target_labels_array = f5["/data/processed/test/target/column_annotations"][...].ravel()
            targets_array = f5["/data/processed/test/target/core_array"][...]
            identifiers_array = f5["/data/processed/test/identifiers/core_array"][...]

            quantiles_dict = {}
            for i in range(len(quantile_data_labels)):

                quantile_data_label = quantile_data_labels[i]
                i_quantile_computed = quantiles_computed[:, i]
                quantile_value_dict = {}

                for j in range(len(quantile_values)):
                    j_quantile_value = quantile_values[j]
                    j_quantile_computed = i_quantile_computed[j]
                    quantile_value_dict["q_" + str(j_quantile_value)] = j_quantile_computed

                quantiles_dict[quantile_data_label] = quantile_value_dict

            quantiles_header = ["q_" + str(q) for q in quantile_values]

            pre_header = ["row_id", "original_sequence_track_id", "sequence_track_id", "original_event_id", "entity_id",
                          "type", "name"]

            test_seq_ds = f5["/data/processed/test/sequence/core_array"]

            train_seq_shape = test_seq_ds.shape
            n_rows, n_time_steps, n_tracks = train_seq_shape

            time_steps_header = ["t_" + str(i) for i in range(n_time_steps)]
            # time_steps_pos = [i for i in range(n_time_steps)]
            header = pre_header + time_steps_header + quantiles_header + ["targets"]

            csv_writer.writerow(header)

            for i_row in range(n_rows):

                #print(i_row)
                if write_n_rows is not None and write_n_rows == i_row:
                    break

                identifier_array = identifiers_array[i_row, :].ravel()
                id_value = int(identifier_array[id_pos])
                identifier_value = int(identifier_array[identifier_pos])

                if export_tracks:

                    sequence_array = test_seq_ds[i_row, :, :]
                    tracks_sums = np.sum(sequence_array, axis=0)

                    track_j = 0
                    i_target_row = targets_array[i_row, :].ravel()
                    target_mask = i_target_row > 0
                    raw_targets = convert_binary_string_array(target_labels_array[target_mask])
                    targets = [t.split("|")[-1] for t in raw_targets]

                    for i_track in range(n_tracks):

                        if int(tracks_sums[i_track]):
                            track_array = sequence_array[:, i_track]

                            row_dict = {"row_id": i_row, "original_sequence_track_id": i_track,
                                        "sequence_track_id": track_j, "original_event_id": id_value,
                                        "entity_id": identifier_value, "type": "track_value"}

                            row_dict["targets"] = json.dumps(targets)

                            track_name = sequence_track_names[i_track]
                            row_dict["name"] = track_name

                            for i_step in range(n_time_steps):
                                track_value = track_array[i_step]

                                if track_value != 0.0:
                                    row_dict["t_" + str(i_step)] = track_value

                            if track_name in quantiles_dict:
                                for qk in quantiles_dict[track_name]:
                                    row_dict[qk] = quantiles_dict[track_name][qk]

                            row_to_write = generate_row(header, row_dict)
                            csv_writer.writerow(row_to_write)

                            track_j += 1

                if hdf5_model_file_name is not None:

                    row_dict = {"row_id": i_row, "original_sequence_track_id": "",
                                "sequence_track_id": "", "original_event_id": id_value,
                                "entity_id": identifier_value, "type": "prediction_probability",
                                "name": relative_hdf5_model_file_name}

                    test_seq_array = test_seq_ds[i_row:i_row+1, :, :]

                    # print(train_seq_ds[i:i + 1, :, :].shape)
                    # print(train_seq_array[0, :, -1].tolist())
                    # print(train_seq_array[0, :, -1].tolist().index(0.0) - 1)
                    # raise

                    # print(train_seq_array.shape)

                    values_array = test_seq_array[0, :, -1].tolist()
                    if 0.0 in values_array:
                        end_position = values_array.index(0.0) - 1
                    else:
                        end_position = len(values_array) - 1

                    for position in range(end_position + 1):

                        copy_train_seq_array = test_seq_array.copy()
                        zero_positions = position + 1
                        # print(position, zero_positions)
                        copy_train_seq_array[:, zero_positions:, :] = 0

                        # print(copy_train_seq_array)
                        # raise

                        predictions = model_to_apply.predict_proba(copy_train_seq_array)
                        row_dict["t_" + str(position)] = float(predictions[0])

                    # print(row_dict)
                    row_to_write = generate_row(header, row_dict)
                    csv_writer.writerow(row_to_write)


if __name__ == "__main__":

    main("C:\\Users\\janos\\data\\ts\\healthfacts\\20191017\\processed_ohdsi_sequence.complete.hdf5",
         "C:\\Users\\janos\\data\\ts\\healthfacts\\20191017\\processed_ohdsi_sequence.complete.hdf5.csv",
         hdf5_model_file_name="C:\\Users\\janos\\data\\ts\\healthfacts\\20191017\\acute_renal_failure_syndrome_20191207_2148_coeffs.hdf5"
         )

    # "C:\\Users\\janos\\data\\ts\\healthfacts\\20191017\\acute_renal_failure_syndrome_20191207_2148_coeffs.hdf5"