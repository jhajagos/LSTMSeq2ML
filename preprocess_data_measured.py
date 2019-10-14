"""
    The goal here is to demonstrate how we can use sequential data from EHRs for making
    DX prediction.

    The data here has been processed using the TimeWeaver library and OHDSI mapping class files.

    TimeWeaver library can support adding medication.
"""

import h5py  # Python library for reading HDF5
import numpy as np
import argparse

"""

Changes:

Sample:
    Run through

Exclude:
    "measurement||0|No matching concept"
    "drug_exposure||0|No matching concept"
Add:
     static: Gender, Age,
     dynamic: Days

Linear interpolation between quantiles for measurement values

"""


def convert_binary_string_array(string_array):
    return [str(c, "utf8", errors="replace") for c in string_array.tolist()]


def slope(x1, x2, q1, q2):
    return (q2 - q1) / (x2 - x1)


def intercept(x1, x2, q1, q2):
    return q1 - slope(x1, x2, q1, q2) * x1


def lambda_factory(x1, x2, q1, q2):
    """Lambda factory for scoping linear function"""
    return lambda x: slope(x1, x2, q1, q2) * x + intercept(x1, x2, q1, q2)


def quantile_linear_function(X, quantiles, list_values):
    """A piecewise linear functions which maps X -> quantile levels"""
    linear_functions = []
    interpolating_ranges = []
    for i in range(1, len(quantiles)):
        x1 = list_values[i - 1]
        x2 = list_values[i]

        q1 = quantiles[i - 1]
        q2 = quantiles[i]

        linear_functions += [lambda_factory(x1, x2, q1, q2)]

        interpolating_ranges += [(x1, x2)]

    linear_functions = [lambda x: 0] + linear_functions + [lambda x: 1]  # set to 0 and 1 if above last quantiles
    interpolating_ranges = [interpolating_ranges[0][0]] + interpolating_ranges + [interpolating_ranges[-1][1]]

    conditions = []
    for i in range(len(linear_functions)):
        if i == 0:
            conditions += [X < interpolating_ranges[i]]
        elif i == len(linear_functions) - 1:
            conditions += [X > interpolating_ranges[i]]
        elif i == len(linear_functions) - 2:
            conditions += [np.logical_and(X >= interpolating_ranges[i][0], X <= interpolating_ranges[i][1])]
        else:
            conditions += [np.logical_and(X >= interpolating_ranges[i][0], X < interpolating_ranges[i][1])]

    return np.piecewise(X, condlist=conditions, funclist=linear_functions)


def main(hdf5_file_name, output_file_name, training_split=0.80, recalculate_samples=True):

    with h5py.File(hdf5_file_name, "r") as f5:

        # We need to normalize the data
        # but first we will split the data into test and training test
        # For now we will split continuous regions
        # We want to make sure that a person is not split across the training and test split

        identifier_array_ds = f5["/static/identifier/data/core_array"][...]
        identifier_array = np.array(identifier_array_ds, dtype="int").ravel()

        n_size = identifier_array.shape[0]  # Number of encounters
        unique_identifier_array = np.unique(identifier_array)
        n_identifiers = unique_identifier_array.shape[0]
        n_i_training_size = int(n_identifiers * training_split)

        end_identifier_train = unique_identifier_array[n_i_training_size]
        start_identifier_test = unique_identifier_array[n_i_training_size + 1]

        start_position_test = identifier_array.tolist().index(start_identifier_test)
        end_position_train = start_position_test - 1

        # We won't have a perfect split percentage wise but we won't have a person split across training and test set

        start_position_train = 0
        end_position_test = n_size

        # Now we are going to normalize the nan values in the training set to generate the distribution
        # Our assumption here is that if we take a sample of 100,000 -  this should be big enough for getting
        # accurate estimates on the upper and lower limits

        max_number_of_samples = 100000

        # We need to know where a sequence ends

        metadata_labels = f5["/dynamic/changes/metadata/column_annotations"][...]
        sequence_index = convert_binary_string_array(metadata_labels).index("_sequence_i")
        metadata_array = f5["/dynamic/changes/metadata/core_array"]

        data_ds = f5["/dynamic/changes/data/core_array"]
        data_labels = f5["/dynamic/changes/data/column_annotations"]

        columns_to_exclude = ["measurement||0|No matching concept", "drug_exposure||0|No matching concept"]
        dynamic_labels = convert_binary_string_array(data_labels[...])

        _, n_sequence_len, n_types = data_ds.shape

        numeric_features = ["measurements"]  # Features which we are going to scale using quantiles -1 to 1
        categorical_features = ["drug_exposure"]  # Cumulative features

        if recalculate_samples:

            with h5py.File(output_file_name, "w") as f5w:

                # We will store the samples so we can use for later transforms

                samples_group = f5w.create_group("/data/samples")
                samples_ds = samples_group.create_dataset("measures", shape=(max_number_of_samples, n_types))
                samples_len_ds = samples_group.create_dataset("n", shape=(1, n_types))

                samples_labels_ds = samples_group.create_dataset("labels", shape=data_labels.shape, dtype=data_labels.dtype)
                samples_labels_ds[...] = data_labels[...]

                seq_len_group = f5w.create_group("/data/sequence_length/")
                freq_group = f5w.create_group("/data/frequency")
                freq_count_ds = freq_group.create_dataset("count", shape=(1, n_types))
                seq_len_ds = seq_len_group.create_dataset("all", shape=(1, n_size))

                # We compute quantiles from the data
                quantile_group = f5w.create_group("/data/summary/quantiles/")
                quantiles_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95,
                                  0.99, 0.999]

                quantiles_values_ds = quantile_group.create_dataset("values", shape=(1, len(quantiles_list)))
                quantiles_values_ds[...] = quantiles_list

                quantiles_computed_ds = quantile_group.create_dataset("computed", shape=(len(quantiles_list), n_types))

                # Collect sequences of measurement from training data to generate distributions
                # This step here needs to be optimized for larger data sets
                for j in range(n_types):
                    data_list = []
                    for i in range(start_position_train, end_position_test):

                        sequence_array = metadata_array[i, :, sequence_index]
                        max_sequence_i = int(np.max(sequence_array))
                        seq_len_ds[0, i] = max_sequence_i + 1

                        if len(data_list) > max_number_of_samples:
                            break
                        else:
                            data_slice = data_ds[i, :, j].tolist()
                            if 0 in data_slice:
                                sequence_end = max_sequence_i
                            else:
                                sequence_end = len(data_slice)

                            data_items = [ds for ds in data_slice[0:sequence_end] if not(np.isnan(ds))]

                            if len(data_items):
                                data_list += data_items
                                freq_count_ds[0, j] += 1

                    data_list = data_list[0:max_number_of_samples]

                    number_of_data_items = len(data_list)
                    samples_ds[0:number_of_data_items, j] = np.array(data_list)
                    samples_len_ds[0, j] = number_of_data_items

                    if j % 10 == 0 and j > 0:
                        print("Processed %s features" % j)

        # We will linear interpolate for quantiles

        # Measurements that occur less than a set threshold will be dropped
        feature_threshold = 0.01

        with h5py.File(output_file_name, "a") as f5a:  # We reopen the processed output HDF5 file
            labels = convert_binary_string_array(f5a["/data/samples/labels"][...])
            quantiles_values = f5a["/data/summary/quantiles/values"][...].ravel()
            quantiles_computed_ds = f5a["/data/summary/quantiles/computed"]
            for i in range(len(labels)):
                n_ds = f5a["/data/samples/n"]
                samples_ds = f5a["/data/samples/measures"]
                sequence_length = int(n_ds[0, i])
                if sequence_length:
                    series = samples_ds[0:sequence_length, i]

                    quantiles_computed_ds[:, i] = np.quantile(series, quantiles_values)

            count_feature_threshold = int(n_i_training_size * feature_threshold)

            frequency_count = f5a["/data/frequency/count"][...]
            feature_mask_raw = frequency_count >= count_feature_threshold
            feature_mask = feature_mask_raw[0]
            feature_mask[0] = False  # unmapped codes

            selected_features = np.array(labels)[feature_mask]
            if "processed" not in list(f5a["/data/"]):
                f5a.create_group("/data/processed/train/")
                f5a.create_group("/data/processed/test/")

            train_processed_group = f5a["/data/processed/train/"]
            test_processed_group = f5a["/data/processed/test/"]

            quantiles = f5a["/data/summary/quantiles/computed"][...]
            selected_quantiles = quantiles[:, feature_mask]

            input_dependent_shape = f5["/static/dependent/data/core_array"].shape
            n_target_columns = input_dependent_shape[1]

            train_n_rows = end_position_train + 1
            test_n_rows = end_position_test - start_position_test

            carry_forward_ds = f5["/dynamic/carry_forward/data/core_array"]

            carry_forward_shape = carry_forward_ds.shape
            number_of_time_steps = carry_forward_shape[1]

            train_shape = (train_n_rows, number_of_time_steps, len(selected_features))
            test_shape = (test_n_rows, number_of_time_steps, len(selected_features))

            train_target_shape = (train_n_rows, n_target_columns)
            test_target_shape = (test_n_rows, n_target_columns)

            if "sequence" not in list(f5a["/data/processed/train/"]):
                f5a.create_group("/data/processed/train/sequence/")
                f5a.create_group("/data/processed/train/target/")

                f5a.create_group("/data/processed/test/sequence/")
                f5a.create_group("/data/processed/test/target/")

            if "core_array" in list(f5a["/data/processed/train/sequence"]):
                del f5a["/data/processed/train/sequence/core_array"]
                del f5a["/data/processed/train/sequence/column_annotations"]
                del f5a["/data/processed/train/target/core_array"]
                del f5a["/data/processed/train/target/column_annotations"]

                del f5a["/data/processed/test/sequence/core_array"]
                del f5a["/data/processed/test/sequence/column_annotations"]
                del f5a["/data/processed/test/target/core_array"]
                del f5a["/data/processed/test/target/column_annotations"]

            train_seq_ds = f5a["/data/processed/train/sequence/"].create_dataset("core_array", shape=train_shape,
                                                                                 dtype="float32")

            train_seq_label_ds = f5a["/data/processed/train/sequence"].create_dataset("column_annotations",
                                                                                      shape=(1, len(selected_features)),
                                                                                      dtype=data_labels.dtype)

            train_target_ds = f5a["/data/processed/train/target/"].create_dataset("core_array", shape=train_target_shape,
                                                                                  dtype="int32")

            train_target_label_ds = f5a["/data/processed/train/target/"].create_dataset("column_annotations",
                                                                                        shape=(1, train_target_shape[1]),
                                                                                        dtype=data_labels.dtype)

            test_seq_ds = f5a["/data/processed/test/sequence/"].create_dataset("core_array", shape=test_shape,
                                                                               dtype="float32")

            test_seq_label_ds = f5a["/data/processed/test/sequence"].create_dataset("column_annotations",
                                                                                    shape=(1, len(selected_features)),
                                                                                    dtype=data_labels.dtype)

            test_target_ds = f5a["/data/processed/test/target/"].create_dataset("core_array", shape=test_target_shape,
                                                                                dtype="int32")

            test_target_label_ds = f5a["/data/processed/test/target/"].create_dataset("column_annotations",
                                                                                      shape=(1, test_target_shape[1]),
                                                                                      dtype=data_labels.dtype)

            train_seq_label_ds[...] = np.array(selected_features, dtype=data_labels.dtype)
            test_seq_label_ds[...] = np.array(selected_features, dtype=data_labels.dtype)

            # Populate targets
            train_target_label_ds[...] = f5["/static/dependent/data/column_annotations"][...]
            test_target_label_ds[...] = f5["/static/dependent/data/column_annotations"][...]

            train_target_ds[...] = f5["/static/dependent/data/core_array"][start_position_train:end_position_train+1, :]

            test_target_ds[...] = f5["/static/dependent/data/core_array"][start_position_test:end_position_test, :]

            # Builds the independent sequence matrices for prediction
            for i in range(n_size):
                sequence_length = int(f5a["/data/sequence_length/all"][0, i])
                i_carry_forward_array = carry_forward_ds[i, 0:sequence_length, feature_mask]

                quantile_01 = quantiles_computed_ds[0, feature_mask]
                quantile_99 = quantiles_computed_ds[-1, feature_mask]

                # This scales the measurements between -1 the lowest quantile and 1 the upper quantile
                t_carry_forward_array = -1 * (1 - (2 * (i_carry_forward_array - quantile_01) / (quantile_99 - quantile_01)))

                t_carry_forward_array[t_carry_forward_array < -1] = -1
                t_carry_forward_array[t_carry_forward_array > 1] = 1

                t_carry_forward_array[np.isnan(t_carry_forward_array)] = 0

                # We need to split into separate
                if i <= end_position_train:
                    train_seq_ds[i, 0:sequence_length, :] = t_carry_forward_array
                else:
                    test_seq_ds[i - start_position_test, 0:sequence_length, :] = t_carry_forward_array

                if i % 100 == 0:
                    print("Wrote %s matrices" % i)


if __name__ == "__main__":

    # arg_parse_obj = argparse.ArgumentParser(description="Pre-process HDF5 for applications")
    # arg_parse_obj.add_argument("-f", "--hdf5-file-name", dest="hdf5_file_name")
    # arg_parse_obj.add_argument("-o", "--output-hdf5-file-name", dest="output_file_name")
    # arg_obj = arg_parse_obj.parse_args()
    #
    # main(arg_obj.hdf5_file_name, arg_obj.output_file_name)
    main("Y:\\healthfacts\\ts\\measurement_drug\\ohdsi_sequences.hdf5.subset.hdf5", "Y:\\healthfacts\\ts\\measurement_drug\\processed_ohdsi_sequences.subset.hdf5")