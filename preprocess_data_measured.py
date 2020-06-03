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
Proposed changes:

Training and test set assignment:
    Sort by person_id,
    
    Training and test split is based on person_id's being randomly assigned to training and test set
    
        numpy.random.permutation(10)
        or sort in order
    Person id
        /static/identifier/data/
            column_annotations
                primary_person_id
    Start time
        /dynamic/carry_forward/metadata/
            column_annotations
                _start_time          
    Encounter
        /static/identifier/id
        
    Store array
        /data/training_test_split/
            core_array
            column_annotations
Add:
     static: 
        -Gender
            /static/independent/data/
                static_visit_person_gender_concept_name|MALE
                static_visit_person_gender_concept_name|FEMALE
        -Age (years)
            /static/independent/data/
                primary_age_at_visit_start_in_years_int
            min/max scale [0,100] -> [0,1]
     dynamic: 
        -Time
            /dynamic/data/carry_forward/metadata
                column_annotations
                    _sequence_time_delta
                core_array
        
        -7 days, 7 days scaling
            Current units are in seconds
                Seconds in a day
                    In [17]: 60*60*24
                    Out[17]: 86400
                Seconds in 7 days
                    In [18]: 7*60*60*24
                    Out[18]: 604800
            [-604800,604800] -> [-1,1]
    identifiers
        -Person
            /static/identifier/data/
                column_annotations
                    primary_person_id
        -Encounter
            /static/identifier/id

Write matrices into different formats
    Quantiles - carry forward
    Quantiles - changes
    
    Non-quantiles - carry forward
    Non-quantiles - changes
            
"""


def convert_binary_string_array(string_array):
    """Convert b'string' to 'python3 string'"""
    return [str(c, "utf8", errors="replace") for c in string_array.tolist()]


def slope(x1, x2, q1, q2):
    """Linear slope equation"""
    return (q2 - q1) / (x2 - x1)


def intercept(x1, x2, q1, q2):
    """Linear intercept formula"""
    return q1 - slope(x1, x2, q1, q2) * x1


def lambda_factory(x1, x2, q1, q2):
    """Lambda factory for scoping linear function"""
    return lambda x: slope(x1, x2, q1, q2) * x + intercept(x1, x2, q1, q2)


def quantile_linear_function(X, quantiles, list_values):
    """A piecewise linear functions which maps X -> quantile levels"""
    linear_functions = []
    interpolating_ranges = []

    reverse_dict = {}
    for i in range(0,len(quantiles)):
        if list_values[i] in reverse_dict:
            reverse_dict[list_values[i]] += [quantiles[i]]
        else:
            reverse_dict[list_values[i]] = [quantiles[i]]

    for i in range(1, len(quantiles)):
        x1 = list_values[i - 1]
        x2 = list_values[i]

        q1 = quantiles[i - 1]
        q2 = quantiles[i]

        if x1 != x2:  # We need to interpolate
            linear_functions += [lambda_factory(x1, x2, q1, q2)]
        else:
            linear_functions += [lambda x: reverse_dict[x1][-1]] # Do the largest jump
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


def get_variables(f5a):
    variables = f5a["/data/split/variables/core_array"][...]
    return int(variables[0, 0]), int(variables[0, 1]), int(variables[0, 2]), int(variables[0, 3]), int(variables[0, 4]), int(variables[0, 5])


def main(hdf5_file_name, output_file_name, steps_to_run, training_fraction_split, randomly_reorder_patients=True,
         compress_alg="gzip", feature_threshold=0.005
         ):

    with h5py.File(hdf5_file_name, "r") as f5:

        # We need to know where a sequence ends

        metadata_labels = f5["/dynamic/changes/metadata/column_annotations"][...]
        metadata_labels_list = convert_binary_string_array(metadata_labels)
        sequence_index = metadata_labels_list.index("_sequence_i")
        start_time_index = metadata_labels_list.index("_start_time")
        delta_time_index = metadata_labels_list.index("_sequence_time_delta")

        if "split" in steps_to_run:

            identifier_array_ds = f5["/static/identifier/data/core_array"][...]
            identifier_array = np.array(identifier_array_ds, dtype="int").ravel()
            n_size = identifier_array.shape[0]  # Number of encounters

            unique_identifier_array = np.unique(identifier_array)
            n_identifiers = unique_identifier_array.shape[0]
            n_i_training_size = int(n_identifiers * training_fraction_split)
            n_i_test_size = n_size - n_i_training_size

            with h5py.File(output_file_name, "w") as f5w:

                if randomly_reorder_patients:
                    unique_identifier_array = unique_identifier_array[np.random.permutation(n_identifiers)]
                else:
                    unique_identifier_array = np.sort(unique_identifier_array)

                train_test_dict = {}
                train_id_list = []
                test_id_list = []
                for i in range(n_identifiers):
                    if i < n_i_training_size:
                        train_test_dict[unique_identifier_array[i]] = 0
                        train_id_list += [unique_identifier_array[i]]
                    else:
                        train_test_dict[unique_identifier_array[i]] = 1
                        test_id_list += [unique_identifier_array[i]]

                identifier_count_dict = {}
                for i in range(n_size):
                    identifier = identifier_array[i]
                    if identifier in identifier_count_dict:
                        identifier_count_dict[identifier] += [i]
                    else:
                        identifier_count_dict[identifier] = [i]

                identifier_sorted_dict = {}
                # Resort the positions based on the time
                start_time_array = f5["/dynamic/carry_forward/metadata/core_array"][:, start_time_index, 0].ravel()
                for identifier in identifier_count_dict:
                    identifier_positions = identifier_count_dict[identifier]
                    start_time_list = []
                    for position in identifier_positions:
                        start_time_list += [start_time_array[position]]

                    # print(start_time_list)
                    if len(identifier_positions):
                        new_positions_index = np.lexsort((start_time_list,))
                        new_positions_list = [0] * len(new_positions_index)

                        for i in range(len(identifier_positions)):
                            new_positions_list[i] = identifier_positions[new_positions_index[i]]
                    else:
                        new_positions_list = identifier_positions

                    identifier_sorted_dict[identifier] = new_positions_list

                # We won't have a perfect split percentage but a person won't be split across the training and test set
                start_identifier_test = unique_identifier_array[n_i_training_size + 1]

                test_id_list.sort()
                train_id_list.sort()

                new_ordering_array = np.zeros(shape=(n_size, 3))
                train_i = 0
                for uid in train_id_list:
                    positions = identifier_sorted_dict[uid]
                    for pos in positions:
                        new_ordering_array[train_i, 0] = pos
                        new_ordering_array[train_i, 1] = 0
                        new_ordering_array[train_i, 2] = train_i
                        train_i += 1

                end_position_train = train_i - 1
                start_position_train = 0
                start_position_test = train_i
                end_position_test = n_size

                test_i = 0
                for uid in test_id_list:
                    positions = identifier_sorted_dict[uid]
                    for pos in positions:
                        new_ordering_array[train_i + test_i, 0] = pos
                        new_ordering_array[train_i + test_i, 1] = 1
                        new_ordering_array[train_i + test_i, 2] = test_i
                        test_i += 1

                # Store ordering of randomization
                training_test_group = f5w.create_group("/data/split/details/")
                training_test_ds = training_test_group.create_dataset("core_array", dtype="int", shape=(n_size, 3))

                training_test_ds[...] = new_ordering_array[...]

                training_test_ca_ds = training_test_group.create_dataset("column_annotations", dtype="S16", shape=(1, 3))
                tt_ca_list = ["source_index", "training_test_0_1", "new_positions"]
                training_test_ca_ds[...] = np.array(tt_ca_list, dtype="S16")

                # Store variables
                training_test_var_group = f5w.create_group("/data/split/variables/")
                dvg_ca_label_ds = training_test_var_group.create_dataset("column_annotations", shape=(1, 6), dtype="S16")
                dvg_ca_ds = training_test_var_group.create_dataset("core_array", shape=(1, 6), dtype="int")

                dvg_ca_label_ds[0, 0] = b"n_size"
                dvg_ca_ds[0, 0] = n_size
                dvg_ca_label_ds[0, 1] = b"end_position_train"
                dvg_ca_ds[0, 1] = end_position_train
                dvg_ca_label_ds[0, 2] = b"start_position_test"
                dvg_ca_ds[0, 2] = start_position_test
                dvg_ca_label_ds[0, 3] = b"end_position_test"
                dvg_ca_ds[0, 3] = end_position_test
                dvg_ca_label_ds[0, 4] = b"n_i_training_size"
                dvg_ca_ds[0, 4] = n_i_training_size
                dvg_ca_label_ds[0, 5] = b"n_i_test_size"
                dvg_ca_ds[0, 5] = n_i_test_size


        data_ds = f5["/dynamic/changes/data/core_array"]
        carry_ds = f5["/dynamic/carry_forward/data/core_array"]  # For categorical
        data_labels = f5["/dynamic/changes/data/column_annotations"]

        features_to_exclude = ["drug_exposure||0|No matching concept"]

        features_to_search_to_exclude = ["measurement||0|No matching concept|"]

        dynamic_labels = convert_binary_string_array(data_labels[...])

        for dynamic_label in dynamic_labels:
            for feature_to_exclude in features_to_search_to_exclude :
                if feature_to_exclude in dynamic_label:
                    features_to_exclude += [dynamic_label]


        dynamic_labels_reverse = list(dynamic_labels)
        dynamic_labels_reverse.reverse()

        class_labels = [d.split("|")[0] for d in dynamic_labels]
        class_labels_reverse = [d.split("|")[0] for d in dynamic_labels_reverse]

        _, n_sequence_len, n_types = data_ds.shape

        numeric_features = ["measurement", "observation"]  # Features which we are going to scale using quantiles 0 to 1
        categorical_features = ["atc5_drug_exposure", "measurement_categorical"]  # Cumulative features

        # Find start and end positions for numeric and categorical features
        numeric_features_pos_dict = {}

        for feature_class in numeric_features:
            if feature_class in class_labels:
                numeric_features_pos_dict[feature_class] = (class_labels.index(feature_class),
                                                        len(class_labels) - class_labels_reverse.index(feature_class) - 1)
        categorical_features_pos_dict = {}
        for feature_class in categorical_features:
            if feature_class in class_labels:
                categorical_features_pos_dict[feature_class] = (class_labels.index(feature_class),
                                                            len(class_labels) - class_labels_reverse.index(feature_class) - 1)

        # We are going to treat numeric features and categorical features differently
        position_class_dict = {}
        for feature_class in categorical_features_pos_dict:
            feature_range = categorical_features_pos_dict[feature_class]
            for i in range(feature_range[0], feature_range[1] + 1):
                position_class_dict[i] = "categorical"

        for feature_class in numeric_features_pos_dict:
            feature_range = numeric_features_pos_dict[feature_class]
            for i in range(feature_range[0], feature_range[1] + 1):
                position_class_dict[i] = "numeric"

        # Include static features
        independent_static_features = ["/static/independent/data/"]

        independent_static_features_names = {}
        for static_feature in independent_static_features:
            independent_static_features_names[static_feature] = convert_binary_string_array(f5[static_feature + "column_annotations"][...])

        total_number_of_independent_features = 0
        for static_feature in independent_static_features_names:
            features = independent_static_features_names[static_feature]
            total_number_of_independent_features += len(features)

        # Compute quantiles
        if "calculate" in steps_to_run:
            # Now we are going to normalize the nan values in the training set to generate the distribution
            # Our assumption here is that if we take a sample of 100,000 -  this should be big enough for getting
            # accurate estimates on the upper and lower limits

            max_number_of_samples = 100000

            with h5py.File(output_file_name, "a") as f5a:

                n_size, end_position_train, start_position_test, end_position_test, n_i_training_size, n_i_test_size = get_variables(f5a)

                # We will store the samples so we can use for later transforms
                samples_group = f5a.create_group("/data/samples")
                samples_ds = samples_group.create_dataset("measures", shape=(max_number_of_samples, n_types),
                                                          compression=compress_alg)
                samples_len_ds = samples_group.create_dataset("n", shape=(1, n_types))

                samples_labels_ds = samples_group.create_dataset("labels", shape=data_labels.shape, dtype=data_labels.dtype)
                samples_labels_ds[...] = data_labels[...]

                seq_len_group = f5a.create_group("/data/sequence_length/")
                freq_group = f5a.create_group("/data/frequency")
                freq_count_ds = freq_group.create_dataset("count", shape=(1, n_types))
                freq_count_array = np.zeros(shape=(1, n_types))
                seq_len_ds = seq_len_group.create_dataset("all", shape=(1, n_size))

                # We compute quantiles from the data
                quantile_group = f5a.create_group("/data/summary/quantiles/")
                quantiles_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95,
                                  0.99, 0.999]

                quantiles_values_ds = quantile_group.create_dataset("values", shape=(1, len(quantiles_list)))
                quantiles_values_ds[...] = quantiles_list

                quantiles_computed_ds = quantile_group.create_dataset("computed", shape=(len(quantiles_list), n_types))

                training_list_indexes = f5a["/data/split/details/core_array"][0:end_position_train, 0].tolist()

                # Collect sequences of measurement from training data to generate distributions
                # This step here needs to be optimized for larger data sets

                metadata_ds = f5["/dynamic/changes/metadata/core_array"]
                sequence_i_array = metadata_ds[:, :, sequence_index]

                for i in range(n_size):
                    sequence_array = sequence_i_array[i, :]
                    max_sequence_i = int(np.max(sequence_array))
                    seq_len_ds[0, i] = max_sequence_i

                max_sequence_array = seq_len_ds[0, :]

                for j in range(n_types):

                    feature_type = position_class_dict[j]

                    if feature_type == "numeric":
                        data_array = data_ds[:, :, j]  # Potential point to optimize
                    else:  # Categorical
                        data_array = carry_ds[:, :, j]

                    data_list = []
                    for k in range(end_position_train):

                        i = int(training_list_indexes[k])  # We get the original positions

                        max_sequence_i = int(max_sequence_array[i])

                        if len(data_list) > max_number_of_samples:
                            break
                        else:
                            if feature_type == "numeric":
                                data_slice = data_array[i, :]  # Potential point to optimize
                            else:  # Categorical
                                data_slice = [data_array[i, max_sequence_i]]

                            if 0.0 in data_slice:
                                sequence_end = max_sequence_i
                            else:  # If there are no zeros than the sequence has the maximum number of steps
                                sequence_end = len(data_slice)

                            if feature_type == "numeric":
                                data_slice = data_slice[0:sequence_end]
                                data_items = data_slice[np.logical_not(np.isnan(data_slice))].tolist()

                            else:  # For categorical just get the last value which is summed
                                if not(data_slice[-1] == 0) and not(np.isnan(data_slice[-1])):
                                    data_items = [data_slice[-1]]
                                else:
                                    data_items = []

                            if len(data_items):
                                data_list += data_items
                                freq_count_array[0, j] += 1

                    data_list = data_list[0:max_number_of_samples]
                    number_of_data_items = len(data_list)
                    samples_ds[0:number_of_data_items, j] = np.array(data_list)
                    samples_len_ds[0, j] = number_of_data_items

                    if j % 10 == 0 and j > 0:
                        print("Processed %s features" % j)

                freq_count_ds[...] = freq_count_array[...]

        # We will linear interpolate for quantiles

        if "write" in steps_to_run:
            # Measurements that occur less than a set threshold will be dropped from the test set

            with h5py.File(output_file_name, "a") as f5a:  # We reopen the processed output HDF5 file

                n_size, end_position_train, start_position_test, end_position_test, n_i_training_size, n_i_test_size = get_variables(f5a)

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
                feature_mask_raw = frequency_count >= count_feature_threshold  # Just based on counts
                feature_mask = feature_mask_raw[0]

                for feature in features_to_exclude:
                    if feature in dynamic_labels:
                        feature_mask[dynamic_labels.index(feature)] = 0
                selected_features = np.array(labels)[feature_mask]

                # Get indices for custom features
                independent_features = convert_binary_string_array(f5["/static/independent/data/column_annotations"][...])

                age_index = independent_features.index("primary_age_at_visit_start_in_years_int")
                male_index = independent_features.index("static_visit_person_gender_concept_name|MALE")
                female_index = independent_features.index("static_visit_person_gender_concept_name|FEMALE")

                if "processed" not in list(f5a["/data/"]):
                    f5a.create_group("/data/processed/train/")
                    f5a.create_group("/data/processed/test/")

                train_processed_group = f5a["/data/processed/train/"]
                test_processed_group = f5a["/data/processed/test/"]

                quantiles = f5a["/data/summary/quantiles/computed"][...]
                selected_quantiles = quantiles[:, feature_mask]

                input_dependent_shape = f5["/static/dependent_hierarchy/data/core_array"].shape
                n_target_columns = input_dependent_shape[1]

                train_n_rows = end_position_train + 1
                test_n_rows = end_position_test - start_position_test

                carry_forward_ds = f5["/dynamic/carry_forward/data/core_array"]

                carry_forward_shape = carry_forward_ds.shape
                number_of_time_steps = carry_forward_shape[1]

                number_of_custom_features = 4  # Age, Gender M, Gender F, Time Delta

                train_shape = (train_n_rows, number_of_time_steps, len(selected_features) + number_of_custom_features)
                test_shape = (test_n_rows, number_of_time_steps, len(selected_features) + number_of_custom_features)

                train_target_shape = (train_n_rows, n_target_columns)
                test_target_shape = (test_n_rows, n_target_columns)

                row_ids = f5["/static/identifier/id/core_array"][...].ravel()
                patient_ids = f5["/static/identifier/data/core_array"][:, 0].ravel()  # TODO: Remove hard coded index
                start_times = f5["/dynamic/carry_forward/metadata/core_array"][:, 0, start_time_index].ravel()

                if "sequence" not in list(f5a["/data/processed/train/"]):
                    f5a.create_group("/data/processed/train/sequence/")
                    f5a.create_group("/data/processed/train/target/")
                    f5a.create_group("/data/processed/train/identifiers/")

                    f5a.create_group("/data/processed/test/sequence/")
                    f5a.create_group("/data/processed/test/target/")
                    f5a.create_group("/data/processed/test/identifiers/")

                if "core_array" in list(f5a["/data/processed/train/sequence"]):
                    del f5a["/data/processed/train/sequence/core_array"]
                    del f5a["/data/processed/train/sequence/column_annotations"]
                    del f5a["/data/processed/train/target/core_array"]
                    del f5a["/data/processed/train/target/column_annotations"]

                    del f5a["/data/processed/test/sequence/core_array"]
                    del f5a["/data/processed/test/sequence/column_annotations"]
                    del f5a["/data/processed/test/target/core_array"]
                    del f5a["/data/processed/test/target/column_annotations"]

                if "core_array" in list(f5a["/data/processed/train/identifiers"]):
                    del f5a["/data/processed/train/identifiers/core_array"]
                    del f5a["/data/processed/train/identifiers/column_annotations"]

                    del f5a["/data/processed/test/identifiers/core_array"]
                    del f5a["/data/processed/test/identifiers/column_annotations"]

                train_seq_ds = f5a["/data/processed/train/sequence/"].create_dataset("core_array", shape=train_shape,
                                                                                     dtype="float32",
                                                                                     compression=compress_alg
                                                                                     )

                train_seq_label_ds = f5a["/data/processed/train/sequence"].create_dataset("column_annotations",
                                                                                          shape=(1, len(selected_features) + number_of_custom_features),
                                                                                          dtype=data_labels.dtype)

                train_target_ds = f5a["/data/processed/train/target/"].create_dataset("core_array", shape=train_target_shape,
                                                                                      dtype="int32",
                                                                                      compression=compress_alg
                                                                                      )

                train_target_label_ds = f5a["/data/processed/train/target/"].create_dataset("column_annotations",
                                                                                            shape=(1, train_target_shape[1]),
                                                                                            dtype=data_labels.dtype,
                                                                                            compression=compress_alg
                                                                                            )

                test_seq_ds = f5a["/data/processed/test/sequence/"].create_dataset("core_array", shape=test_shape,
                                                                                   dtype="float32",
                                                                                   compression=compress_alg)

                test_seq_label_ds = f5a["/data/processed/test/sequence"].create_dataset("column_annotations",
                                                                                        shape=(1, len(selected_features) + number_of_custom_features),
                                                                                        dtype=data_labels.dtype,
                                                                                        )

                test_target_ds = f5a["/data/processed/test/target/"].create_dataset("core_array", shape=test_target_shape,
                                                                                    dtype="int32")

                test_target_label_ds = f5a["/data/processed/test/target/"].create_dataset("column_annotations",
                                                                                          shape=(1, test_target_shape[1]),
                                                                                          dtype=data_labels.dtype)

                train_seq_label_ds[0, 0:len(selected_features)] = np.array(selected_features, dtype=data_labels.dtype)
                test_seq_label_ds[0, 0:len(selected_features)] = np.array(selected_features, dtype=data_labels.dtype)

                custom_features = [b"age_years_fraction_100", b"gender|Male", b"gender|Female", b"time_fraction_weeks"]
                train_seq_label_ds[0, -4:] = np.array(custom_features, dtype=data_labels.dtype)
                test_seq_label_ds[0, -4:] = np.array(custom_features, dtype=data_labels.dtype)

                train_id_ds = f5a["/data/processed/train/identifiers/"].create_dataset("core_array",
                                                                                       shape=(train_n_rows, 3),
                                                                                       dtype="int")
                test_id_ds = f5a["/data/processed/test/identifiers/"].create_dataset("core_array",
                                                                                       shape=(test_n_rows, 3),
                                                                                       dtype="int")

                train_id_labels_ds = f5a["/data/processed/train/identifiers/"].create_dataset("column_annotations",
                                                                                       shape=(1, 3),
                                                                                       dtype="S16")

                test_id_labels_ds = f5a["/data/processed/test/identifiers/"].create_dataset("column_annotations",
                                                                                              shape=(1, 3),
                                                                                              dtype="S16")

                train_id_labels_ds[...] = np.array([b"id", b"identifier_id", b"start_time"])
                test_id_labels_ds[...] = np.array([b"id", b"identifier_id", b"start_time"])

                # Target labels
                train_target_label_ds[...] = f5["/static/dependent_hierarchy/data/column_annotations"][...]
                test_target_label_ds[...] = f5["/static/dependent_hierarchy/data/column_annotations"][...]

                computed_quantiles = quantiles_computed_ds[:, feature_mask]
                quantile_values = f5a["/data/summary/quantiles/values"][...]

                positions_array = f5a["/data/split/details/core_array"][...]
                seq_len_ds = f5a["/data/sequence_length/all"]
                max_sequence_array = seq_len_ds[0, :]

                # Custom feature arrays
                age_array = f5["/static/independent/data/core_array"][:, age_index].ravel()
                age_array = age_array / 100.0
                male_array = f5["/static/independent/data/core_array"][:, male_index].ravel()
                female_array = f5["/static/independent/data/core_array"][:, female_index].ravel()

                # Builds the independent sequence matrices for prediction
                for k in range(n_size):

                    # Setup indexes
                    i = int(positions_array[k, 0])
                    is_test = int(positions_array[k, 1])
                    pos_2_write = int(positions_array[k, 2])

                    sequence_length = int(max_sequence_array[i]) + 1
                    if sequence_length == 0:
                        sequence_length = 1

                    i_carry_forward_array = carry_forward_ds[i, 0:sequence_length, feature_mask]
                    last_step = i_carry_forward_array[-1, :]

                    has_feature = np.logical_not(np.isnan(last_step))
                    feature_position = np.array(range(0, i_carry_forward_array.shape[1]))
                    original_feature_position = feature_position[has_feature]
                    quantile_features = computed_quantiles[:, has_feature]

                    i_carry_forward_feature_array = i_carry_forward_array[:, has_feature]

                    t_carry_forward_array = np.zeros(shape=(sequence_length, np.sum(feature_mask)), dtype=i_carry_forward_feature_array.dtype)
                    t_carry_forward_array[t_carry_forward_array == 0] = np.nan

                    for j in range(quantile_features.shape[1]):

                        has_measured_values = np.logical_not(np.isnan(i_carry_forward_feature_array[:, j]))

                        linear_quantile = quantile_linear_function(i_carry_forward_feature_array[has_measured_values, j],
                                                                   quantile_values.tolist()[0],
                                                                   quantile_features[:, j].tolist())
                        t_carry_forward_array[has_measured_values, original_feature_position[j]] = linear_quantile

                    for feature_class in categorical_features_pos_dict:
                        feature_class_range = categorical_features_pos_dict[feature_class]

                    for feature_class in numeric_features_pos_dict:
                        feature_class_range = numeric_features_pos_dict[feature_class]

                    t_carry_forward_array[np.isnan(t_carry_forward_array)] = 0  # np.isnan = 0.5 for numeric values

                    # Custom variables
                    custom_sub_array = np.zeros(shape=(sequence_length, 4))
                    custom_sub_array[:, 0] = age_array[i]  # Age
                    custom_sub_array[:, 1] = male_array[i]  # Male
                    custom_sub_array[:, 2] = female_array[i]  # Female
                    # Divide number of seconds elapsed by seconds in a week
                    custom_sub_array[:, 3] = f5["/dynamic/carry_forward/metadata/core_array"][i, 0:sequence_length,
                                             delta_time_index] / 604800.0

                    # We need to split into separate
                    if is_test:
                        test_seq_ds[pos_2_write, 0:sequence_length, :] = np.concatenate((t_carry_forward_array, custom_sub_array), axis=1)
                        test_target_ds[pos_2_write, :] = f5["/static/dependent_hierarchy/data/core_array"][i, :]  # Test DX

                    else:
                        train_seq_ds[pos_2_write, 0:sequence_length, :] = np.concatenate((t_carry_forward_array, custom_sub_array), axis=1)
                        train_target_ds[pos_2_write, :] = f5["/static/dependent_hierarchy/data/core_array"][i, :]  # Target DX

                    # Identifiers
                    if is_test:
                        test_id_ds[pos_2_write, :] = np.array([int(row_ids[i]), int(patient_ids[i]), int(start_times[i])])
                    else:
                        train_id_ds[pos_2_write, :] = np.array([int(row_ids[i]), int(patient_ids[i]), int(start_times[i])])

                    if k % 100 == 0:
                        print("Wrote %s matrices" % k)


if __name__ == "__main__":

    arg_parse_obj = argparse.ArgumentParser(description="Preprocess TimeWeaver HDF5 for machine learning creating a test and training sets")
    arg_parse_obj.add_argument("-f", "--hdf5-file-name", dest="hdf5_file_name")
    arg_parse_obj.add_argument("-o", "--output-hdf5-file-name", dest="output_file_name")
    arg_parse_obj.add_argument("-s", "--split-into-training-test-set", dest="split_training_test", default=False,
                               action="store_true")
    arg_parse_obj.add_argument("-r", "--recalculate-quantiles", dest="recalculate_samples", default=False,
                               action="store_true", help="Calculates the quantiles for measured values")
    arg_parse_obj.add_argument("-w", "--write-matrices", dest="write_matrices", default=False, action="store_true")
    arg_parse_obj.add_argument("-a", "--run-all-steps", dest="run_all_steps", default=False, action="store_true",
                               help="Run all steps: split into training and test, recalculate the quantiles, and "
                                    "write matrices")
    arg_parse_obj.add_argument("-t", "--feature-fraction-threshold", dest="feature_fraction_threshold", default="0.005")
    arg_parse_obj.add_argument("--fraction-training", dest="fraction_training", default="0.8")

    arg_obj = arg_parse_obj.parse_args()

    steps_to_run = []
    if arg_obj.run_all_steps:
        steps_to_run = ["split", "calculate", "write"]
    elif arg_obj.split_training_test:
        steps_to_run = ["split"]
    elif arg_obj.recalculate_samples:
        steps_to_run = ["calculate"]
    elif arg_obj.write_matrices:
        steps_to_run = ["write"]
    else:
        raise RuntimeError("Specify either -s, -r, or -w")

    main(arg_obj.hdf5_file_name, arg_obj.output_file_name, steps_to_run=steps_to_run,
         training_fraction_split=float(arg_obj.fraction_training),
         feature_threshold=float(arg_obj.feature_fraction_threshold))

