
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import h5py
import numpy as np

def convert_binary_string_array(string_array):
    return [str(c, "utf8", errors="replace") for c in string_array.tolist()]


def main(hdf5_file_name, output_file_name, training_split=0.80):

    with h5py.File(hdf5_file_name, "r") as f5:

        # We need to normalize the data

        # but first we will split the data into test and training test

        # For now we will split continuous regions

        # We want to make sure that a person is not split across the training and test split

        identifier_array_ds = f5["/static/identifier/data/core_array"][...]

        identifier_array = np.array(identifier_array_ds,dtype="int").ravel()

        n_size = identifier_array.shape[0]

        unique_identifier_array = np.unique(identifier_array)

        n_identifiers = unique_identifier_array.shape[0]

        n_i_training_size = int(n_identifiers * training_split)

        end_identifier_train = unique_identifier_array[n_i_training_size]
        start_identifier_test = unique_identifier_array[n_i_training_size + 1]

        start_position_test  = unique_identifier_array.tolist().index(start_identifier_test)
        end_position_train = start_position_test - 1

        # We won't have a perfect split percentage wise but we won't have a person split across training and test set

        start_position_train = 0
        end_position_test = n_identifiers

        # Now we are going to normalize the nan values in the training set to generate the distribution
        # Our assumption here is that if we take a sample of 10000

        max_number_of_samples = 100000

        # We need to know where a sequence ends for each recorded sequence

        metadata_labels = f5["/dynamic/changes/metadata/column_annotations"][...]
        sequence_index = convert_binary_string_array(metadata_labels).index("_sequence_i")

        metadata_array = f5["/dynamic/changes/metadata/column_annotations"]

        data_ds = f5["/dynamic/changes/data/core_array"]
        print(data_ds.shape)
        _, n_sequence_len, n_types = data_ds.shape

        with h5py.File(output_file_name, "w") as f5w:

            # We will store the samples so we can use for later transforms

            samples_group = f5w.create_group("/data/samples")
            samples_ds = samples_group.create_dataset("measures", shape=(max_number_of_samples, n_types))
            samples_len_ds = samples_group.create_dataset("n", shape=(1, n_types))

            seq_len_group = f5w.create_group("/data/sequence_length/")
            seq_len_ds = seq_len_group.create_dataset("all", shape=(1, n_size))

            for j in range(n_types):
                data_list = []
                for i in range(start_position_train, end_position_test):
                    if len(data_list) > max_number_of_samples:
                        break
                    else:
                        data_slice = data_ds[i,:,j].tolist()

                        if 0 in data_slice:
                            sequence_end = data_slice.index(0) - 1 # TODO: We need to check the sequence_i max
                        else:
                            sequence_end = len(data_slice)

                        data_items = [ds for ds in data_slice[0:sequence_end] if not(np.isnan(ds))]

                        if len(data_items):
                            data_list += data_items

                data_list = data_list[0:max_number_of_samples]
                number_of_data_items = len(data_list)
                samples_ds[0:number_of_data_items, j] = np.array(data_list)
                samples_len_ds[0, j] = number_of_data_items

                print(j)

        # We will compute the 99 and 1 percentiles
        # Values will larger and small will be truncated

        # Measurements that occur less than 0.01 will be dropped








if __name__ == "__main__":


    main("Y:\\healthfacts\\ts\\ohdsi_sequences.hdf5", "Y:\\healthfacts\\ts\\processed_ohdsi_sequences.hdf5")