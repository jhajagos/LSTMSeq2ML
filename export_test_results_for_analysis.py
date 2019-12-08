import h5py
import csv
import argparse
import numpy as np

"""Export CSV results"""


def convert_binary_string_array(string_array):
    """Convert b'string' to 'python3 string'"""
    return [str(c, "utf8", errors="replace") for c in string_array.ravel().tolist()]

def main(processed_hdf5_file_name, csv_export_file_name, original_hdf5_file_name=None, hdf5_model_file_name=None):

    with h5py.File(processed_hdf5_file_name, "r") as f5:

        with open(csv_export_file_name, mode="w", newline="") as fw:
            csv_writer = csv.writer(fw)
            name_identifiers = convert_binary_string_array(f5["/data/processed/test/identifiers/column_annotations"][...])
            sequence_track_names = convert_binary_string_array(f5["/data/processed/test/sequence/column_annotations"][...])

            pre_header = ["row_id", "original_sequence_track_id", "sequence_track_id", "original_event_id", "entity_id",
                      "class", "type", "name"]

            train_seq_ds = f5["/data/processed/test/sequence/core_array"]

            train_seq_shape = train_seq_ds.shape
            n_rows, n_time_steps, n_tracks = train_seq_shape

            time_steps_header = ["t_" + str(i) for i in range(n_time_steps)]
            time_steps_pos = [i for i in range(n_time_steps)]
            header = pre_header + time_steps_header

            csv_writer.writerow(header)

            for i_row in range(n_rows):
                sequence_array = train_seq_ds[i_row, :, :]
                tracks_sums = np.sum(sequence_array, axis=0)

                track_j = 0

                for i_track in range(n_tracks):

                    if int(tracks_sums[i_track]):
                        track_array = sequence_array[:, i_track]

                        row_dict = {"row_id": i_row, "original_sequence_track_id": i_track, "sequence_track_id": track_j}
                        row_dict["name"] = sequence_track_names[i_track]

                        for i_step in range(n_time_steps):
                            track_value = track_array[i_step]

                            if track_value != 0.0:
                                row_dict["t_" + str(i_step)] = track_value

                        row_to_write = [""] * len(header)

                        for field_name in row_dict:
                            field_position = header.index(field_name)
                            row_to_write[field_position] = row_dict[field_name]

                        csv_writer.writerow(row_to_write)

                        track_j += 1



if __name__ == "__main__":

    main("C:\\Users\\janos\\data\\ts\\healthfacts\\20191017\\processed_ohdsi_sequences.subset.hdf5",
         "C:\\Users\\janos\\data\\ts\\healthfacts\\20191017\\processed_ohdsi_sequences.subset.hdf5.csv")