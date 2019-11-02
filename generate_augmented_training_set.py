# Under sample negative augmentation (20 %)

# Augmentation (positive cases added random noise)
# 10 percent

# 100 factor variable

import h5py
import numpy as np
import argparse
from shutil import copyfile


def main(hdf5_file_name, ratio, uniform_error, nearest_integer):

    # If value is 0 we will not inject noise

    print("Copy file '%s'" % hdf5_file_name)
    new_hdf5_file_name = hdf5_file_name + ".augmented.hdf5"
    copyfile(hdf5_file_name, new_hdf5_file_name)

    print("Begin augmenting data")

    target_array_path = "/data/processed/train/target/core_array"
    sequence_array_path = "/data/processed/train/sequence/core_array"

    with h5py.File(hdf5_file_name) as f5:
        with h5py.File(new_hdf5_file_name, "a") as f5w:

            target_array = f5[target_array_path][...]

            mask_positive_array = np.ravel(target_array == 1)
            mask_negative_array = np.ravel(target_array == 0)

            sequence_array_shape = f5[target_array_path].shape
            positive_cases = f5[sequence_array_path][mask_positive_array, :, :]
            number_of_positive_cases = positive_cases.shape[0]

            number_of_elements = positive_cases.shape[1] * positive_cases.shape[2]
            array_shape = (positive_cases.shape[1], positive_cases.shape[2])

            # negative_cases = f5[sequence_array_shape][mask_negative_array, :, :]
            # number_of_negative_cases = negative_cases.shape[0]

            f5w[target_array_path][...] = 0  # Set array to zero

            number_of_cases = f5w[sequence_array_path].shape[0]

            for i in range(number_of_positive_cases * ratio):

                i_pos_sample = np.random.randint(0, number_of_positive_cases-1)

                positive_case = positive_cases[i_pos_sample, :, :]
                #print(number_of_elements)
                random_noise = 1 + np.reshape(np.random.uniform(0, uniform_error, number_of_elements), array_shape)
                positive_case_with_error = positive_case * random_noise  # if 0 error will error out

                if nearest_integer:
                    positive_case_with_error = np.array(positive_case_with_error, dtype="int32")

                case_to_replace = np.random.randint(0, number_of_cases-1)

                f5w[target_array_path][case_to_replace, :] = 1
                f5w[sequence_array_path][case_to_replace, :, :] = positive_case_with_error

                if i > 0 and i % 100 == 0:
                    print("Resampled %s cases" % i)

            print("Total number of cases resampled: %s" % (i+1))


if __name__ == "__main__":

    arg_parse_obj = argparse.ArgumentParser(description="Create an augmented data set")
    #
    arg_parse_obj.add_argument("-f", "--hdf5-file-name", dest="hdf5_file_name")
    arg_parse_obj.add_argument("-n" "--over-sample-positive-case", dest="over_sample_positive_case", default=10000)
    arg_parse_obj.add_argument("-s", "--uniform-error", dest="uniform_error", default=0.05) # 0.05
    arg_parse_obj.add_argument("-i", "--round-to-nearest-integer", dest="round_to_nearest_integer", default=True, action="store_false")
    arg_obj = arg_parse_obj.parse_args()

    main(arg_obj.hdf5_file_name, arg_obj.over_sample_positive_case, arg_obj.uniform_error, arg_obj.round_to_nearest_integer)


    #main("prescriber_multi_year.split.hdf5", 1000, 0.05, True)