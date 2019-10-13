
import argparse
import h5py


def get_all_paths(h5py_group):
    """Recurse and get all non-groups"""
    non_groups = []
    for group_name in h5py_group:
        if not h5py_group[group_name].__class__ == h5py_group.__class__:
            non_groups += [h5py_group[group_name].name]
        else:
            non_groups.extend(get_all_paths(h5py_group[group_name]))

    if len(non_groups):
        return non_groups


def main(hdf5_file_name, number_of_samples):

    with h5py.File(hdf5_file_name, mode="r") as f5r:

        h5_paths = get_all_paths(f5r["/"])

        core_paths = [x for x in h5_paths if "/core_array" in x]
        print(core_paths)

        non_core_paths = [x for x in h5_paths if "/core_array" not in x]
        print(non_core_paths)

        with h5py.File(hdf5_file_name + ".subset.hdf5", "w") as f5w:

            for path in core_paths:
                core_array_shape = f5r[path].shape
                core_array_dtype = f5r[path].dtype
                core_array_compression = f5r[path].compression

                if len(core_array_shape) == 2:
                    core_array = f5r[path][0:number_of_samples, :]
                    new_shape = (number_of_samples, core_array_shape[1])
                elif len(core_array_shape) == 3:
                    core_array = f5r[path][0:number_of_samples, :, :]
                    new_shape = (number_of_samples, core_array_shape[1], core_array_shape[2])

                new_dataset = f5w.create_dataset(path, shape=new_shape, dtype=core_array_dtype, compression=core_array_compression)
                new_dataset[...] = core_array[...]

            for path in non_core_paths:

                ds_shape = f5r[path].shape
                ds_dtype = f5r[path].dtype
                ds_compression = f5r[path].compression

                new_dataset = f5w.create_dataset(path, shape=ds_shape, dtype=ds_dtype, compression=ds_compression)
                new_dataset[...] = f5r[path][...]


if __name__ == "__main__":

    # arg_parse_obj = argparse.ArgumentParser(description="Subset the dynamic and static HDF5 file")
    #
    # arg_parse_obj.add_argument("-f", "--hdf5-file-name", dest="hdf5_file_name")
    # arg_parse_obj.add_argument("-n", "--n-subsets", dest="n_subsets", default="1000")
    #
    # arg_obj = arg_parse_obj.parse_args()
    #
    # main(arg_obj.hdf5_file_name, int(arg_obj.n_subsets))

    main("Y:\\healthfacts\\ts\\output\\ohdsi_sequences.hdf5", 5)