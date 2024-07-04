import h5py


class DataManager():
    def save(data , filename):
        with h5py.File(filename, 'w') as file:
            for key in data:
                file[key] = data[key]
            #file.close()
