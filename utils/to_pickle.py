import numpy as np
import pickle
import os

data_dir = "C:/Users/brugn_z632yho/Documents/EEG_train/"
pickle_dir = "C:/Users/brugn_z632yho/Documents/EEG_train/"
for r, d, f in os.walk(data_dir):
    for file in f:
        print(file)
        d = np.genfromtxt(data_dir + str(file), delimiter = ',')
        print(pickle_dir + str(file).split(".")[0] + ".pkl")
        with open(pickle_dir + str(file).split(".")[0] + ".pkl","wb") as pf:
            pickle.dump(d, pf)
