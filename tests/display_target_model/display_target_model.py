import pickle
from os import listdir, path

if __name__ == "__main__":
    pathname = path.dirname(path.realpath(__file__))
    pathname_target_model = pathname + "/pickles/target_model/"
    for f in listdir(pathname_target_model):
        with open(pathname_target_model+f, "rb") as fp:
            pickle_var = pickle.load(fp)
        print(f)
        print(pickle_var['true_model'])
        print()