import pickle

def save(nn, file_name):
    with open(file_name, 'wb') as fh:
        pickle.dump(nn, fh)

def restore(file_name):
    nn = None
    with open(file_name, 'rb') as fh:
        nn = pickle.load(fh)
    return nn
