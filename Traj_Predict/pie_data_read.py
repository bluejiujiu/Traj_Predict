import os
import pickle

from pie_data import PIE


def save_dict(obj, data_type):
    root = 'data/pie/data_file/'
    if not os.path.exists(root):
        os.makedirs(root)
    with open(root + data_type + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def data_read(data_type):

    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                   'val_data': True,
                                   'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

    pie_path = 'F:/datasets/PIE-master'
    imdb = PIE(data_path=pie_path)

    data = imdb.generate_data_trajectory_sequence(data_type, **data_opts)

    save_dict(data, data_type)


if __name__ == '__main__':
    data_read('train')
    data_read('val')
    data_read('test')
