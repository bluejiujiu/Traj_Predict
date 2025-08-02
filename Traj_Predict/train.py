import os
import sys
import pickle

from pie_predict import PIEPredict

import keras.backend as K  #后端
import tensorflow as tf  #深度学习库
# print(tf.test.is_gpu_available())


from prettytable import PrettyTable  #生成ASCII格式的表格


# dim_ordering = K.image_dim_ordering()  #返回默认的图像的维度顺序（‘tf’或‘th’）
dim_ordering = K.set_image_data_format('channels_last')

def carla_spped_process(data):
    len_speed = len(data['obd_speed'])
    for i in range(len_speed):
        for j in range(len(data['obd_speed'][i])):
            data['obd_speed'][i][j] = [data['obd_speed'][i][j]]
    return data

def load_dict(type, dataset=''):
    root = 'data/' + dataset + '/data_add/'
    with open(root + type + '.pkl', 'rb') as f:
        return pickle.load(f)


def train(dataset='', train_test=2, train_type='loc_offset',
          traj_model_path='', speed_model_path=''):

    traj = PIEPredict()

    model_opts = {'normalize_bbox': True,
                  'bbox_type': 'Interframe_offset',
                  'track_overlap': 0.5,
                  'observe_length': 15,
                  'predict_length': 45,
                  'enc_input_type': ['bbox'],
                  'dec_input_type': [],
                  'prediction_type': ['bbox']
                  }

    if dataset == 'jaad':
        model_opts['track_overlap'] = 0.8

    if train_type == 'loc':
        model_opts['bbox_type'] = 'Firstframe_offset'
    elif train_type == 'loc_ori':
        model_opts['enc_input_type'] = ['bbox', 'degree_pred']
    elif train_type == 'loc_ori_speed':
        model_opts['enc_input_type'] = ['bbox', 'degree_pred']
        model_opts['dec_input_type'] = ['obd_speed']
    elif train_type == 'speed':
        model_opts['enc_input_type'] = ['obd_speed']
        model_opts['prediction_type'] = ['obd_speed']

    if train_test < 2:
        beh_seq_val = load_dict('val', dataset=dataset)
        beh_seq_train = load_dict('train', dataset=dataset)
        if dataset == 'carla':
            beh_seq_val = carla_spped_process(beh_seq_val)
            beh_seq_train = carla_spped_process(beh_seq_train)

        traj_model_path = traj.train(beh_seq_train, beh_seq_val, batch_size=64, epochs=60,
                                     out_path=traj_model_path, **model_opts)

    if train_test > 0:
        beh_seq_test = load_dict('test', dataset=dataset)
        if dataset == 'carla':
            beh_seq_test = carla_spped_process(beh_seq_test)

        t = PrettyTable(['MSE', 'C_MSE'])
        t.title = 'Trajectory prediction model ({})'.format(train_type)

        if train_type == 'loc' or train_type == 'speed':

            perf = traj.test(beh_seq_test, model_path=traj_model_path)
            # t.add_row([perf['mse'], perf['center_mse']])
            t.add_row([perf['mse'], perf['mse_last']])

        elif train_type == 'loc_offset' or train_type == 'loc_ori':
            perf = traj.test_(beh_seq_test, model_path=traj_model_path)
            t.add_row([perf['mse'], perf['center_mse']])

        elif train_type == 'loc_ori_speed':
            perf = traj.test_loc_speed(beh_seq_test, traj_model_path=traj_model_path,
                                       speed_model_path=speed_model_path)
            t.add_row([perf['mse-45'], perf['c-mse-45']])

        print(t)


if __name__ == '__main__':
    try:
        # train_test = int(sys.argv[1])
        train_test = 1
        dataset = 'carla' #pie
        train_type = 'speed'
        out_model_path = 'model/' + dataset + '/trajectory/loc_ori/'
        speed_model_path = 'model/' + dataset + '/speed/speed_model/'

        train(dataset=dataset, train_test=train_test, train_type=train_type,
              traj_model_path=speed_model_path, speed_model_path=speed_model_path)

    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n')