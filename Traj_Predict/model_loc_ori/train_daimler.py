import os  # 访问操作系统功能模块
import sys  # 和python解释器交互的模块
import pickle

from traj_predict import Predict
from test_daimler import daimler_test, draw_tte, draw_tte_all

import keras.backend as K  # 后端

from prettytable import PrettyTable  # 生成ASCII格式的表格

#dim_ordering = K.image_dim_ordering()  # 返回默认的图像的维度顺序（‘tf’或‘th’）
dim_ordering = K.set_image_data_format('channels_last')


def load_dict(type):
    root = '../data/daimler/data_sampling/'
    with open(root + type + '.pkl', 'rb') as f:
        return pickle.load(f)


def train_loc_ori(dataset='daimler',
                  train_test=2,
                  train_type='loc_offset',
                  traj_model_path='',
                  speed_model_path=''):

    traj = Predict()

    model_opts = {'normalize_bbox': True,
                  'bbox_type': 'Interframe_offset',
                  'track_overlap': 0.7,
                  'observe_length': 8,
                  'predict_length': 8,
                  'enc_input_type1': ['bbox'],
                  'enc_input_type2': [],
                  'dec_input_type': [],
                  'prediction_type': ['bbox']
                  }

    if train_type == 'loc':
        model_opts['bbox_type'] = 'Firstframe_offset'
    elif train_type == 'loc_ori':
        model_opts['enc_input_type2'] = ['degree_pred']
    elif train_type == 'loc_ori_speed':
        model_opts['enc_input_type2'] = ['degree_pred']
        model_opts['dec_input_type'] = ['obd_speed']
    elif train_type == 'speed':
        model_opts['enc_input_type1'] = ['obd_speed']
        model_opts['prediction_type'] = ['obd_speed']

    if train_test < 2:
        #beh_seq_val = load_dict('Val')
        beh_seq_val = []
        beh_seq_train = load_dict('Train_aug')

        traj_model_path = traj.train(beh_seq_train, beh_seq_val, batch_size=32, epochs=100, opt='adam',
                                     val=False, out_path=traj_model_path, **model_opts)

    if train_test > 0:
        beh_seq_test = load_dict('TestV')

        if 'obd_speed' in model_opts['prediction_type']:
            perf = traj.test(beh_seq_test, model_path=traj_model_path)
        else:
            perf = daimler_test(beh_seq_test, model_path=traj_model_path, speed_model_path=speed_model_path,
                                bbox_type=model_opts['bbox_type'])

        t = PrettyTable(['MSE_8', 'MSE_last'])
        t.title = 'Trajectory prediction model - Daimler ({})'.format(train_type)
        t.add_row([perf['mse'], perf['mse_last']])

        print(t)

        draw_tte(perf, model_path=traj_model_path)


def main(train_test=2):
    out_model_path = 'model/daimler/trajectory/enc_2/loc_ori_speed/'
    speed_model_path = 'model/daimler/speed/speed_8_8_0.7/'

    train_type = 'loc_ori_speed'
    train_loc_ori(dataset='daimler', train_test=train_test, train_type=train_type,
                 traj_model_path=out_model_path, speed_model_path=speed_model_path)

    draw_tte_all(model_path='model/daimler/trajectory/enc_2/')


if __name__ == '__main__':
    try:
        train_test = int(sys.argv[1])
        main(train_test=train_test)
    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n')