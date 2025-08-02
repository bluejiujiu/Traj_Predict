import os  # 访问操作系统功能模块
import sys  # 和python解释器交互的模块
import pickle

from pie_predict import PIEPredict
from test_daimler import daimler_test, draw_tte, draw_tte_all
from degree_fen import classification, word_to_img

import keras.backend as K  # 后端

from prettytable import PrettyTable  # 生成ASCII格式的表格

#dim_ordering = K.image_dim_ordering()  # 返回默认的图像的维度顺序（‘tf’或‘th’）
dim_ordering = K.set_image_data_format('channels_last')


def load_dict(type):
    root = 'data/daimler/data_sampling/'
    with open(root + type + '.pkl', 'rb') as f:
        return pickle.load(f)


def train_loc_ori(dataset='daimler',
                  train_test=2,
                  train_type='loc_offset',
                  traj_model_path='',
                  speed_model_path=''):

    traj = PIEPredict()

    model_opts = {'normalize_bbox': True,
                  'bbox_type': 'Interframe_offset',
                  'track_overlap': 0.7,
                  'observe_length': 8,
                  'predict_length': 8,
                  'enc_input_type': ['bbox'],
                  'dec_input_type': [],
                  'prediction_type': ['bbox']
                  }

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

    # 配置分区，分区为1，不分区为0
    fen = 0  # 0,1
    # 配置世界坐标系转换到图像坐标系，转换为1，不转换为0
    word_img = 1

    if train_test < 2:
        #beh_seq_val = load_dict('Val')
        beh_seq_val = []
        beh_seq_train = load_dict('Train_aug')

        # 方向分区
        if fen == 1:
            # beh_seq_val = classification(beh_seq_val)
            beh_seq_train = classification(beh_seq_train)
        # 世界坐标系转换到图像坐标系
        if word_img == 1:
            # beh_seq_val = word_to_img(beh_seq_val)
            beh_seq_train = word_to_img(beh_seq_train)

        traj_model_path = traj.train(beh_seq_train, beh_seq_val, batch_size=32, epochs=100, opt='adam',
                                     val=False, out_path=traj_model_path, **model_opts)

    if train_test > 0:
        beh_seq_test = load_dict('TestV')

        # 方向分区
        if fen == 1:
            beh_seq_test = classification(beh_seq_test)
        # 世界坐标系转换到图像坐标系
        if word_img == 1:
            beh_seq_test = word_to_img(beh_seq_test)

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
    out_model_path = 'model/daimler/trajectory/loc_offset/'
    speed_model_path = 'model/daimler/speed/speed_8_8_0.7/'

    train_type = 'loc_offset'
    train_loc_ori(dataset='daimler', train_test=train_test, train_type=train_type,
                  traj_model_path=out_model_path, speed_model_path=speed_model_path)

    #draw_tte_all(model_path=out_model_path)


if __name__ == '__main__':
    try:
        train_test = int(sys.argv[1])
        main(train_test=train_test)
    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n')