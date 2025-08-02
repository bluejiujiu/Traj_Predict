import os
import sys
import pickle
import numpy as np

from traj_predict import Predict
from degree_fen import classification, word_to_img, carla_data_process,  carla_spped_process

import keras.backend as K  #后端
import tensorflow as tf  #深度学习库
# print(tf.test.is_gpu_available())


from prettytable import PrettyTable  #生成ASCII格式的表格

# dim_ordering = K.image_dim_ordering()  #返回默认的图像的维度顺序（‘tf’或‘th’）
dim_ordering = K.set_image_data_format('channels_last')


def load_dict(type, dataset=''):
    root = '../data/' + dataset + '/data_add/'
    with open(root + type + '.pkl', 'rb') as f:
        return pickle.load(f)


def train(dataset='', train_test=2, train_type='loc_offset',
          traj_model_path='', speed_model_path=''):

    traj = Predict()#实例化类

    model_opts = {'normalize_bbox': True,
                  'bbox_type': 'Interframe_offset',
                  'track_overlap': 0.5,
                  'observe_length': 15,
                  'predict_length': 45,
                  'enc_input_type1': ['bbox'],
                  'enc_input_type2': [],#方向坐标
                  'enc_input_type3': [],#光流
                  'dec_input_type': [],
                  'prediction_type': ['bbox']
                  }

    if dataset == 'jaad':
        model_opts['track_overlap'] = 0.8

    if train_type == 'loc':
        model_opts['bbox_type'] = 'Firstframe_offset'
    elif train_type == 'loc_ori':
        model_opts['enc_input_type2'] = ['degree_pred']
    elif train_type == 'loc_ori_speed':
        model_opts['enc_input_type2'] = ['degree_pred']
        model_opts['dec_input_type'] = ['obd_speed']
    elif train_type == 'loc_flow':
        model_opts['enc_input_type2'] = []
        model_opts['enc_input_type3'] = ['ego_op_flow', 'ped_op_flow']
        model_opts['dec_input_type'] = []
    elif train_type == 'loc_ori_flow':
        model_opts['enc_input_type2'] = ['degree_pred']  # ['degree_pred', 'ego_op_flow', 'ped_op_flow']
        model_opts['enc_input_type3'] = ['ego_op_flow', 'ped_op_flow']
        model_opts['dec_input_type'] = []
    elif train_type == 'loc_speed_flow':
        model_opts['enc_input_type2'] = []  # ['degree_pred', 'ego_op_flow', 'ped_op_flow']
        model_opts['enc_input_type3'] = ['ego_op_flow', 'ped_op_flow']
        model_opts['dec_input_type'] = ['obd_speed']
    elif train_type == 'loc_ori_speed_flow':
        model_opts['enc_input_type2'] = ['degree_pred']#['degree_pred', 'ego_op_flow', 'ped_op_flow']
        model_opts['enc_input_type3'] = ['ego_op_flow', 'ped_op_flow']
        model_opts['dec_input_type'] = ['obd_speed']
    elif train_type == 'speed':
        model_opts['enc_input_type1'] = ['obd_speed']
        model_opts['prediction_type'] = ['obd_speed']

    # 配置分区，分区为1，不分区为0
    fen = 0 #0,1
    # 配置世界坐标系转换到图像坐标系，转换为1，不转换为0
    word_img = 0
    if train_test < 2:
        beh_seq_val = load_dict('val', dataset=dataset)
        beh_seq_train = load_dict('train', dataset=dataset)

        # if dataset == 'carla':
        #     beh_seq_val = carla_data_process(beh_seq_val)
        #     beh_seq_train = carla_data_process(beh_seq_train)
        if dataset == 'carla':
            beh_seq_val = carla_spped_process(beh_seq_val)
            beh_seq_train = carla_spped_process(beh_seq_train)

        # 配置光流
        if train_type == 'loc_flow' or train_type == 'loc_ori_speed_flow' or train_type == 'loc_ori_flow' or train_type == 'loc_speed_flow':
            beh_seq_train['ego_op_flow'] = np.load(f'../data/{dataset}/flow/flow_{dataset}_train_ego.npy', allow_pickle=True)
            beh_seq_train['ped_op_flow'] = np.load(f'../data/{dataset}/flow/flow_{dataset}_train_ped.npy', allow_pickle=True)
            beh_seq_val['ego_op_flow'] = np.load(f'../data/{dataset}/flow/flow_{dataset}_val_ego.npy', allow_pickle=True)
            beh_seq_val['ped_op_flow'] = np.load(f'../data/{dataset}/flow/flow_{dataset}_val_ped.npy', allow_pickle=True)

        # 方向分区
        if fen == 1:
            beh_seq_val = classification(beh_seq_val)
            beh_seq_train = classification(beh_seq_train)
        # 世界坐标系转换到图像坐标系
        if word_img == 1:
            beh_seq_val = word_to_img(beh_seq_val)
            beh_seq_train = word_to_img(beh_seq_train)
        traj_model_path = traj.train(beh_seq_train, beh_seq_val, batch_size=64, epochs=60,
                                     out_path=traj_model_path, **model_opts)

    if train_test > 0:
        beh_seq_test = load_dict('test', dataset=dataset)
        # if dataset == 'carla':
        #     beh_seq_test = carla_data_process(beh_seq_test)
        if dataset == 'carla':
            beh_seq_test = carla_spped_process(beh_seq_test)
        if train_type == 'loc_flow' or train_type == 'loc_ori_speed_flow' or train_type == 'loc_ori_flow' or train_type == 'loc_speed_flow':
            beh_seq_test['ego_op_flow'] = np.load(f'../data/{dataset}/flow_me/flow_{dataset}_test_ego.npy', allow_pickle=True)
            beh_seq_test['ped_op_flow'] = np.load(f'../data/{dataset}/flow_me/flow_{dataset}_test_ped.npy', allow_pickle=True)

        if fen == 1:
            beh_seq_test = classification(beh_seq_test)
        if word_img == 1:
            beh_seq_test = word_to_img(beh_seq_test)

        t = PrettyTable(['MSE', 'C_MSE', 'CF_MSE']) #定义了一个t的表格实例，并指定了两列的标题：'MSE' 和 'C_MSE'
        t.title = 'Trajectory prediction model ({})'.format(train_type)

        if train_type == 'loc' or train_type == 'speed':

            perf = traj.test(beh_seq_test, model_path=traj_model_path)
            t.add_row([perf['mse'], perf['center_mse'], perf['center_mse_last']])

        elif train_type == 'loc_offset' or train_type == 'loc_ori' or train_type == 'loc_flow' or train_type == 'loc_ori_flow':
            perf = traj.test_(beh_seq_test, model_path=traj_model_path)
            t.add_row([perf['mse'], perf['center_mse'], perf['center_mse_last']])

        elif train_type == 'loc_ori_speed' or train_type == 'loc_ori_speed_flow' or train_type == 'loc_speed_flow':
            perf = traj.test_loc_speed(beh_seq_test, traj_model_path=traj_model_path,
                                       speed_model_path=speed_model_path)
            t.add_row([perf['mse-45'], perf['c-mse-45'], perf['c-mse-last']])

        print(t)


if __name__ == '__main__':
    while True:  # 无限循环
        try:
            #train_test = int(sys.argv[1])
            train_test = 2

            dataset = 'pie'
            train_type = 'loc_ori' #[p ie：train_type = 'loc_ori_speed'，jaad：train_type = 'loc_ori']
            out_model_path = 'model/' + dataset + '/trajectory/enc_2/loc_ori_speed_30' #如果是jaad，则最后是loc_ori，如果是pie，则为loc_ori_speed
            speed_model_path = 'model/' + dataset + '/speed/speed_15_30/' #speed_model

            # dataset = 'carla'
            # train_type = 'loc'  # [pie：train_type = 'loc_ori_speed'，loc_ori_speed_flow，jaad：train_type = 'loc_ori']
            # out_model_path = 'model/' + dataset + '/trajectory/enc_2/loc/'  # 如果是jaad，则最后是loc_ori，如果是pie，则为loc_ori_speed
            # speed_model_path = 'model/' + dataset + '/speed/speed_model/'

            train(dataset=dataset, train_test=train_test, train_type=train_type,
                  traj_model_path=out_model_path, speed_model_path=speed_model_path)

        except ValueError:
            raise ValueError('Usage: python train_test.py <train_test>\n'
                             'train_test: 0 - train only, 1 - train and test, 2 -  test only\n')