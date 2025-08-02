import os  # 访问操作系统功能模块
import sys  # 和python解释器交互的模块
import pickle
import numpy as np

from keras.optimizer_v2.rmsprop import RMSprop
from keras.optimizer_v2.adam import Adam
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, GridSearchCV
from traj_predict import Predict
from test_daimler import daimler_test, draw_tte, draw_tte_all, avarge

import keras.backend as K  # 后端

from prettytable import PrettyTable  # 生成ASCII格式的表格

#dim_ordering = K.image_dim_ordering()  # 返回默认的图像的维度顺序（‘tf’或‘th’）
dim_ordering = K.set_image_data_format('channels_last')


def load_dict(type):
    root = '../data/daimler/data_flow_sampling/'
    with open(root + type + '.pkl', 'rb') as f:
        return pickle.load(f)


def data_split(data, index):
    image1, pid1, tte1, bbox1, speed1, p_image1, p_pid1, p_tte1, target1, opt1 = [], [], [], [], [], [], [], [], [], []

    for n1 in index:
        image1.append(data['obs_image'][n1])
        pid1.append(data['obs_pid'][n1])
        tte1.append(data['obs_tte'][n1])
        bbox1.append(data['enc_input'][n1])
        speed1.append(data['dec_input'][n1])
        p_image1.append(data['pred_image'][n1])
        p_pid1.append(data['pred_pid'][n1])
        p_tte1.append(data['pred_tte'][n1])
        target1.append(data['pred_target'][n1])

    return{'obs_image': image1,
           'obs_pid': pid1,
           'obs_tte': tte1,
           'enc_input': bbox1,
           'dec_input': speed1,
           'pred_image': p_image1,
           'pred_pid': p_pid1,
           'pred_tte': p_tte1,
           'pred_target': target1,
           'model_opts': data['model_opts']}


def data_split1(data, train_index, val_index):
    image1, pid1, bbox1, speed1, label1, res1, tte1, deg1 = [], [], [], [], [], [], [], []
    image2, pid2, bbox2, speed2, label2, res2, tte2, deg2 = [], [], [], [], [], [], [], []

    for n1 in train_index:
        image1.append(data['image'][n1])
        pid1.append(data['pid'][n1])
        bbox1.append(data['bbox'][n1])
        speed1.append(data['obd_speed'][n1])
        label1.append(data['label'][n1])
        res1.append(data['resolution'][n1])
        tte1.append(data['TTE'][n1])
        deg1.append(data['degree_pred'][n1])

    for n2 in val_index:
        image2.append(data['image'][n2])
        pid2.append(data['pid'][n2])
        bbox2.append(data['bbox'][n2])
        speed2.append(data['obd_speed'][n2])
        label2.append(data['label'][n2])
        res2.append(data['resolution'][n2])
        tte2.append(data['TTE'][n2])
        deg2.append(data['degree_pred'][n2])

    data_train = {}
    data_train['image'] = image1
    data_train['pid'] = pid1
    data_train['bbox'] = bbox1
    data_train['obd_speed'] = speed1
    data_train['label'] = label1
    data_train['resolution'] = res1
    data_train['TTE'] = tte1
    data_train['degree_pred'] = deg1

    data_val = {}
    data_val['image'] = image2
    data_val['pid'] = pid2
    data_val['bbox'] = bbox2
    data_val['obd_speed'] = speed2
    data_val['label'] = label2
    data_val['resolution'] = res2
    data_val['TTE'] = tte2
    data_val['degree_pred'] = deg2

    #print(len(data['bbox']), [len(data['bbox'][i]) for i in range(len(data['bbox']))])
    #print(len(data_train['bbox']), [len(data_train['bbox'][i]) for i in range(len(data_train['bbox']))])
    #print(len(data_val['bbox']), [len(data_val['bbox'][i]) for i in range(len(data_val['bbox']))])

    return data_train, data_val


def train(train_data, val_data,
          batch_size=64,
          epochs=80,
          lr=0.001,
          loss='mse',
          opt='rms',
          learning_scheduler=True,
          val=True,
          out_path='',
          t=None,
          **model_opts):

        optimizer = RMSprop(learning_rate=lr)
        if opt == 'adam':
            optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)

        print("Number of samples:\n Train: %d \n Val: %d \n"
              % (len(train_data['enc_input1']), len(val_data['enc_input1'])))

        t._observe_length = train_data['enc_input1'].shape[1]
        t._predict_length = train_data['pred_target'].shape[1]

        t._encoder_feature_size = train_data['enc_input1'].shape[2]
        t._encoder_feature_size_ori = train_data['enc_input2'].shape[2]
        t._encoder_feature_size_flow = train_data['enc_input3'].shape[2]
        t._decoder_feature_size = train_data['dec_input'].shape[2]

        # Set the output sizes
        t._prediction_size = train_data['pred_target'].shape[2]

        # Set path names for saving configs and model  设置保存配置和模型的路径名称
        model_path = out_path + 'model.h5'
        opts_path = out_path + 'model_opts.pkl'
        config_path = out_path + 'configs.txt'
        history_path = out_path + 'history.pkl'

        # Save data parameters  保存数据参数
        with open(opts_path, 'wb') as fid:
            pickle.dump(train_data['model_opts'], fid,
                        pickle.HIGHEST_PROTOCOL)

        # save training and model parameters  保存训练和模型参数
        t.log_configs(config_path, batch_size, epochs,
                      lr, loss, learning_scheduler,
                      train_data['model_opts'])

        pie_model = t.pie_encdec()

        # Generate training data  生成训练数据
        train_data = ([train_data['enc_input1'],
                       train_data['enc_input2'],
                       train_data['enc_input3'],
                       train_data['dec_input']],
                       train_data['pred_target'])

        if val:
            val_data = ([val_data['enc_input1'],
                         val_data['enc_input2'],
                         val_data['enc_input3'],
                         val_data['dec_input']],
                         val_data['pred_target'])

        pie_model.compile(loss=loss, optimizer=optimizer)  # 在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准

        print("##############################################")
        print(" Training for predicting sequences of size %d" % t._predict_length)
        print("##############################################")

        if val:
            checkpoint = ModelCheckpoint(filepath=model_path,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         monitor='val_loss')
        else:
            checkpoint = ModelCheckpoint(filepath=model_path,
                                         save_weights_only=False)

        call_backs = [checkpoint]

        #  Setting up learning schedulers  设置学习调度器
        if learning_scheduler and val:
            early_stop = EarlyStopping(monitor='val_loss',
                                       min_delta=1.0, patience=10,
                                       verbose=1)
            plateau_sch = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.2, patience=5,
                                            min_lr=1e-07, verbose=1)
            call_backs.extend([early_stop, plateau_sch])

        if val:
            history = pie_model.fit(x=train_data[0], y=train_data[1],
                                    batch_size=batch_size, epochs=epochs,
                                    validation_data=val_data, verbose=1,
                                    callbacks=call_backs)
        else:
            history = pie_model.fit(x=train_data[0], y=train_data[1],
                                    batch_size=batch_size, epochs=epochs,
                                    verbose=1, callbacks=call_backs)

        print('Train model is saved to {}'.format(model_path))

        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return out_path


def train_test_daimler(train_test=2,
                       traj_model_path='',
                       speed_model_path=''):

    traj_model_opts = {'normalize_bbox': True,
                       'bbox_type': 'Interframe_offset',
                       'track_overlap': 0.7,
                       'observe_length': 8,
                       'predict_length': 8,
                       'enc_input_type1': ['bbox'],
                       'enc_input_type2': ['degree_pred'],
                       'enc_input_type3': ['ego_op_flow', 'ped_op_flow'],
                       'dec_input_type': ['obd_speed'],
                       'prediction_type': ['bbox']
                       }

    if train_test == 1:
        data_train = load_dict('Train_aug')

        # param = [32, 64, 128, 256, 512, 1024]
        param = [256]
        for p in param:
            traj = Predict(num_hidden_units=p)

            data = traj.get_data(data_train, **traj_model_opts)

            # 五折交叉验证
            kf = KFold(n_splits=5)
            n = 0
            for train_index, val_index in kf.split(data['obs_image']):
                # print(train_index, val_index)
                n += 1

                # 训练
                train_data = traj.get_data(data_train, train_index, **traj_model_opts)
                val_data = traj.get_data(data_train, val_index, **traj_model_opts)
                out_path = traj_model_path + 'HiddenUnits_' + str(p) + '_KFold_' + str(n) + '/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                out_model_path = train(train_data, val_data, batch_size=32, epochs=100, opt='adam',
                                       out_path=out_path, val=True, t=traj, **traj_model_opts)

                # 测试
                beh_seq_test = load_dict('TestV')

                if 'obd_speed' in traj_model_opts['prediction_type']:
                    perf = traj.test(beh_seq_test, model_path=out_model_path)
                else:
                    perf = daimler_test(beh_seq_test, model_path=out_model_path, speed_model_path=speed_model_path,
                                        bbox_type=traj_model_opts['bbox_type'])

                disp = PrettyTable(['MSE_8', 'MSE_last'])
                disp.title = 'Trajectory prediction model - Daimler (loc_offset_ori)'
                disp.add_row([perf['mse'], perf['mse_last']])

                print(disp)

                draw_tte(perf, model_path=out_model_path)

    if train_test == 2:
        traj = Predict()
        beh_seq_test = load_dict('TestV')

        if 'obd_speed' in traj_model_opts['prediction_type']:
            perf = traj.test(beh_seq_test, model_path=traj_model_path)
        else:
            perf = daimler_test(beh_seq_test, model_path=traj_model_path, speed_model_path=speed_model_path,
                                bbox_type=traj_model_opts['bbox_type'])

        t = PrettyTable(['MSE_8', 'MSE_last'])
        t.title = 'Trajectory prediction model - Daimler (loc)'
        t.add_row([perf['mse'], perf['mse_last']])

        print(t)

        draw_tte(perf, model_path=traj_model_path)


def main(train_test=2):
    out_model_path = 'model/daimler/trajectory/enc_2/loc_ori_speed_flow/'
    speed_model_path = 'model/daimler/speed/speed_8_8_0.7'
    train_test_daimler(train_test=train_test,
                       traj_model_path=out_model_path, speed_model_path=speed_model_path)

    #draw_tte_all(model_path=out_model_path)
    avarge(out_model_path)


if __name__ == '__main__':
    try:
        # train_test = int(sys.argv[1])
        train_test = 1
        main(train_test=train_test)
    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n')