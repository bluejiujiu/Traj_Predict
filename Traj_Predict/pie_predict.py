import os
import time
import math
import pickle
import numpy as np

from keras.layers import Input, RepeatVector, Dense, Permute
from keras.layers import Concatenate, Multiply, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# from keras.optimizers import RMSprop
# from keras.optimizers import Adam
from keras.optimizer_v2.rmsprop import RMSprop
from keras.optimizer_v2.adam import Adam
from keras import regularizers

class PIEPredict(object):
    """
    An encoder decoder model for pedestrian trajectory prediction  行人轨迹预测的编码器-解码器模型

    Attributes:
       _num_hidden_units: Number of LSTM hidden units  LSTM隐藏单元数
       _regularizer_value: The value of L2 regularizer for training  用于训练的L2正则化器的值
       _regularizer: Training regularizer set as L2  训练正则器设置为L2
       _activation: LSTM actications  LSTM激活
       _embed_size: The size embedding unit applied to the representation produced by encoder  应用于编码器产生的表示的嵌入单元大小
       _embed_dropout: the dropout of embedding unit 嵌入单元的dropout

    Model attributes: The following attributes will be set during training depending on the training data  模型属性:在训练过程中，根据训练数据设置以下属性
       _observe_length: Observation duration in frames (number of time steps of the encoder)  帧内观测持续时间(编码器的时间步长)
       _predict_length: Prediciton duration in frames (number of time steps of the decoder)  帧中的预测持续时间(解码器的时间步数)
       _encoder_feature_size: The number of data points entering the encoder, e.g. for 'bounding boxes' the size is 4 (x1 y1 x2 y2)
         进入编码器的数据点数量，例如，对于“bounding boxes”，大小为4 (x1 y1 x2 y2)
       _decoder_feature_size: The number of data points entering the decoder, e.g. for 'speed' the size is 1
         进入解码器的数据点的数量，例如“speed”的大小为1
       _prediction_size: The size of the decoder output after dense layer, e.g. for 'bounding boxes' the size is 4 (x1 y1 x2 y2)
         密集层后解码器输出的大小，例如，对于'bounding boxes'，大小为4 (x1 y1 x2 y2)

    Methods:
        get_tracks: Generates trajectory tracks by sampling from pedestrian sequences  通过从行人序列中采样生成轨迹
        get_data_helper: Create training data by combining sequences with different modalities  通过组合具有不同模态的序列来创建训练数据
        get_data: Generates training and testing data  生成训练和测试数据
        log_configs: Writes model and training configurations to a file  将模型和训练配置写入文件
        train: Trains the model  训练模型
        test: Tests the model  测试模型
        pie_encdec: Generates the network model  生成网络模型
        create_lstm_model: A helper function for creating an LSTM unit  用于创建LSTM单元的辅助函数
        attention_temporal: Temporal attention custom layer  时间注意力自定义层
        attention_element: Elementwise attention custom layer  自我注意力自定义层
    """
    def __init__(self,
                 num_hidden_units=256,
                 regularizer_val=0.0001,
                 activation='softsign',
                 embed_size=64,
                 embed_dropout=0):

        # Network parameters
        self._num_hidden_units = num_hidden_units
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)

        self._activation = activation
        self._embed_size = embed_size
        self._embed_dropout = embed_dropout

        # model parameters
        self._observe_length = 15
        self._predict_length = 15

        self._encoder_feature_size = 4
        self._decoder_feature_size = 4

        self._prediction_size = 4

    def get_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize, bbox_type='Interframe_offset'):
        """
        Generates tracks by sampling from pedestrian sequences  从行人序列中采样生成轨迹
        :param dataset: The raw data passed to the method  传递给方法的原始数据
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
          编码器和解码器的数据类型的规范。数据类型取决于数据集。如JAAD有“bbox”，“center”，PIE此外还有“obd_speed”，“heading_angle”等。
        :param observe_length: The length of the observation (i.e. time steps of the encoder)  观察的长度(即编码器的时间步长)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)  预测的长度(即解码器的时间步长)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected  采样轨迹重叠的程度。应该选择[0,1)之间的值
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
          是否将中心/包围框坐标归一化，即转换为速度。注意:当轨道归一化，观测长度缩短1步，即第一步被移除。
        :return: A dictinary containing sampled tracks for each data modality  包含每个数据模态的采样轨迹的字典
        """
        #  Calculates the overlap in terms of number of frames  根据帧数计算重叠
        seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type  检查用户作为数据类型选择的键的有效性
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except KeyError:
                raise ('Wrong data type is selected %s' % dt)

        d['image'] = dataset['image']
        d['pid'] = dataset['pid']
        if 'TTE' in dataset.keys():
            d['TTE'] = dataset['TTE']

        #  Sample tracks from sequneces  序列的样本轨迹
        for k in d.keys():
            tracks = []
            n = 0
            for track in d[k]:
                tracks.extend([track[i:i + seq_length] for i in
                               range(0, len(track) - seq_length + 1, overlap_stride)])
                #if k == 'bbox' and n == 0:
                    #print(tracks)
                n += 1
            d[k] = tracks

        if 'degree_pred' in data_types:

            # 计算边界框底部中心的帧间差值
            bottom_center = np.zeros((len(d['bbox']), len(d['bbox'][0]), 2))
            bottom_offset = np.zeros((len(d['bbox']), len(d['bbox'][0]), 2))
            for i in range(len(d['bbox'])):
                for j in range(len(d['bbox'][0])):

                    bottom_center[i][j][:] = [d['bbox'][i][j][2] - (d['bbox'][i][j][2] - d['bbox'][i][j][0]) / 2,
                                              d['bbox'][i][j][3]]
                    if j > 0:
                        bottom_offset[i][j][:] = np.subtract(bottom_center[i][j][:], bottom_center[i][j - 1][:])

            # 计算距离r的方向坐标
            offset = np.zeros((len(d['degree_pred']), len(d['degree_pred'][0]), 4))
            for i in range(len(d['degree_pred'])):
                for j in range(len(d['degree_pred'][0])):
                    r = math.sqrt(math.pow(bottom_offset[i][j][0], 2) + math.pow(bottom_offset[i][j][1], 2))
                    body_dx = r * math.cos(math.radians(d['degree_pred'][i][j][0]))
                    body_dy = r * math.sin(math.radians(d['degree_pred'][i][j][0]))

                    head_dx = r * math.cos(math.radians(d['degree_pred'][i][j][1]))
                    head_dy = r * math.sin(math.radians(d['degree_pred'][i][j][1]))
                    #if i == 0 and j == 0:
                        #print(r, body_dx, body_dy, head_dx, head_dy)
                    offset[i][j][:] = [body_dx, body_dy, head_dx, head_dy]
            d['degree_pred'] = offset[:, 1:, :]
            #print(bottom_offset)
            #print(offset)
            #print(len(d['degree_pred']), len(d['degree_pred'][0]))

        #  Normalize tracks by subtracting bbox/center at first time step from the rest  通过从其余部分中减去bbox/center的第一步来规范化轨道
        if normalize:
            if 'bbox' in data_types:
                if bbox_type == 'Firstframe_offset':
                    for i in range(len(d['bbox'])):
                        d['bbox'][i] = np.subtract(d['bbox'][i][1:], d['bbox'][i][0]).tolist()
                elif bbox_type == 'Interframe_offset':
                    bbox_offset = np.zeros((len(d['bbox']), len(d['bbox'][0]), 4))
                    for i in range(len(d['bbox'])):
                        for j in range(len(d['bbox'][0])-1):
                            bbox_offset[i][j+1][:] = np.subtract(d['bbox'][i][j+1][:], d['bbox'][i][j][:])
                    #print(bbox_offset.shape)
                    d['bbox'] = bbox_offset[:, 1:, :]
                    #print(len(d['bbox']), len(d['bbox'][0]))
            
            if 'center' in data_types:
                for i in range(len(d['center'])):
                    d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()

            #  Adjusting the length of other data types  调整其他数据类型的长度
            for k in d.keys():
                if k != 'bbox' and k != 'center' and k != 'degree_pred':
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][1:]
        return d

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        用于生成数据的辅助函数，它将不同的数据类型组合到一个表示形式中
        :param data: A dictionary of different data types  包含不同数据类型的字典
        :param data_type: The data types defined for encoder and decoder input/output  为编码器和解码器输入/输出定义的数据类型
        :return: A unified data representation as a list  以列表形式表示的统一数据
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))

        #  Concatenate different data points into a single representation  将不同的数据点连接到单个表示中
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        else:
            return d[0]

    def data_split(self, data, index):
        image1, pid1, bbox1, speed1, tte1, deg1 = [], [], [], [], [], []

        for n1 in index:
            image1.append(data['image'][n1])
            pid1.append(data['pid'][n1])
            bbox1.append(data['bbox'][n1])
            if 'obd_speed' in data.keys():
                speed1.append(data['obd_speed'][n1])
            tte1.append(data['TTE'][n1])
            if 'degree_pred' in data.keys():
                deg1.append(data['degree_pred'][n1])

        data_ = {}
        data_['image'] = image1
        data_['pid'] = pid1
        data_['bbox'] = bbox1
        if 'obd_speed' in data.keys():
            data_['obd_speed'] = speed1
        data_['TTE'] = tte1
        if 'degree_pred' in data.keys():
            data_['degree_pred'] = deg1

        # print(len(data['bbox']), [len(data['bbox'][i]) for i in range(len(data['bbox']))])
        # print(len(data_train['bbox']), [len(data_train['bbox'][i]) for i in range(len(data_train['bbox']))])
        # print(len(data_val['bbox']), [len(data_val['bbox'][i]) for i in range(len(data_val['bbox']))])

        return data_

    def get_data(self, data, index=None, **model_opts):
        """
        Main data generation function for training/testing  训练/测试的主要数据生成函数
        :param data: The raw data  原始数据
        :param model_opts: Control parameters for data generation characteristics (see below for default values)  数据生成特征的控制参数(默认值见下文)
        :return: A dictionary containing training and testing data  包含训练和测试数据的字典
        """
        opts = {
            'normalize_bbox': True,
            'bbox_type': 'Interframe_offset',
            'track_overlap': 0.5,
            'observe_length': 15,
            'predict_length': 45,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }

        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_tracks(data, data_types, observe_length,
                                      opts['predict_length'], opts['track_overlap'],
                                      opts['normalize_bbox'], opts['bbox_type'],)

        if opts['normalize_bbox']:
            observe_length -= 1

        obs_slices = {}
        pred_slices = {}

        #  Generate observation/prediction sequences from the tracks  从航迹生成观测/预测序列
        for k in data_tracks.keys():
            #print(k)
            #print(np.array(data_tracks[k]).shape)
            obs_slices[k] = []
            pred_slices[k] = []
            obs_slices[k].extend([d[0:observe_length] for d in data_tracks[k]])
            pred_slices[k].extend([d[observe_length:] for d in data_tracks[k]])

        if index is not None:
            obs_slices = self.data_split(obs_slices, index)
            pred_slices = self.data_split(pred_slices, index)

        # Generate observation data input to encoder  生成观测数据输入到编码器
        enc_input = self.get_data_helper(obs_slices, opts['enc_input_type'])

        # Generate data for prediction decoder  为预测解码器生成数据
        dec_input = self.get_data_helper(pred_slices, opts['dec_input_type'])
        pred_target = self.get_data_helper(pred_slices, opts['prediction_type'])

        if not len(dec_input) > 0:
            dec_input = np.zeros(shape=pred_target.shape)

        if 'TTE' in obs_slices.keys():
            return {'obs_image': obs_slices['image'],
                    'obs_pid': obs_slices['pid'],
                    'obs_tte': obs_slices['TTE'],
                    'pred_image': pred_slices['image'],
                    'pred_pid': pred_slices['pid'],
                    'pred_tte': pred_slices['TTE'],
                    'enc_input': enc_input,
                    'dec_input': dec_input,
                    'pred_target': pred_target,
                    'model_opts': opts}
        else:
            return {'obs_image': obs_slices['image'],
                    'obs_pid': obs_slices['pid'],
                    'pred_image': pred_slices['image'],
                    'pred_pid': pred_slices['pid'],
                    'enc_input': enc_input,
                    'dec_input': dec_input,
                    'pred_target': pred_target,
                    'model_opts': opts}

    def log_configs(self, config_path, batch_size, epochs,
                    lr, loss, learning_scheduler, opts):
        """
        Logs the parameters of the model and training  记录模型参数和训练参数
        :param config_path: The path to save the file  文件保存路径
        :param batch_size: Batch size of training  训练批次大小
        :param epochs: Number of epochs for training  训练的epoch数
        :param lr: Learning rate of training  训练学习率
        :param loss: Type of loss function  损失函数类型
        :param learning_scheduler: Whether learning scheduler was used  是否使用学习调度器
        :param opts: Model options (see get_data)  模型选项(参见get_data)
        """
        # Save config and training param files  保存配置和训练参数文件
        with open(config_path, 'wt') as fid:
            fid.write("####### Model options #######\n")
            for k in opts:
                fid.write("%s: %s\n" % (k, str(opts[k])))

            fid.write("\n####### Network config #######\n")
            fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
            fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))
            fid.write("%s: %s\n" % ('activation', str(self._activation)))
            fid.write("%s: %s\n" % ('embed_size', str(self._embed_size)))
            fid.write("%s: %s\n" % ('embed_dropout', str(self._embed_dropout)))

            fid.write("%s: %s\n" % ('observe_length', str(self._observe_length)))
            fid.write("%s: %s\n" % ('predict_length ', str(self._predict_length)))
            fid.write("%s: %s\n" % ('encoder_feature_size', str(self._encoder_feature_size)))
            fid.write("%s: %s\n" % ('decoder_feature_size', str(self._decoder_feature_size)))
            fid.write("%s: %s\n" % ('prediction_size', str(self._prediction_size)))

            fid.write("\n####### Training config #######\n")
            fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
            fid.write("%s: %s\n" % ('epochs', str(epochs)))
            fid.write("%s: %s\n" % ('lr', str(lr)))
            fid.write("%s: %s\n" % ('loss', str(loss)))
            fid.write("%s: %s\n" % ('learning_scheduler', str(learning_scheduler)))

        print('Wrote configs to {}'.format(config_path))

    def train(self, data_train, data_val,
              batch_size=64,
              epochs=80,
              lr=0.001,
              loss='mse',
              opt='rms',
              learning_scheduler=True,
              val=True,
              out_path='',
              **model_opts):
        """
        Training method for the model  模型的训练方法
        :param data_train: Training data  训练数据
        :param data_val: Validation data  验证数据
        :param batch_size: Batch size of training  训练批次大小
        :param epochs: Number of epochs for training  训练的epoch数
        :param lr: Learning rate of training  训练学习率
        :param loss: Type of loss function  损失函数类型
        :param learning_scheduler: Whether learning scheduler was used  是否使用学习调度器
        :param model_opts: Data generation parameters (see get_data)  数据生成参数(参见get_data)
        :return: The path to where the final model is saved  保存最终模型的路径
        """

        optimizer = RMSprop(learning_rate=lr)
        if opt == 'adam':
            optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)

        train_data = self.get_data(data_train, **model_opts)
        if val:
            val_data = self.get_data(data_val, **model_opts)

            print("Number of samples:\n Train: %d \n Val: %d \n"
                  % (train_data['enc_input'].shape[0], val_data['enc_input'].shape[0]))
        else:
            print("Number of samples:\n Train: %d \n"
                  % (train_data['enc_input'].shape[0]))

        self._observe_length = train_data['enc_input'].shape[1]
        self._predict_length = train_data['pred_target'].shape[1]

        self._encoder_feature_size = train_data['enc_input'].shape[2]
        self._decoder_feature_size = train_data['dec_input'].shape[2]

        # Set the output sizes
        self._prediction_size = train_data['pred_target'].shape[2]

        # Set path names for saving configs and model  设置保存配置和模型的路径名称
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        model_path = out_path + 'model.h5'
        opts_path = out_path + 'model_opts.pkl'
        config_path = out_path + 'configs.txt'
        history_path = out_path + 'history.pkl'

        # Save data parameters  保存数据参数
        with open(opts_path, 'wb') as fid:
            pickle.dump(train_data['model_opts'], fid,
                        pickle.HIGHEST_PROTOCOL)

        # save training and model parameters  保存训练和模型参数
        self.log_configs(config_path, batch_size, epochs,
                         lr, loss, learning_scheduler,
                         train_data['model_opts'])

        pie_model = self.pie_encdec()

        # Generate training data  生成训练数据
        train_data = ([train_data['enc_input'],
                       train_data['dec_input']],
                       train_data['pred_target'])
        if val:
            val_data = ([val_data['enc_input'],
                        val_data['dec_input']],
                        val_data['pred_target'])

        pie_model.compile(loss=loss, optimizer=optimizer)  #在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准

        print("##############################################")
        print(" Training for predicting sequences of size %d" % self._predict_length)
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

    def test(self, data_test, model_path=''):
        """
        Testing method for the model  模型的测试方法
        :param data_test: Testing data  测试数据
        :param model_path: The path to where the model to be tested is saved  要测试的模型的保存路径
        :return: Mean squared error (MSE) of the prediction  预测的均方误差(MSE)
        """
        test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model.summary()  #输出模型各层的参数状况

        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        test_data = self.get_data(data_test, **model_opts)
        test_obs_data = [test_data['enc_input'], test_data['dec_input']]
        test_target_data = test_data['pred_target']

        print("Number of samples:\n Test: %d \n"
              % (test_data['enc_input'].shape[0]))

        test_results = test_model.predict(test_obs_data, batch_size=2048, verbose=1)

        perf = {}
        #  Performance on bounding boxes  边界框的性能
        performance = np.square(test_target_data - test_results)
        perf['mse'] = performance.mean(axis=None)
        perf['mse_last'] = performance[:, -1, :].mean(axis=None)

        # print("MSE  %f" % perf['mse'])
        # print("mse-15: %.2f\nmse-30: %.2f\nmse-45: %.2f"
        #       % (perf['mse-15'], perf['mse-30'], perf['mse']))
        # print("MSE last %f" % perf['mse_last'])

        if model_opts['prediction_type'][0] == 'bbox':

            if model_opts['predict_length'] == 45:
                perf['mse-15'] = performance[:, 0:15, :].mean(axis=None)
                perf['mse-30'] = performance[:, 0:30, :].mean(axis=None)

            #  Performance on centers (displacement)
            norm = model_opts['normalize_bbox']

            model_opts['normalize_bbox'] = False
            test_data = self.get_data(data_test, **model_opts)
            test_obs_data_org = [test_data['enc_input'], test_data['dec_input']]
            test_target_data_org = test_data['pred_target']

            if norm:
                results_org = test_results + np.expand_dims(test_obs_data_org[0][:, 0, 0:4], axis=1)
            else:
                results_org = test_results

            #  Performance measures for centers
            res_centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
            centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
            for b in range(test_results.shape[0]):
                for s in range(test_results.shape[1]):
                    centers[b, s, 0] = (test_target_data_org[b, s, 2] + test_target_data_org[b, s, 0]) / 2
                    centers[b, s, 1] = (test_target_data_org[b, s, 3] + test_target_data_org[b, s, 1]) / 2
                    res_centers[b, s, 0] = (results_org[b, s, 2] + results_org[b, s, 0]) / 2
                    res_centers[b, s, 1] = (results_org[b, s, 3] + results_org[b, s, 1]) / 2

            c_performance = np.square(centers - res_centers)
            perf['center_mse'] = c_performance.mean(axis=None)
            perf['center_mse_last'] = c_performance[:, -1, :].mean(axis=None)

            # # print("Center MSE  %f" % perf['center_mse'])
            # print("c-mse-15: %.2f\nc-mse-30: %.2f\nc-mse-45: %.2f"
            #       % (perf['c-mse-15'], perf['c-mse-30'], perf['center_mse']))
            # print("Center MSE last %f" % perf['center_mse_last'])

        save_results_path = os.path.join(model_path,
                                         '{:.2f}.pkl'.format(perf['mse']))
        save_performance_path = os.path.join(model_path,
                                         '{:.2f}.txt'.format(perf['mse']))

        with open(save_performance_path, 'wt') as fid:
            for k in sorted(perf.keys()):
                fid.write("%s: %s\n" % (k, str(perf[k])))

        if not os.path.exists(save_results_path):
            try:
                results = {'img_seqs': data_test['pred_image'],
                           'results': test_results,
                           'gt': test_target_data,
                           'performance': perf}
            except:
                results = {'img_seqs': [],
                           'results': test_results,
                           'gt': test_target_data,
                           'performance': perf}

            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)

        return perf

    def test_(self, data_test, model_path=''):
        """
        Testing method for the model  模型的测试方法(按照每帧减去前一帧的值进行归一化)
        :param data_test: Testing data  测试数据
        :param model_path: The path to where the model to be tested is saved  要测试的模型的保存路径
        :return: Mean squared error (MSE) of the prediction  预测的均方误差(MSE)
        """
        test_model = load_model(os.path.join(model_path, 'model.h5'))
        test_model.summary()  #输出模型各层的参数状况

        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        test_data = self.get_data(data_test, **model_opts)
        test_obs_data = [test_data['enc_input'], test_data['dec_input']]
        test_target_data = test_data['pred_target']

        print("Number of samples:\n Test: %d \n"
              % (test_data['enc_input'].shape[0]))

        test_results = test_model.predict(test_obs_data, batch_size=2048, verbose=1)

        model_opts['normalize_bbox'] = False
        test_data = self.get_data(data_test, **model_opts)
        test_obs_data_org = [test_data['enc_input'], test_data['dec_input']]
        test_target_data_org = test_data['pred_target']

        for j in range(len(test_results[0])):
            if j == 0:
                test_results[:, j, :] = test_results[:, j, :] + test_obs_data_org[0][:, -1, 0:4]
            else:
                test_results[:, j, :] = test_results[:, j, :] + test_results[:, j-1, :]

        perf = {}
        #  Performance on bounding boxes  边界框的性能
        performance = np.square(test_target_data_org - test_results)
        perf['mse'] = performance.mean(axis=None)
        perf['mse_last'] = performance[:, -1, :].mean(axis=None)
        if model_opts['predict_length'] == 45:
            perf['mse-15'] = performance[:, 0:15, :].mean(axis=None)
            perf['mse-30'] = performance[:, 0:30, :].mean(axis=None)

        # print("MSE  %f" % perf['mse'])
        # print("mse-15: %.2f\nmse-30: %.2f\nmse-45: %.2f"
        #       % (perf['mse-15'], perf['mse-30'], perf['mse']))
        # print("MSE last %f" % perf['mse_last'])

        if model_opts['prediction_type'][0] == 'bbox':
            #  Performance on centers (displacement)

            #  Performance measures for centers
            res_centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
            centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
            for b in range(test_results.shape[0]):
                for s in range(test_results.shape[1]):
                    centers[b, s, 0] = (test_target_data_org[b, s, 2] + test_target_data_org[b, s, 0]) / 2
                    centers[b, s, 1] = (test_target_data_org[b, s, 3] + test_target_data_org[b, s, 1]) / 2
                    res_centers[b, s, 0] = (test_results[b, s, 2] + test_results[b, s, 0]) / 2
                    res_centers[b, s, 1] = (test_results[b, s, 3] + test_results[b, s, 1]) / 2

            c_performance = np.square(centers - res_centers)
            perf['center_mse'] = c_performance.mean(axis=None)
            perf['center_mse_last'] = c_performance[:, -1, :].mean(axis=None)

            # # print("Center MSE  %f" % perf['center_mse'])
            # print("c-mse-15: %.2f\nc-mse-30: %.2f\nc-mse-45: %.2f"
            #       % (perf['c-mse-15'], perf['c-mse-30'], perf['center_mse']))
            # print("Center MSE last %f" % perf['center_mse_last'])

        save_results_path = os.path.join(model_path,
                                         '{:.2f}.pkl'.format(perf['mse']))
        save_performance_path = os.path.join(model_path,
                                         '{:.2f}.txt'.format(perf['mse']))

        with open(save_performance_path, 'wt') as fid:
            for k in sorted(perf.keys()):
                fid.write("%s: %s\n" % (k, str(perf[k])))

        if not os.path.exists(save_results_path):
            try:
                results = {'img_seqs': data_test['pred_image'],
                           'results': test_results,
                           'gt': test_target_data,
                           'performance': perf}
            except:
                results = {'img_seqs': [],
                           'results': test_results,
                           'gt': test_target_data,
                           'performance': perf}

            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)

        return perf

    def test_loc_speed(self, data_test, traj_model_path='', speed_model_path=''):

        with open(os.path.join(traj_model_path, 'model_opts.pkl'), 'rb') as fid:
            try:
                model_opts = pickle.load(fid)
            except:
                model_opts = pickle.load(fid, encoding='bytes')

        model_opts['dec_input_type'] = []
        box_data = self.get_data(data_test, **model_opts)
        print("Number of samples:\n Test: %d \n"
              % (box_data['enc_input'].shape[0]))

        #speed_path = '/home/aras/PycharmProjects/release_code/test/speed_models/pie_speed'
        speed_model = load_model(os.path.join(speed_model_path, 'model.h5'))

        #bis_path = '/home/aras/PycharmProjects/release_code/test/box_intent_speed'
        box_speed_model = load_model(os.path.join(traj_model_path, 'model.h5'))

        ################## run speed model ####################
        model_opts['enc_input_type'] = ['obd_speed']
        model_opts['prediction_type'] = ['obd_speed']
        speed_data = self.get_data(data_test, **model_opts)

        _speed_data = [speed_data['enc_input'], speed_data['dec_input']]
        speed_results = speed_model.predict(_speed_data,
                                            batch_size=2056,
                                            verbose=1)

        # speed
        test_results = box_speed_model.predict([box_data['enc_input'], speed_results], batch_size=2056, verbose=1)

        model_opts['normalize_bbox'] = False
        model_opts['enc_input_type'] = ['bbox']
        model_opts['prediction_type'] = ['bbox']
        test_data = self.get_data(data_test, **model_opts)
        test_obs_data_org = [test_data['enc_input'], test_data['dec_input']]
        test_target_data_org = test_data['pred_target']

        for j in range(len(test_results[0])):
            if j == 0:
                test_results[:, j, :] = test_results[:, j, :] + test_obs_data_org[0][:, -1, 0:4]
            else:
                test_results[:, j, :] = test_results[:, j, :] + test_results[:, j - 1, :]

        # Performance measures for bounding boxes
        perf = {}
        performance = np.square(test_results - test_target_data_org)
        perf['mse-15'] = performance[:, 0:15, :].mean(axis=None)
        perf['mse-30'] = performance[:, 0:30, :].mean(axis=None)  # 15:30
        perf['mse-45'] = performance.mean(axis=None)
        perf['mse-last'] = performance[:, -1, :].mean(axis=None)

        # print("mse-15: %.2f\nmse-30: %.2f\nmse-45: %.2f"
        #       % (perf['mse-15'], perf['mse-30'], perf['mse-45']))
        # print("mse-last %.2f\n" % (perf['mse-last']))

        #  Performance on centers (displacement)
        results_org = test_results

        #  Performance measures for centers
        res_centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
        centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
        for b in range(test_results.shape[0]):
            for s in range(test_results.shape[1]):
                centers[b, s, 0] = (test_target_data_org[b, s, 2] + test_target_data_org[b, s, 0]) / 2
                centers[b, s, 1] = (test_target_data_org[b, s, 3] + test_target_data_org[b, s, 1]) / 2
                res_centers[b, s, 0] = (results_org[b, s, 2] + results_org[b, s, 0]) / 2
                res_centers[b, s, 1] = (results_org[b, s, 3] + results_org[b, s, 1]) / 2

        c_performance = np.square(centers - res_centers)
        perf['c-mse-15'] = c_performance[:, 0:15, :].mean(axis=None)
        perf['c-mse-30'] = c_performance[:, 0:30, :].mean(axis=None)  # 0:30
        perf['c-mse-45'] = c_performance.mean(axis=None)
        perf['c-mse-last'] = c_performance[:, -1, :].mean(axis=None)

        # print("c-mse-15: %.2f\nc-mse-30: %.2f\nc-mse-45: %.2f" \
        #       % (perf['c-mse-15'], perf['c-mse-30'], perf['c-mse-45']))
        # print("c-mse-last: %.2f\n" % (perf['c-mse-last']))

        save_results_path = os.path.join(traj_model_path,
                                         '{:.2f}.pkl'.format(perf['mse-45']))
        save_performance_path = os.path.join(traj_model_path,
                                             '{:.2f}.txt'.format(perf['mse-45']))

        with open(save_performance_path, 'wt') as fid:
            for k in sorted(perf.keys()):
                fid.write("%s: %s\n" % (k, str(perf[k])))

        if not os.path.exists(save_results_path):
            try:
                results = {'img_seqs': box_data['pred_image'],
                           'results': test_results,
                           'gt': box_data['pred_target'],
                           'performance': perf}
            except:
                results = {'img_seqs': [],
                           'results': test_results,
                           'gt': box_data['pred_target'],
                           'performance': perf}
            with open(save_results_path, 'wb') as fid:
                pickle.dump(results, fid, pickle.HIGHEST_PROTOCOL)
        return perf

    def angle_transform(self, body, head):

        body_x = math.cos(math.radians(body*360))
        body_y = math.sin(math.radians(body*360))

        head_x = math.cos(math.radians(head*360))
        head_y = math.sin(math.radians(head*360))

        direction = [body_x, body_y, head_x, head_y]

        return direction

    def pie_encdec(self):
        """
        Generates the encoder decoder method  生成编码器解码器方法
        :return: An instance of the network model  网络模型的实例
        """

        # Generate input data. the shapes is (sequence_lenght,length of flattened features)
        _encoder_input = Input(shape=(self._observe_length, self._encoder_feature_size),
                               name='encoder_input')

        # Temporal attention module
        _attention_net = self.attention_temporal(_encoder_input, self._observe_length)

        # Generate Encoder LSTM Unit
        encoder_model = self.create_lstm_model(name='encoder_network')
        _encoder_outputs_states = encoder_model(_attention_net)
        _encoder_states = _encoder_outputs_states[1:]

        # Generate Decoder LSTM unit
        decoder_model = self.create_lstm_model(name='decoder_network', r_state=False)
        _hidden_input = RepeatVector(self._predict_length)(_encoder_states[0])
        _decoder_input = Input(shape=(self._predict_length, self._decoder_feature_size),
                               name='pred_decoder_input')

        # Embedding unit on the output of Encoder
        _embedded_hidden_input = Dense(self._embed_size, activation='relu')(_hidden_input)
        _embedded_hidden_input = Dropout(self._embed_dropout,
                                         name='dropout_dec_input')(_embedded_hidden_input)

        decoder_concat_inputs = Concatenate(axis=2)([_embedded_hidden_input, _decoder_input])

        # Self attention unit
        att_input_dim = self._embed_size + self._decoder_feature_size
        decoder_concat_inputs = self.attention_element(decoder_concat_inputs, att_input_dim)

        # Initialize the decoder with encoder states
        decoder_output = decoder_model(decoder_concat_inputs,
                                       initial_state=_encoder_states)
        decoder_output = Dense(self._prediction_size,
                               activation='linear',
                               name='decoder_dense')(decoder_output)

        net_model = Model(inputs=[_encoder_input, _decoder_input],
                          outputs=decoder_output)
        net_model.summary()

        return net_model

    def create_lstm_model(self, name='lstm', r_state=True, r_sequence=True):
        """
        A Helper function that generates an instance of LSTM  生成LSTM实例的辅助函数
        :param name: Name of the layer  层名
        :param r_state: Whether to return states  是否返回状态
        :param r_sequence: Whether to return sequences  是否返回序列
        :return: An LSTM instance  LSTM实例
        """

        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    activity_regularizer=None,
                    activation=self._activation,
                    name=name)

    # Custom layers
    def attention_temporal(self, input_data, sequence_length):
        """
        A temporal attention layer  时间注意力层
        :param input_data: Network input  网络输入
        :param sequence_length: Length of the input sequence  输入序列的长度
        :return: The output of attention layer  注意力层的输出
        """
        a = Permute((2, 1))(input_data)
        a = Dense(sequence_length, activation='sigmoid')(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = Multiply()([input_data, a_probs])
        return output_attention_mul

    def attention_element(self, input_data, input_dim):
        """
        A self-attention unit  自我注意单元
        :param input_data: Network input
        :param input_dim: The feature dimension of the input  输入的特征维数
        :return: The output of the attention network
        """
        input_data_probs = Dense(input_dim, activation='sigmoid')(input_data)  # sigmoid
        output_attention_mul = Multiply()([input_data, input_data_probs])  # name='att_mul'
        return output_attention_mul
