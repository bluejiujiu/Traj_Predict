import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from traj_predict import Predict


def load_dict(name):
    root = 'model/daimler/trajectory/enc_2/'
    with open(root + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def draw_param(model_path):
    s = model_path.split('/')
    perf = load_dict(s[-2] + '/' + s[-2] + '_cross_val')
    param = [32, 64, 128, 256, 512, 1024]
    y_data = []
    for p in param:
        y_data.append(perf['HiddenUnits_' + str(p) + '_mse(mean)'])

    x_data = range(1, 7, 1)
    plt.plot(x_data, y_data, color="blue", marker="*", )

    plt.xlabel('Dimension of LSTM hidden units')  # x_label
    plt.ylabel('MSE')  # y_label
    plt.ylim([100, 160])
    plt.xticks(x_data, ('32', '64', '128', '256', '512', '1024'))

    plt.savefig(model_path + '/LSTM.png')
    # plt.show()
    plt.clf()
    plt.close()


def avarge(model_path):
    s = model_path.split('/')
    #print(s)
    perf1 = {}
    #param = [32, 64, 128, 256, 512, 1024]
    param = [256]
    for p in param:
        p1, p2, p3, p4, p5 = [], [], [], [], []

        for n in range(1, 6):
            save_path = s[-2] + '/HiddenUnits_' + str(p) + '_KFold_' + str(n) + '/'
            perf = load_dict(save_path + 'TTE')
            p1.append(perf['performance']['mse'])
            p2.append(perf['performance']['mse_BendingIn'])
            p3.append(perf['performance']['mse_Crossing'])
            p4.append(perf['performance']['mse_Starting'])
            p5.append(perf['performance']['mse_Stopping'])

        perf1['HiddenUnits_' + str(p) + '_mse(mean)'] = np.array(p1).mean(axis=None)
        perf1['HiddenUnits_' + str(p) + '_mse_BendingIn(mean)'] = np.array(p2).mean(axis=None)
        perf1['HiddenUnits_' + str(p) + '_mse_Crossing(mean)'] = np.array(p3).mean(axis=None)
        perf1['HiddenUnits_' + str(p) + '_mse_Starting(mean)'] = np.array(p4).mean(axis=None)
        perf1['HiddenUnits_' + str(p) + '_mse_Stopping(mean)'] = np.array(p5).mean(axis=None)

    save_performance_path = os.path.join(model_path,
                                         s[-2] + '_cross_val.txt')
    save_results_path = os.path.join(model_path,
                                     s[-2] + '_cross_val.pkl')

    with open(save_performance_path, 'wt') as fid:
        for k in sorted(perf1.keys()):
            fid.write("%s: %s\n" % (k, str(perf1[k])))

    with open(save_results_path, 'wb') as fid:
        pickle.dump(perf1, fid, pickle.HIGHEST_PROTOCOL)


def draw_tte_all(model_path):
    root = 'model/daimler/trajectory/enc_2/'

    perf = load_dict('BendingIn_loc_offset_ori_speed/TTE')
    draw_tte(perf['performance'], model_path + 'BendingIn_loc_offset_ori_speed')
    perf = load_dict('Crossing_loc_offset_ori_speed/TTE')
    draw_tte(perf['performance'], model_path + 'Crossing_loc_offset_ori_speed')
    perf = load_dict('Starting_loc_offset_ori_speed/TTE')
    draw_tte(perf['performance'], model_path + 'Starting_loc_offset_ori_speed')
    perf = load_dict('Stopping_loc_offset_ori_speed/TTE')
    draw_tte(perf['performance'], model_path + 'Stopping_loc_offset_ori_speed')

    perf_loc = load_dict('loc/TTE')
    perf_loc_offset = load_dict('loc_offset/TTE')
    perf_loc_offset_ori = load_dict('loc_ori/TTE')
    perf_loc_offset_ori_speed = load_dict('loc_ori_speed/TTE')
    perf_loc_offset_ori_flow = load_dict('loc_ori_flow/TTE')
    perf_loc_offset_ori_speed_flow = load_dict('loc_ori_speed_flow/TTE')
    #print(perf_loc)

    draw_tte(perf_loc['performance'], model_path + 'loc')
    draw_tte(perf_loc_offset['performance'], model_path + 'loc_offset')
    draw_tte(perf_loc_offset_ori['performance'], model_path + 'loc_ori')
    draw_tte(perf_loc_offset_ori_speed['performance'], model_path + 'loc_ori_speed')
    draw_tte(perf_loc_offset_ori_flow['performance'], model_path + 'loc_ori_flow')
    draw_tte(perf_loc_offset_ori_speed_flow['performance'], model_path + 'loc_ori_speed_flow')

    x_data = [i for i in range(20, -20 - 1, -1)]
    y_data1 = perf_loc['performance']['mse_TTE']
    y_data2 = perf_loc_offset['performance']['mse_TTE']
    y_data3 = perf_loc_offset_ori['performance']['mse_TTE']
    y_data4 = perf_loc_offset_ori_speed['performance']['mse_TTE']
    y_data5 = perf_loc_offset_ori_flow['performance']['mse_TTE']
    y_data6 = perf_loc_offset_ori_speed_flow['performance']['mse_TTE']

    #plt.plot(x_data, y_data1, color="purple", marker=".", label="loc")
    #plt.plot(x_data, y_data2, color="red", marker=".", label="loc_offset")
    #plt.plot(x_data, y_data3, color="green", marker=".", label="loc_offset_ori")
    #plt.plot(x_data, y_data4, color="blue", marker=".", label="loc_offset_ori_speed")
    plt.plot(x_data, y_data2, color="purple", marker=".", label="X")
    plt.plot(x_data, y_data3, color="green", marker=".", label="X+O")
    plt.plot(x_data, y_data4, color="blue", marker=".", label="X+O+S")
    plt.plot(x_data, y_data5, color="cyan", marker=".", label="X+O+F")
    plt.plot(x_data, y_data6, color="red", marker=".", label="X+O+S+F")

    plt.legend()  # 显示label
    plt.xlabel('TTE')  # x_label
    plt.ylabel('MSE')  # y_label
    plt.xlim((20, -20))
    plt.ylim((0, 1000))
    plt.grid()

    plt.savefig(root + 'TTE.png')
    # plt.show()
    plt.clf()
    plt.close()


def draw_tte(perf, model_path):

    x_data = [i for i in range(20, -20-1, -1)]
    y_data = perf['mse_TTE']

    plt.plot(x_data, y_data, color="orange", marker=".",)

    #plt.legend(['length_8', 'length_16'])  # 显示label
    plt.xlabel('TTE')  # x_label
    plt.ylabel('MSE')  # y_label
    plt.xlim((20, -20))
    plt.ylim((0, 1000))
    plt.grid()

    plt.savefig(model_path + '/TTE.png')
    #plt.show()
    plt.clf()
    plt.close()


def daimler_test(data_test, model_path='', speed_model_path='', bbox_type='Interframe_offset'):
    """
    Daimler数据集测试评估
    Args:
        self:
        data_test: 测试数据
        model_path: 轨迹训练模型的保存路径
        speed_model_path: 车速预测模型的保存路径
        bbox_type: 边界框数据类型['Firstframe_offset', 'Interframe_offset']

    Returns:
        TTE在[-20, 20]区间内的 MES

    """
    test_model = load_model(os.path.join(model_path, 'model.h5'))
    test_model.summary()  # 输出模型各层的参数状况

    with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
        try:
            model_opts = pickle.load(fid)
        except:
            model_opts = pickle.load(fid, encoding='bytes')

    t = Predict()
    test_data = t.get_data(data_test, **model_opts)
    test_obs_data = [test_data['enc_input1'], test_data['enc_input2'],
                     test_data['enc_input3'], test_data['dec_input']]
    test_target_data = test_data['pred_target']

    print("Number of samples:\n Test: %d \n"
          % (test_data['enc_input1'].shape[0]))

    if 'obd_speed' in model_opts['dec_input_type']:
        # 车速预测
        speed_model = load_model(os.path.join(speed_model_path, 'model.h5'))

        ################## run speed model ####################
        model_opts['enc_input_type1'] = ['obd_speed']
        model_opts['enc_input_type2'] = []
        model_opts['dec_input_type'] = []
        model_opts['prediction_type'] = ['obd_speed']
        speed_data = t.get_data(data_test, **model_opts)

        _speed_data = [speed_data['enc_input1'], speed_data['dec_input']]
        speed_results = speed_model.predict(_speed_data,
                                            batch_size=2056,
                                            verbose=1)

        # speed
        test_results = test_model.predict([test_data['enc_input1'], test_data['enc_input2'],
                                           test_data['enc_input3'],
                                           speed_results],
                                          batch_size=2056, verbose=1)

    elif model_opts['dec_input_type'] == []:
        test_results = test_model.predict(test_obs_data, batch_size=2048, verbose=1)

    model_opts['normalize_bbox'] = False
    model_opts['dec_input_type'] = []
    model_opts['enc_input_type1'] = ['bbox']
    model_opts['enc_input_type2'] = []
    model_opts['prediction_type'] = ['bbox']
    test_data = t.get_data(data_test, **model_opts)
    test_obs_data_org = [test_data['enc_input1'], test_data['dec_input']]
    test_target_data_org = test_data['pred_target']

    if bbox_type == 'Interframe_offset':
        # 每帧位置坐标加上前一帧的坐标
        for j in range(len(test_results[0])):
            if j == 0:
                test_results[:, j, :] = test_results[:, j, :] + test_obs_data_org[0][:, -1, 0:4]
            else:
                test_results[:, j, :] = test_results[:, j, :] + test_results[:, j - 1, :]

    elif bbox_type == 'Firstframe_offset':
        # 每帧位置坐标加上第一帧的坐标
        test_results = test_results + np.expand_dims(test_obs_data_org[0][:, 0, 0:4], axis=1)

    # 边界框底部中心
    test_target_foot = np.zeros(shape=(test_target_data_org.shape[0], test_target_data_org.shape[1], 2))
    test_results_foot = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
    for b in range(test_results.shape[0]):
        for s in range(test_results.shape[1]):
            test_target_foot[b, s, 0] = (test_target_data_org[b, s, 0] + test_target_data_org[b, s, 2]) / 2
            test_target_foot[b, s, 1] = test_target_data_org[b, s, 3]
            test_results_foot[b, s, 0] = (test_results[b, s, 0] + test_results[b, s, 2]) / 2
            test_results_foot[b, s, 1] = test_results[b, s, 3]

    perf = {}
    # 边界框底部中心预测性能
    performance = np.square(test_target_foot - test_results_foot)
    perf['mse'] = performance.mean(axis=None)
    perf['mse_last'] = performance[:, -1, :].mean(axis=None)

    # 各个类别预测性能
    pid = test_data['pred_pid']
    bending_tru, bending_res, crossing_tru, crossing_res = [], [], [], []
    starting_tru, starting_res, stopping_tru, stopping_res = [], [], [], []
    for num, id in enumerate(pid):
        s = id[0][0]
        s = s.split('_')
        label_n = s[-1]
        # print(label_n)
        if label_n == 'BendingIn':
            bending_tru.append(test_target_foot[num])
            bending_res.append(test_results_foot[num])
        elif label_n == 'Crossing':
            crossing_tru.append(test_target_foot[num])
            crossing_res.append(test_results_foot[num])
        elif label_n == 'Starting':
            starting_tru.append(test_target_foot[num])
            starting_res.append(test_results_foot[num])
        elif label_n == 'Stopping':
            stopping_tru.append(test_target_foot[num])
            stopping_res.append(test_results_foot[num])
    #print(bending_tru)
    performance = np.square(np.array(bending_tru) - np.array(bending_res))
    perf['mse_BendingIn'] = performance.mean(axis=None)
    performance = np.square(np.array(crossing_tru) - np.array(crossing_res))
    perf['mse_Crossing'] = performance.mean(axis=None)
    performance = np.square(np.array(starting_tru) - np.array(starting_res))
    perf['mse_Starting'] = performance.mean(axis=None)
    performance = np.square(np.array(stopping_tru) - np.array(stopping_res))
    perf['mse_Stopping'] = performance.mean(axis=None)

    # TTE在[-20,20]间的性能
    TTE = test_data['pred_tte']
    TTE = np.array(TTE)
    #print(TTE)

    perf['TTE'] = [i for i in range(20, -20-1, -1)]
    perf['mse_TTE'], perf['mse_last_TTE'] = [], []

    for t in perf['TTE']:

        truth, results = [], []
        for b in range(TTE.shape[0]):
            tru, res = [], []
            for s in range(TTE.shape[1]):
                if TTE[b, s, 0] == t:
                    tru.append(test_target_foot[b, s, :])
                    res.append(test_results_foot[b, s, :])
            if len(tru) > 0 and len(res) > 0:
                truth.append(tru)
                results.append(res)

        truth = np.array(truth)
        results = np.array(results)
        performance = np.square(truth - results)
        #print(performance)
        perf['mse_TTE'].append(performance.mean(axis=None))
        #perf['mse_last_TTE'].append(performance[:, -1, :].mean(axis=None))

    # 保存测试结果
    save_results_path = os.path.join(model_path, 'TTE.pkl')
    save_performance_path = os.path.join(model_path,
                                         'TTE_' + '{:.2f}.txt'.format(perf['mse']))

    with open(save_performance_path, 'wt') as fid:
        for k in sorted(perf.keys()):
            fid.write("%s: %s\n" % (k, str(perf[k])))

    #if not os.path.exists(save_results_path):
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

