import os
import pickle
import cv2
import numpy as np
import sys
from traj_predict import Predict
from degree_fen import classification, word_to_img, carla_data_process


def load_data(dataset=''):
    root = '../data/' + dataset + '/data_file/test'
    if dataset == 'daimler':
        root = '../data/' + dataset + '/data_sampling/TestV'
    with open(root + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_results(dataset=''):
    root = 'model/' + dataset + '/trajectory/enc_2/'
    if dataset == 'jaad':
        path_x = root + 'loc_offset/1177.50.pkl'
        path_x_o = root + 'loc_ori/1145.93.pkl'
        path_x_o_f = root + 'loc_ori_flow/997.08.pkl'
        results_x_o_s = {'results': [], 'gt': []}
        results_x_o_s_f = {'results': [], 'gt': []}
    elif dataset == 'pie':
        path_x = root + 'loc_offset/597.60.pkl'
        path_x_o = root + 'loc_ori/540.74.pkl'
        path_x_o_s = root + 'loc_ori_speed/503.17.pkl'
        path_x_o_f = root + 'loc_ori_flow/432.44.pkl'
        path_x_o_s_f = root + 'loc_ori_speed_flow/449.69.pkl'
    elif dataset == 'daimler':
        path_x = root + 'loc_offset/TTE.pkl'
        path_x_o = root + 'loc_ori/TTE.pkl'
        path_x_o_s = root + 'loc_ori_speed/TTE.pkl'
        path_x_o_f = root + 'loc_ori_flow/TTE.pkl'
        path_x_o_s_f = root + 'loc_ori_speed_flow/TTE.pkl'
    elif dataset == 'carla':
        path_x = root + 'loc_offset/796.37.pkl'
        path_x_o = root + 'loc_ori/787.22.pkl'
        path_x_o_s = root + 'loc_ori/787.22.pkl'
        path_x_o_f = root + 'loc_ori_speed/696.89.pkl'
        # results_x_o_s = {'results': [], 'gt': []}
        path_x_o_s_f = root + 'loc_ori_speed/662.59.pkl'
    with open(path_x, 'rb') as f:
        results_x = pickle.load(f)
    with open(path_x_o, 'rb') as f:
        results_x_o = pickle.load(f)
    with open(path_x_o_f, 'rb') as f:
        results_x_o_f = pickle.load(f)
    if dataset != 'jaad':
        with open(path_x_o_s, 'rb') as f:
            results_x_o_s = pickle.load(f)
        with open(path_x_o_s_f, 'rb') as f:
            results_x_o_s_f = pickle.load(f)
    return results_x, results_x_o, results_x_o_s, results_x_o_f, results_x_o_s_f

def carla_draw(seq, dataset='carla'):
    t = Predict()

    # 读取测试数据
    data_test = load_data(dataset=dataset)
    # data_test = carla_data_process(data_test)
    opts = {
        'normalize_bbox': False,
        'bbox_type': 'Interframe_offset',
        'track_overlap': 0.5,
        'observe_length': 15,
        'predict_length': 45,
        'enc_input_type1': ['bbox'],
        'enc_input_type2': [],
        'dec_input_type': [],
        'prediction_type': ['bbox']
    }
    test_data = t.get_data(data_test, **opts)

    # 获取预测数据
    results_x, results_x_o, results_x_o_s, results_x_o_f, results_x_o_s_f = load_results(dataset=dataset)
    print(len(test_data['obs_image']), len(test_data['obs_image'][0]))
    #print(len(test_data['enc_input1']), len(test_data['enc_input1'][0]))
    print(results_x['results'].shape)
    #print(results_x['gt'].shape)

    for idx in range(seq[0], seq[1]):
        image_15 = test_data['obs_image'][idx][-1]  # 观测最后一帧的图像
        pid_15 = test_data['obs_pid'][idx][-1][0]  # 观测最后一帧的行人
        bbox_15 = test_data['enc_input1'][idx][-1]  # 观测最后一帧的边界框
        image_45 = test_data['pred_image'][idx][-1]  # 预测最后一帧的图像
        pid_45 = test_data['pred_pid'][idx][-1][0]  # 预测最后一帧的行人
        pred_x = results_x['results'][idx]  # 未来预测轨迹45*4（仅输入边界框）
        pred_x_o = results_x_o['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向）
        pred_x_o_f = results_x_o_f['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向+光流）
        pred_x_o_s_f = results_x_o_s_f['results'][idx]

        # 未来真实轨迹45*4
        truth = results_x['gt'][idx]
        for j in range(len(truth)):
            if j == 0:
                truth[j, :] = truth[j, :] + bbox_15
            else:
                truth[j, :] = truth[j, :] + truth[j - 1, :]

        print(image_15, image_45)
        #print(bbox_15)
        #print(truth)
        #print(pred_x_o)

        # 观测行人边界框可视化
        s = image_15.split('/')
        save_path = 'result_image/' + dataset + '/' + s[-3]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = cv2.imread('/'.join(s))
        box = np.array(bbox_15)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 3)
        # 保存图片
        s1 = s[-1].split('.')
        cv2.imwrite(os.path.join(save_path, s1[0] + '_' + pid_15 + '_obs.png'), img)

        # 预测行人边界框可视化
        s = image_45.split('/')
        img = cv2.imread('/'.join(s))
        box = np.array(truth[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
        box = np.array(pred_x[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)
        box = np.array(pred_x_o[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 3)
        box = np.array(pred_x_o_f[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 3)  # 青色
        box = np.array(pred_x_o_s_f[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)  # 红色
        # 保存图片
        s1 = s[-1].split('.')
        cv2.imwrite(os.path.join(save_path, s1[0] + '_' + pid_45 + '_pred.png'), img)

def jaad_draw(seq, dataset='jaad'):
    t = Predict()

    # 读取测试数据
    data_test = load_data(dataset=dataset)
    opts = {
        'normalize_bbox': False,
        'bbox_type': 'Interframe_offset',
        'track_overlap': 0.8,
        'observe_length': 15,
        'predict_length': 45,
        'enc_input_type1': ['bbox'],
        'enc_input_type2': [],
        'dec_input_type': [],
        'prediction_type': ['bbox']
    }
    test_data = t.get_data(data_test, **opts)

    # 获取预测数据
    results_x, results_x_o, results_x_o_s, results_x_o_f, results_x_o_s_f = load_results(dataset=dataset)
    print(len(test_data['obs_image']), len(test_data['obs_image'][0]))
    #print(len(test_data['enc_input1']), len(test_data['enc_input1'][0]))
    print(results_x['results'].shape)
    #print(results_x['gt'].shape)

    for idx in range(seq[0], seq[1]):
        image_15 = test_data['obs_image'][idx][-1]  # 观测最后一帧的图像
        pid_15 = test_data['obs_pid'][idx][-1][0]  # 观测最后一帧的行人
        bbox_15 = test_data['enc_input1'][idx][-1]  # 观测最后一帧的边界框
        image_45 = test_data['pred_image'][idx][-1]  # 预测最后一帧的图像
        pid_45 = test_data['pred_pid'][idx][-1][0]  # 预测最后一帧的行人
        pred_x = results_x['results'][idx]  # 未来预测轨迹45*4（仅输入边界框）
        pred_x_o = results_x_o['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向）
        pred_x_o_f = results_x_o_f['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向+光流）

        # 未来真实轨迹45*4
        truth = results_x['gt'][idx]
        for j in range(len(truth)):
            if j == 0:
                truth[j, :] = truth[j, :] + bbox_15
            else:
                truth[j, :] = truth[j, :] + truth[j - 1, :]

        print(image_15, image_45)
        #print(bbox_15)
        #print(truth)
        #print(pred_x_o)

        # 观测行人边界框可视化
        s = image_15[0].split('/')
        save_path = 'result_image/' + dataset + '/' + s[3]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = cv2.imread(os.path.join('/home/data/zmk/', s[1], s[2], s[3], s[4]))
        box = np.array(bbox_15)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 3)
        # 保存图片
        s1 = s[-1].split('.')
        cv2.imwrite(os.path.join(save_path, s1[0] + '_' + pid_15 + '_obs.png'), img)

        # 预测行人边界框可视化
        s = image_45[0].split('/')
        img = cv2.imread(os.path.join('/home/data/zmk/', s[1], s[2], s[3], s[4]))
        box = np.array(truth[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
        box = np.array(pred_x[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)
        box = np.array(pred_x_o[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 3)
        box = np.array(pred_x_o_f[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 3)  # 青色
        # 保存图片
        s1 = s[-1].split('.')
        cv2.imwrite(os.path.join(save_path, s1[0] + '_' + pid_45 + '_pred.png'), img)


def pie_draw(seq, dataset='pie'):
    t = Predict()

    # 读取测试数据
    data_test = load_data(dataset=dataset)
    opts = {
        'normalize_bbox': False,
        'bbox_type': 'Interframe_offset',
        'track_overlap': 0.5,
        'observe_length': 15,
        'predict_length': 45,
        'enc_input_type1': ['bbox'],
        'enc_input_type2': [],
        'dec_input_type': [],
        'prediction_type': ['bbox']
    }
    test_data = t.get_data(data_test, **opts)

    # 获取预测数据
    results_x, results_x_o, results_x_o_s, results_x_o_f, results_x_o_s_f = load_results(dataset=dataset)
    print(len(test_data['obs_image']), len(test_data['obs_image'][0]))
    #print(len(test_data['enc_input1']), len(test_data['enc_input1'][0]))
    print(results_x['results'].shape)
    #print(results_x['gt'].shape)

    for idx in range(seq[0], seq[1]):
        image_15 = test_data['obs_image'][idx][-1]  # 观测最后一帧的图像
        pid_15 = test_data['obs_pid'][idx][-1][0]  # 观测最后一帧的行人
        bbox_15 = test_data['enc_input1'][idx][-1]  # 观测最后一帧的边界框
        image_45 = test_data['pred_image'][idx][-1]  # 预测最后一帧的图像
        pid_45 = test_data['pred_pid'][idx][-1][0]  # 预测最后一帧的行人
        pred_x = results_x['results'][idx]  # 未来预测轨迹45*4（仅输入边界框）
        pred_x_o = results_x_o['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向）
        pred_x_o_s = results_x_o_s['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向+车速）
        pred_x_o_f = results_x_o_f['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向+光流）
        pred_x_o_s_f = results_x_o_s_f['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向+车速+光流）

        # 未来真实轨迹45*4
        truth = results_x['gt'][idx]
        for j in range(len(truth)):
            if j == 0:
                truth[j, :] = truth[j, :] + bbox_15
            else:
                truth[j, :] = truth[j, :] + truth[j - 1, :]

        print(image_15, image_45)
        #print(pid_15)
        #print(bbox_15)
        #print(truth)
        #print(pred_x_o)

        # 观测行人边界框可视化
        s = image_15.split('/')
        save_path = 'result_image/' + dataset + '/' + s[6] + '/' + s[7]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = cv2.imread(image_15)
        box = np.array(bbox_15)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 3)  # 白色
        # 保存图片
        s1 = s[-1].split('.')
        cv2.imwrite(os.path.join(save_path, s1[0] + '_' + pid_15 + '_obs.png'), img)

        # 预测行人边界框可视化
        s = image_45.split('/')
        img = cv2.imread(image_45)
        box = np.array(truth[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)  # 绿色
        box = np.array(pred_x[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 3)  # 黄色
        box = np.array(pred_x_o[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 3)  # 紫色
        # box = np.array(pred_x_o_s[-1])
        # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)  # 蓝色
        box = np.array(pred_x_o_f[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 3)  # 青色
        box = np.array(pred_x_o_s_f[-1])
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)  # 红色
        # 保存图片
        s1 = s[-1].split('.')
        cv2.imwrite(os.path.join(save_path, s1[0] + '_' + pid_45 + '_pred.png'), img)


def daimler_draw(seq, dataset='daimler'):
    t = Predict()

    # 读取测试数据
    data_test = load_data(dataset=dataset)
    opts = {
        'normalize_bbox': False,
        'bbox_type': 'Interframe_offset',
        'track_overlap': 0.7,
        'observe_length': 8,
        'predict_length': 8,
        'enc_input_type1': ['bbox'],
        'enc_input_type2': [],
        'dec_input_type': [],
        'prediction_type': ['bbox']
    }
    test_data = t.get_data(data_test, **opts)

    # 获取预测数据
    results_x, results_x_o, results_x_o_s, results_x_o_f, results_x_o_s_f  = load_results(dataset=dataset)
    print(len(test_data['obs_image']), len(test_data['obs_image'][0]))
    #print(len(test_data['enc_input1']), len(test_data['enc_input1'][0]))
    print(results_x['results'].shape)
    #print(results_x['gt'].shape)

    for idx in range(seq[0], seq[1]):
        image_obs = test_data['obs_image'][idx][-1][0]  # 观测最后一帧的图像
        bbox_obs = test_data['enc_input1'][idx]  # 观测边界框
        real_bb = test_data['pred_target'][idx]
        pred_x = results_x['results'][idx]  # 未来预测轨迹8*4（仅输入边界框）
        pred_x_o = results_x_o['results'][idx]  # 未来预测轨迹8*4（输入边界框+方向）
        pred_x_o_s = results_x_o_s['results'][idx]  # 未来预测轨迹8*4（输入边界框+方向+车速）
        pred_x_o_f = results_x_o_f['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向+光流）
        pred_x_o_s_f = results_x_o_s_f['results'][idx]  # 未来预测轨迹45*4（输入边界框+方向+车速+光流）

        # 未来真实轨迹8*4
        truth = results_x['gt'][idx]
        for j in range(len(truth)):
            if j == 0:
                truth[j, :] = truth[j, :] + bbox_obs[-1]
            else:
                truth[j, :] = truth[j, :] + truth[j - 1, :]

        print(image_obs)
        #print(bbox_15)
        #print(truth)
        #print(pred_x_o)

        # 读取图像
        s = image_obs.split('/')
        save_path = 'result_image/' + dataset + '/' + s[3] + '/' + s[4]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img = cv2.imread(os.path.join('/home/data/zmk/', s[1], s[2], s[3], s[4], s[5], s[6]))
        # 画观测轨迹
        bbox = np.array(bbox_obs)
        for n in range(len(bbox)-1):
            point1 = [(bbox[n, 0] + bbox[n, 2])/2, bbox[n, 3]]
            point2 = [(bbox[n+1, 0] + bbox[n+1, 2]) / 2, bbox[n+1, 3]]
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 255, 255), 3)
        # box = np.array(real_bb[-1])
        # # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
        # box = bbox[-1]
        # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        # 画预测轨迹
        bbox = np.array(truth)
        for n in range(len(bbox) - 1):
            point1 = [(bbox[n, 0] + bbox[n, 2]) / 2, bbox[n, 3]]
            point2 = [(bbox[n + 1, 0] + bbox[n + 1, 2]) / 2, bbox[n + 1, 3]]
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 3)  # 绿色

        bbox = np.array(pred_x)
        for n in range(len(bbox) - 1):
            point1 = [(bbox[n, 0] + bbox[n, 2]) / 2, bbox[n, 3]]
            point2 = [(bbox[n + 1, 0] + bbox[n + 1, 2]) / 2, bbox[n + 1, 3]]
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 255), 3)  # 黄色

        bbox = np.array(pred_x_o)
        for n in range(len(bbox) - 1):
            point1 = [(bbox[n, 0] + bbox[n, 2]) / 2, bbox[n, 3]]
            point2 = [(bbox[n + 1, 0] + bbox[n + 1, 2]) / 2, bbox[n + 1, 3]]
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 255), 3)  # 紫色

        bbox = np.array(pred_x_o_s)
        for n in range(len(bbox) - 1):
            point1 = [(bbox[n, 0] + bbox[n, 2]) / 2, bbox[n, 3]]
            point2 = [(bbox[n + 1, 0] + bbox[n + 1, 2]) / 2, bbox[n + 1, 3]]
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 0), 3)  # 蓝色

        bbox = np.array(pred_x_o_f)
        for n in range(len(bbox) - 1):
            point1 = [(bbox[n, 0] + bbox[n, 2]) / 2, bbox[n, 3]]
            point2 = [(bbox[n + 1, 0] + bbox[n + 1, 2]) / 2, bbox[n + 1, 3]]
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 255, 0), 3)

        bbox = np.array(pred_x_o_s_f)
        for n in range(len(bbox) - 1):
            point1 = [(bbox[n, 0] + bbox[n, 2]) / 2, bbox[n, 3]]
            point2 = [(bbox[n + 1, 0] + bbox[n + 1, 2]) / 2, bbox[n + 1, 3]]
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 0, 255), 3)

        # 保存图片
        s1 = s[-1].split('_')
        cv2.imwrite(os.path.join(save_path, s1[1] + '.png'), img)


if __name__ == '__main__':
    dataset = 'jaad'
    seq = [15000, 20000]
    if dataset == 'jaad':
        jaad_draw(seq=seq, dataset='jaad')  # idx[0, 49590]
    elif dataset == 'pie':
        pie_draw(seq=seq, dataset='pie')  # idx[0, 36208]
    elif dataset == 'daimler':
        daimler_draw(seq=seq, dataset='daimler')  # idx[0, 2412]
    elif dataset == 'carla':
        carla_draw(seq=seq, dataset='carla')  # idx[0, 2412]