import numpy as np
import pickle
import math
def carla_spped_process(data):
    len_speed = len(data['obd_speed'])
    for i in range(len_speed):
        for j in range(len(data['obd_speed'][i])):
            data['obd_speed'][i][j] = [data['obd_speed'][i][j]]
    return data
def carla_data_process(data):
    # 删除不连续的图片
    min_length = 60
    len_pid = len(data['pid'])
    data_1s = []
    for i in range(len_pid):
        data_2s = []
        for j in range(len(data['pid'][i])):
            data_2s.append(int(data['image'][i][j].split('/')[-1].split('.')[0].split('_')[-1]))
        data_1s.append(data_2s)

    data_image = []
    data_pid = []
    data_bbox = []
    data_degree_pred = []
    for i in range(len_pid):
        data_image2 = [data['image'][i][0]]
        data_pid2 = [data['pid'][i][0]]
        data_bbox2 = [data['bbox'][i][0]]
        data_degree_pred2 = [data['degree_pred'][i][0]]
        for j in range(len(data_1s[i])-1):
            if data_1s[i][j+1] == data_1s[i][j] + 1:
                data_image2.append(data['image'][i][j+1])
                data_pid2.append(data['pid'][i][j+1])
                data_bbox2.append(data['bbox'][i][j + 1])
                data_degree_pred2.append(data['degree_pred'][i][j + 1])
            else:
                data_image.append(data_image2)
                data_pid.append(data_pid2)
                data_bbox.append(data_bbox2)
                data_degree_pred.append(data_degree_pred2)
                data_image2 = [data['image'][i][j+1]]
                data_pid2 = [data['pid'][i][j + 1]]
                data_bbox2 = [data['bbox'][i][j + 1]]
                data_degree_pred2 = [data['degree_pred'][i][j + 1]]
            if j == len(data_1s[i])-2:
                data_image.append(data_image2)
                data_pid.append(data_pid2)
                data_bbox.append(data_bbox2)
                data_degree_pred.append(data_degree_pred2)
    new_data = {'image':None, 'pid':None, 'bbox':None, 'degree_pred':None}
    new_data['image'] = data_image
    new_data['pid'] = data_pid
    new_data['bbox'] = data_bbox
    new_data['degree_pred'] = data_degree_pred

    pid_num = len(new_data['pid'])
    del_id = []
    for i in range(pid_num):
        if len(new_data['pid'][i]) < min_length:
            del_id.append(i)
        for j in range(len(new_data['bbox'][i])):
            new_data['bbox'][i][j] = new_data['bbox'][i][j][0].tolist()

    del_id.sort(reverse=True)
    for i in del_id:
        del new_data['pid'][i]
        del new_data['image'][i]
        del new_data['bbox'][i]
        del new_data['degree_pred'][i]

    return new_data

def load_dict(type='', dataset=''):
    root = './data/' + dataset + '/data_add/'
    with open(root + type + '.pkl', 'rb') as f:
        return pickle.load(f)

def classification(classify_data):
    length = len(classify_data['degree_pred'])
    for i in range(length):
        for j in range(len(classify_data['degree_pred'][i])):
            for su in range(2):
                if 22.5 <= classify_data['degree_pred'][i][j][su] < 67.5:
                    classify_data['degree_pred'][i][j][su] = 1.0
                elif 67.5 <= classify_data['degree_pred'][i][j][su] < 112.5:
                    classify_data['degree_pred'][i][j][su] = 2.0
                elif 112.5 <= classify_data['degree_pred'][i][j][su] < 157.5:
                    classify_data['degree_pred'][i][j][su] = 3.0
                elif 157.5 <= classify_data['degree_pred'][i][j][su] < 202.5:
                    classify_data['degree_pred'][i][j][su] = 4.0
                elif 202.5 <= classify_data['degree_pred'][i][j][su] < 247.5:
                    classify_data['degree_pred'][i][j][su] = 5.0
                elif 247.5 <= classify_data['degree_pred'][i][j][su] < 292.5:
                    classify_data['degree_pred'][i][j][su] = 6.0
                elif 292.5 <= classify_data['degree_pred'][i][j][su] < 337.5:
                    classify_data['degree_pred'][i][j][su] = 7.0
                else:
                    classify_data['degree_pred'][i][j][su] = 8.0

    return classify_data

def word_to_img(data=''):
    # 导入索引矩阵
    # new_degree_data = np.loadtxt('./data/pie/corner.txt')
    # new_degree_data = np.round(new_degree_data, decimals=2)
    new_degree_data = {}
    # with open('../data/pie/corner.txt', 'r') as file:
    with open('/home/data/ly/Traj_Predict/data/pie/corner.txt', 'r') as file:
        for line in file:
            # 假设文档中的数字是用空格分隔的
            key, value = line.strip().split()
            new_degree_data[int(key)] = float(value)
    length = len(data['degree_pred'])
    for i in range(length):
        for j in range(len(data['degree_pred'][i])):
            for su in range(2):
                degree = data['degree_pred'][i][j][su]
                new_degree = new_degree_data[math.ceil(degree)-1] + (degree - math.ceil(degree) + 1) * (new_degree_data[math.ceil(degree)] - new_degree_data[math.ceil(degree)-1])
                data['degree_pred'][i][j][su] = new_degree
    return data


def fen_main(type='', dataset=''):
    data = load_dict(type,dataset)
    data_classify = classification(data)
    print(1)

if __name__ == '__main__':
    word_to_img([1,1])
    dataset = 'jaad' #jaad,pie
    train_test = ['val', 'train', 'test']
    for type in train_test:
        fen_main(type,dataset)