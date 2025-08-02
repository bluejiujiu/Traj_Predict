import os
import pickle
import numpy as np

def flow_data_process(type):
    me_data_ego = np.load(f'data/carla/flow_me/flow_carla_{type}_ego.npy', allow_pickle=True)
    me_data_ped = np.load(f'data/carla/flow_me/flow_carla_{type}_ped.npy', allow_pickle=True)
    for i in range(len(me_data_ped)):
        for j in range(len(me_data_ped[i])):
            me_data_ped[i][j] = me_data_ped[i][j].astype(np.float32)
    for i in range(len(me_data_ego)):
        for j in range(len(me_data_ego[i])):
            me_data_ego[i][j] = me_data_ego[i][j].astype(np.float32)
    np.save(f'data/carla/flow_me/flow_carla_{type}_ego.npy',me_data_ego)
    np.save(f'data/carla/flow_me/flow_carla_{type}_ped.npy', me_data_ped)

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
    data_speed = []
    data_degree_pred = []
    for i in range(len_pid):
        data_image2 = [data['image'][i][0]]
        data_speed2 = [data['obd_speed'][i][0]]
        data_pid2 = [data['pid'][i][0]]
        data_bbox2 = [data['bbox'][i][0]]
        for j in range(len(data_1s[i])-1):
            if data_1s[i][j+1] == data_1s[i][j] + 1:
                data_image2.append(data['image'][i][j+1])
                data_speed2.append(data['obd_speed'][i][j + 1])
                data_pid2.append(data['pid'][i][j+1])
                data_bbox2.append(data['bbox'][i][j + 1])
            else:
                data_image.append(data_image2)
                data_speed.append(data_speed2)
                data_pid.append(data_pid2)
                data_bbox.append(data_bbox2)
                data_image2 = [data['image'][i][j+1]]
                data_speed2 = [data['obd_speed'][i][j + 1]]
                data_pid2 = [data['pid'][i][j + 1]]
                data_bbox2 = [data['bbox'][i][j + 1]]
            if j == len(data_1s[i])-2:
                data_image.append(data_image2)
                data_speed.append(data_speed2)
                data_pid.append(data_pid2)
                data_bbox.append(data_bbox2)
    new_data = {'image':None, 'pid':None, 'bbox':None, 'obd_speed':None}
    new_data['image'] = data_image
    new_data['obd_speed'] = data_speed
    new_data['pid'] = data_pid
    new_data['bbox'] = data_bbox

    pid_num = len(new_data['pid'])
    del_id = []
    for i in range(pid_num):
        if len(new_data['pid'][i]) < min_length:
            del_id.append(i)
        for j in range(len(new_data['bbox'][i])):
            new_data['bbox'][i][j] = new_data['bbox'][i][j]

    del_id.sort(reverse=True)
    for i in del_id:
        del new_data['pid'][i]
        del new_data['image'][i]
        del new_data['bbox'][i]
        del new_data['obd_speed'][i]
    return new_data

def save_dict(obj, data_type):
    root = 'data/carla/data_file/'
    if not os.path.exists(root):
        os.makedirs(root)
    with open(root + data_type + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def carla_data_pack_process(path, data_type):
    data_id_file = '/home/data/ly/carla/PythonAPI/examples/YHX/ly/carla_study/split_ids/' + data_type + '.txt'
    with open(data_id_file, 'r', encoding='utf-8') as file:
        files_raw = file.read().split()
    # files_raw = os.listdir(path)
    files = sorted(files_raw, key=lambda x: int(x[3:]))
    data_all = {key: None for key in ['image', 'pid', 'bbox', 'obd_speed']}
    data_img, data_pid, data_bbox, data_speed = list(), list(), list(), list()
    for fi_name in files:
        path_now = path + '/' + fi_name + '/data'
        files_now = os.listdir(path_now)
        # 所有文件和子文件夹
        data = {key: None for key in ['image', 'pid', 'bbox', 'obd_speed']}
        image_list, pid_list, bbox_list, speed_list = list(), list(), list(), list()
        for file in files_now:
            data_file = np.load(path_now + '/' + file, allow_pickle=True).item()
            new_pid = [[fi_name[3:] + '_' + str(x)] for x in data_file['pid']]
            new_bbox = [x[0] for x in data_file['bbox']]
            new_image = [data_file['image'] for _ in range(len(data_file['pid']))]
            new_speed = [data_file['obd_speed'] for _ in range(len(data_file['pid']))]
            image_list.extend(new_image)
            speed_list.extend(new_speed)
            pid_list.extend(new_pid)
            bbox_list.extend(new_bbox)
        sorted_list = sorted(enumerate(pid_list),key=lambda x: x[1])
        sorted_indices, pid_sort = zip(*sorted_list)
        image_sort = [image_list[i] for i in sorted_indices]
        speed_sort = [speed_list[i] for i in sorted_indices]
        bbox_sort = [bbox_list[i] for i in sorted_indices]
        image_list, pid_list, bbox_list, speed_list = list(), list(), list(), list()
        pid_list_now, image_list_now, bbox_list_now, speed_list_now = list(), list(), list(), list()
        pid_list_now.append(pid_sort[0])
        image_list_now.append(image_sort[0])
        speed_list_now.append(speed_sort[0])
        bbox_list_now.append(bbox_sort[0])
        for i in range(1, len(pid_sort)):
            if pid_sort[i] == pid_sort[i-1]:
                pid_list_now.append(pid_sort[i])
                image_list_now.append(image_sort[i])
                speed_list_now.append(speed_sort[i])
                bbox_list_now.append(bbox_sort[i])
            else:
                if len(image_list_now) == 0:
                    continue
                sorted_list = sorted(enumerate(image_list_now), key=lambda x: x[1])
                sorted_indices, image_sort_now = zip(*sorted_list)
                pid_sort_now = [pid_list_now[j] for j in sorted_indices]
                bbox_sort_now = [bbox_list_now[j] for j in sorted_indices]
                speed_sort_now = [speed_list_now[j] for j in sorted_indices]
                pid_list.append(pid_sort_now)
                image_list.append(image_sort_now)
                speed_list.append(speed_sort_now)
                bbox_list.append(bbox_sort_now)
                pid_list_now, image_list_now, bbox_list_now, speed_list_now = list(), list(), list(), list()
        data['image'], data['bbox'], data['pid'], data['obd_speed'] = image_list, bbox_list, pid_list, speed_list
        data_img.extend(image_list)
        data_speed.extend(speed_list)
        data_pid.extend(pid_list)
        data_bbox.extend((bbox_list))
        data_all['image'], data_all['bbox'], data_all['pid'], data_all['obd_speed'] = data_img, data_bbox, data_pid, data_speed
    return data_all

def main(type):
    path = '/home/data/ly/carla/PythonAPI/examples/YHX/ly/carla_study/output'
    data = carla_data_pack_process(path, type)
    data = carla_data_process(data)
    return data


if __name__ == '__main__':
    data_type = ["train", "val", "test"]
    # data_type = ['train']
    for type in data_type:
        print('---------------------------------------------------------')
        print("Generating " + type + " data")
        # data = main(type)
        # save_dict(data, type)
        flow_data_process(type)
