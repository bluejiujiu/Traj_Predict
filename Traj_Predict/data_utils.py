import pickle
from pie_predict import PIEPredict
from prettytable import PrettyTable


def save_dict(obj, save_path, name):
    with open(save_path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def data_merge():
    # 融合两个文件的数据

    data_type = ["Train", "TestV"]

    for type in data_type:
        data1 = load_dict('data/daimler/data_file/', type)
        data2 = load_dict('data/daimler/data_add/', type + '_data')

        data = {}
        data['image'] = data2['image']
        data['pid'] = data2['pid']
        data['bbox'] = data2['bbox']
        data['obd_speed'] = data2['obd_speed']
        data['label'] = data2['label']
        data['resolution'] = data1['resolution']
        data['TTE'] = data1['TTE']

        data['skeleton_2d_pred'] = data2['skeleton_2d_pred']
        data['skeleton_3d_pred'] = data2['skeleton_3d_pred']
        data['orientation_pred'] = data2['orientation_pred']
        data['degree_pred'] = data2['degree_pred']

        save_dict(data, 'data/daimler/data_add/', type)


def class_data():
    # 按类别划分数据

    data_type = ["Train_aug", "TestV"]

    for type in data_type:
        # root = 'data_daimler/data_add/'
        root = 'data/daimler/data_sampling/'
        data = load_dict(root, type)
        # print(data.keys())

        L = {'BendingIn': 0, 'Crossing': 1, 'Starting': 2, 'Stopping': 3}
        for key, idx in L.items():
            print(key, idx)

            image, pid, bbox, speed, label, res, tte, deg = [], [], [], [], [], [], [], []

            for n in range(len(data['label'])):

                class_label = data['label'][n]
                #print(class_label[0][0])

                if class_label[0][0] == idx:
                    image.append(data['image'][n])
                    pid.append(data['pid'][n])
                    bbox.append(data['bbox'][n])
                    speed.append(data['obd_speed'][n])
                    label.append(data['label'][n])
                    res.append(data['resolution'][n])
                    tte.append(data['TTE'][n])
                    deg.append(data['degree_pred'][n])

            data_ = {}
            data_['image'] = image
            data_['pid'] = pid
            data_['bbox'] = bbox
            data_['obd_speed'] = speed
            data_['label'] = label
            data_['resolution'] = res
            data_['TTE'] = tte
            data_['degree_pred'] = deg

            save_dict(data_, root, type + '_' + key)


def class_num(data):
    # 统计各个类别的序列数目

    num_bending, num_crossing, num_starting, num_stopping = 0, 0, 0, 0
    for image, pid in zip(data['obs_image'], data['obs_pid']):
        s = pid[0][0]
        s = s.split('_')
        label_n = s[-1]
        #print(label_n)
        if label_n == 'BendingIn':
            num_bending += 1
        elif label_n == 'Crossing':
            num_crossing += 1
        elif label_n == 'Starting':
            num_starting += 1
        elif label_n == 'Stopping':
            num_stopping += 1
        else:
            print(pid[0][0], 'error')

    return num_bending, num_crossing, num_starting, num_stopping


def seq_length():
    # 统计序列长度并输出

    # root = 'data/daimler/data_add/'
    root = 'data/daimler/data_sampling/'

    t = PIEPredict()

    model_opts = {'normalize_bbox': True,
                  'track_overlap': 0.7,
                  'observe_length': 8,
                  'predict_length': 8,
                  'enc_input_type': ['bbox'],
                  'dec_input_type': [],
                  'prediction_type': ['bbox']
                  }

    data_train = load_dict(root, 'Train_aug')
    data_test = load_dict(root, 'TestV')
    data_train = load_dict(root, 'Train_aug_BendingIn')
    data_test = load_dict(root, 'TestV_BendingIn')

    train_data = t.get_data(data_train, **model_opts)
    test_data = t.get_data(data_test, **model_opts)

    train_bending, train_crossing, train_starting, train_stopping = class_num(train_data)
    test_bending, test_crossing, test_starting, test_stopping = class_num(test_data)

    t = PrettyTable(['BendingIn', 'Crossing', 'Starting', 'Stopping', 'Total'])
    t.title = 'Train Data'
    t.add_row([train_bending, train_crossing, train_starting, train_stopping, train_data['enc_input'].shape[0]])
    print(t)

    t = PrettyTable(['BendingIn', 'Crossing', 'Starting', 'Stopping', 'Total'])
    t.title = 'Test Data'
    t.add_row([test_bending, test_crossing, test_starting, test_stopping, test_data['enc_input'].shape[0]])
    print(t)

    print(train_data['enc_input'].shape, test_data['enc_input'].shape)


data_merge()    #融合两个文件中的数据

# class_data()    # 按类别划分数据

# seq_length()    #统计序列长度
