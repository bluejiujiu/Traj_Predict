import os
import pickle
import numpy as np

def save_dict(obj, save_path, name):
    with open(save_path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def sampling(data):
    image, pid, bbox, speed, label, res, tte, deg = [], [], [], [], [], [], [], []

    for num in range(len((data['bbox']))):
        image1, pid1, bbox1, speed1, label1, res1, tte1, deg1 = [], [], [], [], [], [], [], []
        image2, pid2, bbox2, speed2, label2, res2, tte2, deg2 = [], [], [], [], [], [], [], []

        for length in range(len((data['bbox'][num]))):

            if length % 2 == 0:
                image1.append(data['image'][num][length][:])
                pid1.append(data['pid'][num][length][:])
                bbox1.append(data['bbox'][num][length][:])
                speed1.append(data['obd_speed'][num][length][:])
                label1.append(data['label'][num][length][:])
                res1.append(data['resolution'][num][length][:])
                tte1.append(data['TTE'][num][length][:])
                deg1.append(data['degree_pred'][num][length][:])

            else:
                image2.append(data['image'][num][length][:])
                pid2.append(data['pid'][num][length][:])
                bbox2.append(data['bbox'][num][length][:])
                speed2.append(data['obd_speed'][num][length][:])
                label2.append(data['label'][num][length][:])
                res2.append(data['resolution'][num][length][:])
                tte2.append(data['TTE'][num][length][:])
                deg2.append(data['degree_pred'][num][length][:])

        #print(bbox1)
        #print(bbox2)
        image.append(image1), image.append(image2)
        pid.append(pid1), pid.append(pid2)
        bbox.append(bbox1), bbox.append(bbox2)
        speed.append(speed1), speed.append(speed2)
        label.append(label1), label.append(label2)
        res.append(res1), res.append(res2)
        tte.append(tte1), tte.append(tte2)
        deg.append(deg1), deg.append(deg2)

    data_ = {}
    data_['image'] = image
    data_['pid'] = pid
    data_['bbox'] = bbox
    data_['obd_speed'] = speed
    data_['label'] = label
    data_['resolution'] = res
    data_['TTE'] = tte
    data_['degree_pred'] = deg

    #print(len(data['bbox']))
    #print(len(data_['bbox']))
    #print(len(data['bbox'][0]))
    #print(len(data_['bbox'][0]))

    return data_


if __name__ == '__main__':
    data_type = ["Train_aug", "TestV"]

    for type in data_type:
        data = load_dict('data/daimler/data_add/', type)

        out_data = sampling(data)

        save_path = 'data/daimler/data_sampling/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dict(out_data, save_path, type)