import os
import pickle


def save_dict(obj, data_type):
    root = 'data/daimler/data_file/'
    if not os.path.exists(root):
        os.makedirs(root)
    with open(root + data_type + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def data_read(gt_root, vehicle_root, image_root):
    '''
    读取视频序列中的数据（即一个行人的序列数据）
    Args:
        name: 序列名称
        gt_root: 标记文件的根路径
        vehicle_root: 车速标记文件的根路径
        image_root: 数据集图像的根路径

    Returns:
        image_seq: 序列图像路径列表
        pids_seq: 序列的名称及类别列表
        box_seq: 序列的边界框列表
        speed_seq: 序列的车速列表
        label_seq: 序列的类别标签

    '''

    seq_name = os.listdir(gt_root)

    image_seq, pids_seq, box_seq, speed_seq, label_seq, resolution_seq, tte_seq = [], [], [], [], [], [], []
    for name in seq_name:

        images, pids, bbox, v_speed, labels, resolution, f_tte = [], [], [], [], [], [], []
        with open(os.path.join(gt_root, name, 'LabelData/gt.db')) as f:
            lines = f.readlines()
            #print(len(lines))
            for i, line in enumerate(lines):
                line = line.strip()
                if line == ":":    #读取该序列的名称、类别和序列长度
                    id = lines[i+1].strip()
                    print(id)
                    s = id.split('_')
                    label_n = s[-1]
                    #print(label_n)
                    seq_length = lines[i+3].strip()
                    #print(seq_length)

                    if label_n == 'BendingIn':
                        label = 0
                    elif label_n == 'Crossing':
                        label = 1
                    elif label_n == 'Starting':
                        label = 2
                    elif label_n == 'Stopping':
                        label = 3

                if line == ";":    #读取每帧的数据
                    img = lines[i + 1].strip()
                    s = img.split('_')
                    img_path = os.path.join(image_root, label_n, name, 'RectGrabber', 'imgrect_' + s[1] + '_' + s[2])
                    #print(img_path)

                    res = lines[i + 2].strip()
                    res = res.split(' ')

                    if i + 7 < len(lines):
                        box = lines[i + 7].strip()
                    else:
                        box = '0'
                    box = box.split(' ')
                    #print(box)

                    with open(os.path.join(vehicle_root, name, 'TxtVehicle/vehicle_' + s[1] + '.txt')) as v_f:
                        v_lines = v_f.readlines()
                        speed = v_lines[1].split('\t')
                        speed = float(speed[0])
                        #print(speed)
                    v_f.close()

                    if len(box) == 4:

                        if i + 12 < len(lines):
                            TTE = lines[i + 12].strip()
                            TTE = TTE.split(' ')
                            if TTE[0] == '%':
                                #print(TTE)
                                tte = int(TTE[2])
                            else:
                                tte = 1000

                        pids.append([f'{id}'])
                        images.append([f'{img_path}'])
                        bbox.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                        #print([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                        v_speed.append([speed])
                        labels.append([label])
                        resolution.append([int(res[0]), int(res[1])])
                        f_tte.append([tte])
                    else:
                        if len(images) > 16:
                            image_seq.append(images)
                            pids_seq.append(pids)
                            box_seq.append(bbox)
                            speed_seq.append(v_speed)
                            label_seq.append(labels)
                            resolution_seq.append(resolution)
                            tte_seq.append(f_tte)
                            #print(tte_seq)
                            #print(bbox)
                        #else:
                        #    print(img, bbox)
                        images, pids, bbox, v_speed, labels, resolution, f_tte = [], [], [], [], [], [], []

        #print(images)
        #print(pids)
        #print(bbox)
        #print(v_speed)
        #print(labels)
        #print(resolution)
        f.close()

        if len(images) > 16:
            image_seq.append(images)
            pids_seq.append(pids)
            box_seq.append(bbox)
            speed_seq.append(v_speed)
            label_seq.append(labels)
            resolution_seq.append(resolution)
            tte_seq.append(f_tte)
            #print(tte_seq)
            #print(bbox)
        #else:
        #    print(bbox)

    return image_seq, pids_seq, box_seq, speed_seq, label_seq, resolution_seq, tte_seq


def main(type):
    root = "/home/data/zmk/Daimler/"
    image_root = root + type + "Data_Image/"
    vehicle_root = root + type + "Data_Vehicle/"
    gt_root = root + type + "Data_Annotations/"

    data = {}
    image_seq, pids_seq, box_seq, speed_seq, label_seq, resolution_seq, tte_seq = data_read(gt_root, vehicle_root, image_root)

    data['image'] = image_seq
    data['pid'] = pids_seq
    data['bbox'] = box_seq
    data['obd_speed'] = speed_seq
    data['label'] = label_seq
    data['resolution'] = resolution_seq
    data['TTE'] = tte_seq

    #print(data['image'])
    #print(data['bbox'])
    #print(data['resolution'])
    print(data['TTE'])

    return data


if __name__ == '__main__':
    data_type = ["Train", "TestV"]

    for type in data_type:
        print('---------------------------------------------------------')
        print("Generating " + type + " data")
        data = main(type)
        save_dict(data, type)
