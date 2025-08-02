import os
import cv2
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def load_dict(type):
    root = 'data/daimler/data_add/'
    with open(root + type + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_dict(obj, save_path):
    with open(save_path + 'Train_aug.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def orientation_plot(body, head):
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(2, 2, 1)

    # 画圆
    # 点的横坐标为a
    a = np.arange(-2 * np.pi, 2 * np.pi, 0.00001)
    # 点的纵坐标为b
    b = np.sqrt(np.power(2, 2) - np.power((a + 3), 2))
    ax.plot(a, b, color='r', linestyle='-')
    ax.plot(a, -b, color='r', linestyle='-')
    ax.scatter(-3, 0, c='b', marker='o')

    #画身体方向箭头
    dx = 2 * math.cos(math.radians(body))
    dy = 2 * math.sin(math.radians(body))
    ax.quiver(-3, 0, dx, dy, angles='xy', scale=1, scale_units='xy')

    # 画圆
    # 点的横坐标为a
    a = np.arange(-2 * np.pi, 2 * np.pi, 0.00001)
    # 点的纵坐标为b
    b = np.sqrt(np.power(2, 2) - np.power((a - 3), 2))
    ax.plot(a, b, color='r', linestyle='-')
    ax.plot(a, -b, color='r', linestyle='-')
    ax.scatter(3, 0, c='b', marker='o')

    #画头部方向箭头
    dx = 2 * math.cos(math.radians(head))
    dy = 2 * math.sin(math.radians(head))
    ax.quiver(3, 0, dx, dy, angles='xy', scale=1, scale_units='xy')

    ax.axis([-6, 6, -4, 4])
    ax.set_title("body-head orientation")
    ax.set_xticks([])  # 坐标轴不可见
    ax.set_yticks([])
    ax.text(-4.5, 3, f"body:{'{:.2f}'.format(body)}°", fontsize=10)
    ax.text(1.5, 3, f"head:{'{:.2f}'.format(head)}°", fontsize=10)

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()

    return np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)


def draw(img, box, degree):

    #真实行人边界框可视化
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    #身体和头部角度
    body_degree = degree[0]
    head_degree = degree[1]
    #print('{:.2f}'.format(body_degree), '{:.2f}'.format(head_degree))

    # 预测方向可视化
    orientation_img = orientation_plot(body_degree, head_degree)
    img[0:round(orientation_img.shape[0]/2), 0:round(orientation_img.shape[1]/2)] = orientation_img[0:round(orientation_img.shape[0]/2), 0:round(orientation_img.shape[1]/2)]

    cv2.imshow('mirror', img)
    cv2.waitKey(200)

    plt.clf()
    plt.close()


def data_augment():
    data_train = load_dict('Train')
    m_images = data_train['image'].copy()
    m_pids = data_train['pid'].copy()
    m_bbox = data_train['bbox'].copy()
    m_speed = data_train['obd_speed'].copy()
    m_degree = data_train['degree_pred'].copy()
    m_res = data_train['resolution'].copy()
    m_label = data_train['label'].copy()
    m_tte = data_train['TTE'].copy()
    data = {}

    for img_path, pid, bbox, degree, resolution, obd_speed, \
        labels, TTE in zip(data_train['image'], data_train['pid'], data_train['bbox'], data_train['degree_pred'],
                           data_train['resolution'], data_train['obd_speed'], data_train['label'], data_train['TTE']):

        m_img, m_id, m_box, m_deg = [], [], [], []
        for path, id, box, deg, res in zip(img_path, pid, bbox, degree, resolution):
            s = path[0].split('\\')
            name = str(s[-1])
            name = name.split('.')
            name = str(name[0])

            s = id[0].split('_')
            action = str(s[-1])
            seq_ = s[:-1]
            #print(seq_)
            #print(len(seq_))
            if len(seq_) == 2:
                seq = seq_[0] + '_' + seq_[1]
            elif len(seq_) == 3:
                seq = seq_[0] + '_' + seq_[1] + '_' + seq_[2]

            #print(action, seq, name)

            #原始图像
            cap = cv2.imread(path[0])
            img = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

            # 图像镜像翻转
            img_mirror = cv2.flip(img, 1)
            save_path = os.path.join('..\\..\\datasets\\Daimler\\TrainData_Image\\', action, 'mirror' + '_' + seq)
            #if not os.path.exists(save_path):
            #    os.makedirs(save_path)

            #保存翻转后的图像
            m_img_path = save_path + '\\' + name + '.jpg'
            #print(m_img_path)
            #cv2.imwrite(m_img, img_mirror)
            m_img.append([f'{m_img_path}'])

            m_id.append(['mirror' + '_' + id[0]])
            print(['mirror' + '_' + id[0]])

            #边界框翻转
            b = [res[0] - box[2], box[1], res[0] - box[0], box[3]]
            m_box.append(b)
            #print(box)
            #print(b)

            #方向翻转
            if deg[0] > 0 and deg[0] < 180:
                f0 = 180 - deg[0]
            elif deg[0] > 180 and deg[0] < 360:
                f0 = 360 - deg[0] + 180

            if deg[1] > 0 and deg[1] < 180:
                f1 = 180 - deg[1]
            elif deg[1] > 180 and deg[1] < 360:
                f1 = 360 - deg[1] + 180
            #print([f0, f1])
            m_deg.append([f0, f1])

            #镜像前后可视化
            #draw(img, box, deg)
            #draw(img_mirror, b, [f0, f1])

        m_images.append(m_img)
        m_pids.append(m_id)
        m_bbox.append(m_box)
        m_speed.append(obd_speed)
        m_degree.append(m_deg)
        m_res.append(resolution)
        m_label.append(labels)
        m_tte.append(TTE)

    data['image'] = m_images
    data['pid'] = m_pids
    data['bbox'] = m_bbox
    data['obd_speed'] = m_speed
    data['degree_pred'] = m_degree
    data['resolution'] = m_res
    data['label'] = m_label
    data['TTE'] = m_tte

    return data


if __name__ == '__main__':

    save_path = 'F:\\NWPU\\PIEPredict\\data\\daimler\\data_add\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = data_augment()
    save_dict(data, save_path)