import os
import pickle
import numpy as np

from jaad_data import JAAD


def save_dict(obj, data_type):
    root = 'data/jaad/data_file/'
    if not os.path.exists(root):
        os.makedirs(root)
    with open(root + data_type + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def data_read(data_type, min_track_size=61):

    jaad_path = '../JAAD_dataset/'

    data = np.genfromtxt(jaad_path + 'split_ids/high_visibility/' + data_type + '.txt', dtype=str, delimiter='\n')

    imdb = JAAD(data_path=jaad_path)
    database = imdb.generate_database()  # 读取标记数据

    image_seq, pids_seq, box_seq, resolution_seq, occ_seq = [], [], [], [], []
    for name in database.keys():

        if name in data:
            video_data = database[name]
            frames = video_data['num_frames']
            res = [video_data['width'], video_data['height']]

            peds = video_data['ped_annotations']
            for ped_id in peds:
                if 'p' in ped_id:
                    continue
                ped_frames = peds[ped_id]['frames']
                ped_occ = peds[ped_id]['occlusion']
                ped_bbox = peds[ped_id]['bbox']

                height_mean = np.mean(np.array(ped_bbox)[:, 3] - np.array(ped_bbox)[:, 1])

                img_path = [[os.path.join(jaad_path, 'images', name, "{:05d}.png".format(num))] for num in ped_frames]
                ped_ids = [[ped_id]] * len(ped_bbox)
                resolution = [res] * len(ped_bbox)

                if len(ped_bbox) < min_track_size:
                    continue

                image_seq.append(img_path)
                pids_seq.append(ped_ids)
                box_seq.append(ped_bbox)
                resolution_seq.append(resolution)
                ped_occ = [[occ] for occ in ped_occ]
                occ_seq.append(ped_occ)

    print(len(box_seq))
    data = {}
    data['image'] = image_seq
    data['pid'] = pids_seq
    data['bbox'] = box_seq
    data['resolution'] = resolution_seq
    data['occlusion'] = occ_seq

    save_dict(data, data_type)


if __name__ == '__main__':
    data_read('train')
    data_read('val')
    data_read('test')
