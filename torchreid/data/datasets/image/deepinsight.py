from __future__ import division, print_function, absolute_import
import glob
import numpy as np
import os.path as osp
import zipfile

from ....utils import read_json, write_json

from ..dataset import ImageDataset


class DeepInsight(ImageDataset):
    """DeepInsight.

    Reference:
        CloudMile x DeepInsight
    
    Dataset statistics:
        - identities: 971.
        - images: 3884.
        - cameras: 4.
    """
    dataset_dir = 'deepinsight'
    dataset_url = None

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.check_before_run([self.dataset_dir])

        # Get data partition
        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError('split_id exceeds range, received {}, but expected between 0 and {}'
                             .format(split_id, len(splits) - 1))
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        super(DeepInsight, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        if not osp.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')
            img_paths = sorted(glob.glob(osp.join(self.dataset_dir, '*.jpg')))
            img_list = []
            pid_container = set()
            camid_container = set()
            for img_path in img_paths:
                img_name = osp.basename(img_path)
                pid, camid, datetime, sn = img_name.split('-')
                camid = int(camid)
                if '_' in pid:
                    pid, vid = pid.split('_')
                    pid = int(pid) - 1
                    img_list.append((img_path, pid, vid, camid))
                else:
                    pid = int(pid) - 1
                    img_list.append((img_path, pid, camid))
                pid_container.add(pid)
                camid_container.add(camid)

            num_pids = len(pid_container)
            num_train_pids, num_train_camids = num_pids // 2, len(camid_container) // 2

            splits = []
            for _ in range(10):
                # Shuffle the pids
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = np.sort(order[:num_train_pids])
                idx2label = {idx: label for label, idx in enumerate(train_idxs)}

                # Shuffle the camids
                order = np.array(list(camid_container))
                np.random.shuffle(order)
                cam_a_ids = np.sort(order[:num_train_camids])

                train, test_a, test_b = [], [], []
                for attr in img_list:  # (img_path, pid, (vid), camid)
                    if attr[1] in train_idxs:
                        attr = list(attr)
                        attr[1] = idx2label[attr[1]]
                        train.append(tuple(attr))
                    else:
                        if attr[-1] in cam_a_ids:
                            test_a.append(attr)
                        else:
                            test_b.append(attr)

                # use cameraA as query and cameraB as gallery
                split = {
                    'train': train,
                    'query': test_a,
                    'gallery': test_b,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {
                    'train': train,
                    'query': test_b,
                    'gallery': test_a,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))
