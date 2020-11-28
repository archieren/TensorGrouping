# -*- coding: utf-8 -*-
import os
import tensorflow as tf


from tensorgroup.models.dataset.coco import coco_kp as KP
from tensorgroup.models.dataset import mode_keys as ModeKey

DATASETNAME = 'three_point'

def build_Data_From_JSONs():  # 有必要注释一下目录结构 和约定目录结构 TODO
    root = os.path.join(os.getcwd(), 'data_coco_kp', DATASETNAME)
    source_dir = os.path.join(root, 'data', ModeKey.TRAIN)
    output_dir = os.path.join(root, 'tf_records')
    if not os.path.exists(output_dir):   #
        os.makedirs(output_dir)
    KP.produce_dataset_from_jsons(dataset_name=ModeKey.TRAIN, json_source_dir=source_dir, target_directory=output_dir)
    pass


build_Data_From_JSONs()
