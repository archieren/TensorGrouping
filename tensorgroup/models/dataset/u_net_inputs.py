import tensorflow as tf
import numpy as np

class DefineInputs:
    """
    """
    def __init__(self, ):
        pass

def img_to_example(image_name, img_dir, mask_ann_dir):
    
def dataset2tfrecord(img_dir, mask_ann_dir, output_dir, name, total_shards=2):
    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)
    tf.io.gfile.mkdir(output_dir)
    outputfiles = []
    img_list = tf.io.gfile.glob(os.path.join(img_dir, '*.jpg'))
    num_per_shard = int(math.ceil(len(img_list)) / float(total_shards))
    for shard_id in range(total_shards):
        outputname = '%s_%05d-of-%05d.tfrecords' % (name, shard_id+1, total_shards)
        outputname = os.path.join(output_dir, outputname)
        outputfiles.append(outputname)
        with tf.io.TFRecordWriter(outputname) as tf_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(xmllist))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d/%d' % (
                    i+1, len(xmllist), shard_id+1, total_shards))
                sys.stdout.flush()
                example = img_to_example(img_list[i], img_dir)
                tf_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
    return outputfiles

