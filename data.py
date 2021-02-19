import numpy as np
import tensorflow as tf
import tf2lib as tl
import os


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset

def make_zip_dataset_from_single_dir(img_dir, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    lve_imgpaths = tf.io.gfile.glob(os.path.join(img_dir, '[!2018]*.png'))
    lv_imgpaths = tf.io.gfile.glob(os.path.join(img_dir, '2018*.png'))
    #zip two datasets aligned by the longer one
    if repeat:
        lve_repeat = lv_repeat = None  # cycle both
    else:
        if len(lve_imgpaths) >= len(lv_imgpaths):
            lve_repeat = 1
            lv_repeat = None  # cycle the shorter one
        else:
            lve_repeat = None  # cycle the shorter one
            lv_repeat = 1

    lve_dataset = make_dataset(lve_imgpaths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=lve_repeat)
    lv_dataset = make_dataset(lv_imgpaths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=lv_repeat)

    lve_lv_dataset = tf.data.Dataset.zip((lve_dataset, lv_dataset))
    len_dataset = max(len(lve_imgpaths), len(lv_imgpaths)) // batch_size

    return lve_lv_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)
