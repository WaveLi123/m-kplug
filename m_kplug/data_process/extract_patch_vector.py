""" 多个大文件的shuffle
"""

import os
import sys
import time
import mmap
from random import shuffle
import numpy as np
from tqdm import tqdm
import traceback

cur_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(cur_path)
sys.path.insert(0, base_path)

from raw_data_process.aaai_step2_img_to_vec import Img2Vec, get_patch_vector_file, get_pooling_vector_file


def get_img_patch_file(sku_file, img2vec=None):
    vec_mat = []
    sku_map = dict()
    idx = 0

    file_num = sku_file.split('.')[-2]
    fout_map = open(".".join(sku_file.split('.')[:-1] + ['sku_map']), "w")
    fout_img2ids = open("/".join(sku_file.split('/')[:-2] + ['part_' + str(file_num) + '.img2ids']), "w")
    fout_sku_err = open("/".join(sku_file.split('/')[:-2] + ['part_' + str(file_num) + '.sku_err']), "w")
    img_dir = "/".join(sku_file.split('/')[:-3] + ['img'])
    img_dir_rgb = "/".join(sku_file.split('/')[:-3] + ['img_rgb'])
    vec_np_file = "/".join(sku_file.split('/')[:-2] + ['image_patch_vectors_part_' + str(file_num)])

    if not img2vec:
        img2vec = Img2Vec(cuda=True, model='resnet-101', layer='default')

    for line in tqdm(open(sku_file).readlines()):
        sku = line.strip()
        try:
            img_file = os.path.join(img_dir_rgb, sku + '.jpg')
            if not os.path.exists(img_file):
                img_file = os.path.join(img_dir, sku + '.jpg')

            if sku not in sku_map:
                sku_map[sku] = idx
                vec = get_patch_vector_file(img_file, img2vec)
                vec_mat.append(vec)
                fout_map.write(sku + "\n")
                idx += 1
            fout_img2ids.write(str(sku_map[sku]) + "\n")
        except Exception as error:
            sys.stderr.write(sku + "\t" + str(error) + "\n")
            fout_sku_err.write(sku + "\t" + str(error) + "\n")

    vec_mat = np.array(vec_mat).squeeze()
    np.save(vec_np_file, vec_mat)

    fout_map.close()
    fout_img2ids.close()
    sys.stdout.write("The Task Is Finished! Got Numpy Data: {}, File: {}\n".format(vec_mat.shape, sku_file))


def get_img_patch_vec_file(sku_file, img_dir, out_dir, img2vec=None, file_tag=None):
    vec_mat = []
    sku_map = dict()
    idx = 0
    if not file_tag:
        file_tag = sku_file.split('/')[-1].split('.')[0]
    fout_map = open('/'.join([out_dir] + [file_tag + '.sku_map']), "w")
    fout_img2ids = open("/".join([out_dir] + [file_tag + '.img2ids']), "w")
    fout_sku_err = open("/".join([out_dir] + [file_tag + '.sku_err']), "w")

    img_dir_rgb = "/".join(img_dir.split('/')[:-1] + ['img_rgb'])
    vec_np_file = "/".join([out_dir] + ['image_patch_vectors_' + str(file_tag)])

    if not img2vec:
        img2vec = Img2Vec(cuda=True, model='resnet-101', layer='default')

    for line in tqdm(open(sku_file).readlines()):
        sku = line.strip()
        try:
            img_file = os.path.join(img_dir_rgb, sku + '.jpg')
            if not os.path.exists(img_file):
                img_file = os.path.join(img_dir, sku + '.jpg')

            if sku not in sku_map:
                sku_map[sku] = idx
                # vec = get_patch_vector_file(img_file, img2vec)
                vec = get_pooling_vector_file(img_file, img2vec)
                vec_mat.append(vec)
                fout_map.write(sku + "\n")
                idx += 1
            fout_img2ids.write(str(sku_map[sku]) + "\n")
        except Exception as error:
            sys.stderr.write(sku + "\t" + str(traceback.format_exc()) + "\n")
            fout_sku_err.write(sku + "\t" + str(traceback.format_exc()) + "\n")

    vec_mat = np.array(vec_mat).squeeze()
    np.save(vec_np_file, vec_mat)

    fout_map.close()
    fout_img2ids.close()
    sys.stdout.write("The Task Is Finished! Got Numpy Data: {}, File: {}\n".format(vec_mat.shape, sku_file))


if __name__ == "__main__":
    # get_img_patch_file(sku_file=sys.argv[1])

    if len(sys.argv) < 4:
        sku_file = '/data/liweikang/jdsum/cases_bags/bpe_mg/valid.sku'
        img_dir = '/data/liweikang/jdsum/cases_bags/img'
        out_dir = '/data/liweikang/jdsum/cases_bags/bpe_mg_batch/'
        file_tag = None
    else:
        sku_file = sys.argv[1]
        img_dir = sys.argv[2]
        out_dir = sys.argv[3]
        file_tag = sys.argv[4]

    get_img_patch_vec_file(sku_file, img_dir, out_dir, file_tag=file_tag)
