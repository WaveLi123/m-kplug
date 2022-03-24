import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tqdm import tqdm
import time
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Running [%s] consumes %.3f seconds" %
              (function.__name__, (t1 - t0))
              # (function.func_name, str(t1 - t0)) #for python2.x
              )
        return result

    return function_timer


class Img2Vec():
    def __init__(self, cuda=False, model='resnet-50', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model
        self.layer = layer

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)
        if self.layer in ('second_last', 'third_last'):
            self.feature_extractor = nn.Sequential(*self.extraction_layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_none_last_vec(self, img):
        image_origin = self.to_tensor(self.scaler(img))
        if image_origin.shape[0] == 1:
            image_origin = image_origin.repeat(3, 1, 1)
        image = self.normalize(image_origin).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_features = self.feature_extractor(image)
        out_features = out_features.permute(0, 2, 3, 1)
        bs_size = out_features.size()[0]
        feature_size = out_features.size()[-1]
        out_features = out_features.view(bs_size, -1, feature_size)
        return out_features

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image_origin = self.to_tensor(self.scaler(img))
        if image_origin.shape[0] == 1:
            image_origin = image_origin.repeat(3, 1, 1)
        image = self.normalize(image_origin).unsqueeze(0).to(self.device)

        if self.model_name == 'alexnet':
            my_embedding = torch.zeros(1, self.layer_output_size)
        else:
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            # print(o.shape)
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)

        # img2vec = Img2Vec(cuda=True)
        # Read in an image
        # img = Image.open(sys.argv[1])
        # print(image)

        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            if self.model_name == 'alexnet':
                return my_embedding.numpy()[0, :]
            else:
                return my_embedding.numpy()[0, :, :, :]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            elif layer == 'second_last':
                layer = list(model.children())[:-2]
            elif layer == 'third_last':
                layer = list(model.children())[:-3]
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'resnet-50':
            model = models.resnet50(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512 * 4
            elif layer == 'second_last':
                layer = list(model.children())[:-2]
            elif layer == 'third_last':
                layer = list(model.children())[:-3]
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet-101':
            model = models.resnet101(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512 * 4
            elif layer == 'second_last':
                layer = list(model.children())[:-2]
            elif layer == 'third_last':
                layer = list(model.children())[:-3]
            else:
                layer = model._modules.get(layer)

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)


@fn_timer
def get_pooling_vector_dir():
    input_dir = sys.argv[1]
    if input_dir[-1] == "/":
        input_name = input_dir[:-1]
    else:
        input_name = input_dir
    input_name = input_name.split("/")[-1]
    vec_mat = []
    img2vec = Img2Vec(cuda=True)
    fout = open("image_fc_names_" + input_name, "w")
    for f in tqdm(os.listdir(input_dir)):
        try:
            img = Image.open(os.path.join(input_dir, f))
            vec = img2vec.get_vec(img)
            vec_mat.append(vec)
            fout.write(f + "\n")
        except:
            print(f)
    vec_mat = np.array(vec_mat).squeeze()
    fout.close()
    np.save("image_fc_vectors_" + input_name, vec_mat)


@fn_timer
def get_patch_vector_dir():
    input_dir = sys.argv[1]
    if input_dir[-1] == "/":
        input_name = input_dir[:-1]
    else:
        input_name = input_dir
    input_name = input_name.split("/")[-1]
    vec_mat = []
    img2vec = Img2Vec(cuda=True, model='resnet-101', layer='second_last')

    fout = open(input_dir + "/../image_patch_names_" + input_name, "w")
    for f in tqdm(os.listdir(input_dir)):
        try:
            img_file = os.path.join(input_dir, f)
            vec = get_patch_vector_file(img_file, img2vec)
            vec_mat.append(vec)
            fout.write(f + "\n")
        except:
            print(f)
    vec_mat = np.array(vec_mat).squeeze()
    fout.close()
    np.save(input_dir + "/../image_patch_vectors_" + input_name, vec_mat)
    sys.stdout.write("The Task Is Finished! Got Numpy Data: {}\n".format(vec_mat.shape))


@fn_timer
def get_patch_vector_dir_single():
    input_dir = sys.argv[1]
    if input_dir[-1] == "/":
        input_name = input_dir[:-1]
    else:
        input_name = input_dir
    input_name = input_name.split("/")[-1]
    vec_mat = []
    img2vec = Img2Vec(cuda=True, model='resnet-101', layer='third_last')

    fout = open(input_dir + "/../image_patch_names_single_" + input_name, "w")
    for f in tqdm(os.listdir(input_dir)):
        try:
            img_file = os.path.join(input_dir, f)
            vec = get_patch_vector_file(img_file, img2vec)
            np.save(input_dir + "/../image_patch_vectors/" + str(f.split('.')[0]), vec)
            # vec_mat.append(vec)
            fout.write(f + "\n")
        except:
            print(f)
    vec_mat = np.array(vec_mat).squeeze()
    fout.close()
    sys.stdout.write("The Task Is Finished! Got Numpy Data: {}\n".format(vec_mat.shape))


# @fn_timer
def get_patch_vector_file(img_file, img2vec):
    img = Image.open(img_file)
    if img.mode != 'RGB':
        img = img.convert("RGB")
    vec = img2vec.get_none_last_vec(img)
    vec_np = vec.cpu().numpy()
    return vec_np


@fn_timer
def get_pooling_vector_file(img_file, img2vec):
    img = Image.open(img_file)
    if img.mode != 'RGB':
        img = img.convert("RGB")
    vec = img2vec.get_vec(img)
    return vec


if __name__ == "__main__":
    # test
    img_file = sys.argv[1]
    #
    # img2vec = Img2Vec(cuda=True, model='resnet-101', layer='second_last')
    # get_patch_vector_file(img_file, img2vec)
    #
    # img2vec = Img2Vec(cuda=True, model='resnet-101')
    # get_pooling_vector_file(img_file, img2vec)

    img2vec = Img2Vec(cuda=True, model='resnet-101')
    get_pooling_vector_file(img_file, img2vec)

    # get_patch_vector_dir()
    get_patch_vector_dir_single()
