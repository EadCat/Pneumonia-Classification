import datetime
import numpy as np
from typing import Union
import cv2
from parameters import *

def snapshot_maker(param_dict, dir:str):
    # record <.pth> model infomation snapshot.
    with open(dir, 'w') as file:
        for key, value in param_dict.items():
            file.write(key + ' : ' + str(value) + '\n')
        time_now = datetime.datetime.now()
        file.write('record time : ' + time_now.strftime('%Y-%m-%d %H:%M:%S'))


def write_line(dict_in:dict, dir:str):
    # record loss in real time.
    import os
    import torch
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    with open(dir, 'a') as file:
        for key, value in dict_in.items():
            if isinstance(key, torch.Tensor):
                key = float(key)
            if isinstance(value, torch.Tensor):
                value = float(value)
            if isinstance(key, float):
                key = round(key, 4)
            if isinstance(value, float):
                value = round(value, 6)
            file.write(str(key) + ' : ' + str(value) + '\n')


def cuda2np(tensor) -> np.ndarray:
    # cuda tensor -> cpu numpy
    arr = tensor.cpu()
    arr = arr.detach().numpy()
    return arr


def tensorview(Intensor, batch_idx):
    # show target tensor
    arr = cuda2np(Intensor)
    print(arr[batch_idx])


class Captioner:
    def __init__(self, text:str, point:Union[tuple, list], fontface=cv2.FONT_HERSHEY_DUPLEX,
                 fontsize=2, color=(0, 0, 255), thickness=2, linetype=cv2.LINE_AA):
        self.text = text
        self.location = point
        self.fontface = fontface
        self.fontsize = fontsize
        self.color = color
        self.thickness = thickness
        self.linetype = linetype

    def fix_text(self, text):
        self.text = text

    def write(self, img:np.ndarray):
        cv2.putText(img, text=self.text, org=self.location, fontFace=self.fontface,
                    fontScale=self.fontsize, color=self.color, thickness=self.thickness,
                    lineType=self.linetype)

def dimension_change(Intensor) -> np.ndarray:
    # Pytorch dim -> opencv dim
    #  C X H X W  ->  H X W X C
    try:
        image = cuda2np(Intensor)
    except TypeError as te:
        image = np.array(Intensor)
    finally:
        pass

    if len(image.shape) == 3:
        image = image.transpose((1, 2, 0))
    elif len(image.shape) == 4:
        image = image.transpose((0, 2, 3, 1))
    else:
        'weird dimension inputs.'

    return image


def imgstore(inarray, nums:int, save_dir:str, epoch:Union[int, str], filename='', cls='pred'):
    # function for saving prediction image.
    import os
    import cv2

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(filename, str) or len(filename) == 1:  # stores only one image, batch == 1
        image = inarray
        if isinstance(filename, list):
            filename = filename[0]
        if isinstance(epoch, str):
            cv2.imwrite(os.path.join(save_dir, cls + '_' + epoch + '_[' + filename + '].png'), image)
        else:
            cv2.imwrite(os.path.join(save_dir, cls+'_'+'epoch_'+str(epoch)+'_['+filename+'].png'), image)

    elif isinstance(filename, list):  # stores <nums:int> images, batch > 1
        img_list = []

        for i, img in enumerate(inarray):
            if i == nums:
                break
            img_list.append(img)

        for idx, unit in enumerate(img_list):
            if isinstance(epoch, str):
                cv2.imwrite(os.path.join(save_dir, cls + '_' + epoch + '_[' + filename[idx] + '].png'), unit)
                print(f"{os.path.join(save_dir, cls+'_'+epoch+'_['+filename[idx]+'].png')} saved.")
            else:
                cv2.imwrite(os.path.join(save_dir, cls+'_'+'epoch_'+str(epoch)+'_['+filename[idx]+'].png'), unit)