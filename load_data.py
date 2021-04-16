
import torch
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import os

def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

class Load_Face_Mask(data.Dataset):
    def __init__(self,face_root, mask_root):
        self.List_face_files = GetFileFromThisRootDir(face_root)
        self.List_mask_files = GetFileFromThisRootDir(mask_root)

    def __len__(self):
        return len(self.List_face_files)

    def __getitem__(self, index):
        face_path = self.List_face_files[index]
        mask_path = self.List_mask_files[index]

        #print(face_path)
        #print(mask_path)
        face_img = Image.open(face_path)
        mask_img = Image.open(mask_path)

        face_img = transforms.Resize(224)(face_img)
        face_img = transforms.ToTensor()(face_img)
        face_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(face_img)

        mask_img = transforms.Resize(224)(mask_img)
        mask_img = transforms.ToTensor()(mask_img)
        mask_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(mask_img)

        return face_img,mask_img

def imshow(tensor, name, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    tensor.mul_(std).add_(mean)
    #tensor.sub_(mean).div_(std)

    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    #image = image * std + mean
    image = unloader(image)
    image.save(name)

    # plt.imshow(image)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(50) # pause a bit so that plots are updated

if __name__ == '__main__':
    import time
    face_root = '/home/datadisk/faceset/CASIA-face/CASIA-cropface/'
    mask_root = '/home/datadisk/faceset/CASIA-face/CASIA-facemask/'

    loader = Load_Face_Mask(face_root, mask_root)
    face_img,mask_img = loader.__getitem__(90)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    imshow(face_img, 'face.png',mean, std)
    time.sleep(2)
    imshow(mask_img, 'mask.png',mean, std)
    time.sleep(2)

    print('ok')


    # List_face_files  = GetFileFromThisRootDir(face_root)
    # List_mask_files = GetFileFromThisRootDir(mask_root)
    # num = 0
    # for i, face in enumerate(List_face_files):
    #     mask = List_mask_files[i]
    #     fn = face.split('/')[-1]
    #     mn = mask.split('/')[-1]
    #     if fn != mn:
    #         print(face)
    #     else:
    #         num +=1

