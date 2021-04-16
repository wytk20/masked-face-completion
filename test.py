
import torch
from PIL import Image
import torchvision.transforms as transforms
from build_network import FaceRecov_Net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(tensor, name, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    tensor = torch.squeeze(tensor, dim=0)
    tensor.mul_(std).add_(mean)

    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = unloader(image)
    image.save(name)

if __name__ == '__main__':
    import time

    # face_root = '/home/datadisk/faceset/CASIA-face/CASIA-cropface/'
    # mask_root = '/home/datadisk/faceset/CASIA-face/CASIA-facemask/'
    # loader = Load_Face_Mask(face_root, mask_root)
    # face_img,mask_img = loader.__getitem__(6000)

    mask_path = '/home/datadisk/faceset/CASIA-face/CASIA-facemask/2003700/004.jpg'
    #mask_path = '/home/cbzeng/Downloads/train/5/179.jpg'
    #mask_path = '/home/cbzeng/Downloads/train/10/ki.jpg'
    mask_img = Image.open(mask_path)
    mask_img = transforms.Resize(224)(mask_img)
    mask_img = transforms.ToTensor()(mask_img)
    mask_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(mask_img)

    model_ft = FaceRecov_Net()
    model_ft.load_state_dict(torch.load('checkpoint/imagenet_recovery_epoch_2.pth'))

    model_ft = model_ft.to(device)
    model_ft.eval()
    with torch.no_grad():
        mask_img = torch.unsqueeze(mask_img, dim=0)
        mask_img = mask_img.to(device)

        output = model_ft(mask_img)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        #save_name = '/home/cbzeng/Downloads/0000117_025.png'
        save_name = '/home/cbzeng/Downloads/fig10/recovery_2003700_004_e2.png'
        imshow(output, save_name ,mean, std)
        time.sleep(1)


    print('ok')



