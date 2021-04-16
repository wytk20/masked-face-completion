import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F
from load_data import Load_Face_Mask
from build_network import FaceRecov_Net

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, input,target):
        return F.mse_loss(input, target)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        model.train()  # Set model to training mode
        running_loss = 0.0
        num_samples = 0.0
        num = 0

        # Iterate over data.
        for face_img, mask_img in dataloaders:
            face_img = face_img.to(device)
            mask_img = mask_img.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(mask_img)
                loss = criterion(outputs, face_img)

                loss.backward()
                optimizer.step()

            # statistics
            num +=1
            num_samples += mask_img.size(0)
            running_loss += loss.item() * mask_img.size(0)

            if num % 1000 == 0:
                print('average running loss:{:.4f}'.format(running_loss/num_samples))

        epoch_loss = running_loss / num_samples

        print('**********************************')
        print('eopch Loss: {:.4f}'.format(epoch_loss))
        print('')
        torch.save(model.state_dict(), 'checkpoint/imagenet_recovery_epoch_{}.pth'.format(epoch))

    return model


if __name__ == '__main__':
    face_root = '/home/datadisk/faceset/CASIA-face/CASIA-cropface/'
    mask_root = '/home/datadisk/faceset/CASIA-face/CASIA-facemask/'

    loader = Load_Face_Mask(face_root, mask_root)
    dataloaders= torch.utils.data.DataLoader(loader,batch_size=16, shuffle=True, num_workers=4)

    model_ft = FaceRecov_Net(num_classes=501)
    criterion = ContentLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)
    optimizer_ft = optim.Adadelta(model_ft.parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = model_ft.to(device)
    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=30)

    print('ok')

'''
record:
Epoch 0/29
average running loss:0.5028
average running loss:0.4697
average running loss:0.4567
average running loss:0.4490
average running loss:0.4426
average running loss:0.4373
**********************************
eopch Loss: 0.4370

Epoch 1/29
average running loss:0.4112
average running loss:0.4070
average running loss:0.4040
average running loss:0.4026
average running loss:0.4014
average running loss:0.4003
**********************************
eopch Loss: 0.4003

Epoch 2/29
average running loss:0.3947
average running loss:0.3934
average running loss:0.3929
average running loss:0.3931
average running loss:0.3928
average running loss:0.3924
**********************************
eopch Loss: 0.3922

Epoch 3/29
average running loss:0.3871
average running loss:0.3891
average running loss:0.3898
average running loss:0.3891
average running loss:0.3887
average running loss:0.3894
**********************************
eopch Loss: 0.3894

Epoch 4/29
average running loss:0.3878
average running loss:0.3894
average running loss:0.3888
average running loss:0.3875
average running loss:0.3874
average running loss:0.3876
**********************************
eopch Loss: 0.3876

Epoch 5/29
average running loss:0.3886
average running loss:0.3872
average running loss:0.3874
average running loss:0.3865
average running loss:0.3861
average running loss:0.3860
**********************************
eopch Loss: 0.3861

Epoch 6/29
average running loss:0.3862
average running loss:0.3871
average running loss:0.3851
average running loss:0.3846
average running loss:0.3846
average running loss:0.3849
**********************************
eopch Loss: 0.3849

Epoch 7/29
average running loss:0.3825
average running loss:0.3831
average running loss:0.3834
average running loss:0.3833
average running loss:0.3838
average running loss:0.3838
**********************************
eopch Loss: 0.3838

Epoch 8/29
average running loss:0.3856
average running loss:0.3838
average running loss:0.3837
average running loss:0.3832
average running loss:0.3833
average running loss:0.3828
**********************************
eopch Loss: 0.3829

Epoch 9/29
average running loss:0.3796
average running loss:0.3812
average running loss:0.3809
average running loss:0.3810
average running loss:0.3813
average running loss:0.3822
**********************************
eopch Loss: 0.3821

Epoch 10/29
average running loss:0.3774
average running loss:0.3783
average running loss:0.3805
average running loss:0.3811
average running loss:0.3808
average running loss:0.3809
**********************************
eopch Loss: 0.3813

Epoch 11/29
average running loss:0.3823
average running loss:0.3804
average running loss:0.3810
'''

