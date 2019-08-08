import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from PIL import Image
import os
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io, transform
import numpy as np
import time

def key(name):
    if name[-10:-4] == 'input0':
        a = name[:-11]
        return int(a)
    else:
        a = name[:-12]
        return int(a)

def order_dataset():
    dir_rcan = os.path.join(os.path.expanduser('~'),'robotics_drl/data/rcan_data')
    names = sorted(os.listdir(dir_rcan), key=key)
    len_dataset = len(names)

    prev_ext = -1
    eq_i = 0
    saving_i = 0
    to_remove = False

    for name in names:
        standby = []
        if name[-10:-4] == 'input0':
            ext = int(name[:-11])
            standby.append(str(ext) + '_' + 'input0' + '.png')
            for i in range(2):
                other_name = str(ext) + '_' + 'output%s'%(i) + '.png'
                if not os.path.exists(dir_rcan + '/'+  other_name):
                    to_remove = True
                else:
                    standby.append(other_name)

        elif name[-11:-4] == 'output0':
            ext = int(name[:-12])
            standby.append(str(ext) + '_' + 'output0' + '.png')
            other_names = ['output1', 'input0']
            for i in range(2):
                other_name = str(ext) + '_' + other_names[i] + '.png'
                if not os.path.exists(dir_rcan + '/'+  other_name):
                    to_remove = True
                else:
                    standby.append(other_name)

        elif name[-11:-4] == 'output1':
            ext = int(name[:-12])
            standby.append(str(ext) + '_' + 'output1' + '.png')
            other_names = ['output0', 'input0']
            for i in range(2):
                other_name = str(ext) + '_' + other_names[i] + '.png'
                if not os.path.exists(dir_rcan + '/'+  other_name):
                    to_remove = True
                else:
                    standby.append(other_name)

        if to_remove:
            count_rem += 1
            for name in standby:
                try:
                    os.remove(dir_rcan + '/' + name)
                except:
                    pass
        to_remove = False

    names = sorted(os.listdir(dir_rcan), key=key)
    for idx ,name in enumerate(names):
        k = idx // 3
        new_name = str(k) + '_'
        if name[-10:-4] == 'input0':
            new_name = new_name + 'input0.png'
            ext = int(name[:-11])
        elif name[-11:-4] == 'output0':
            new_name = new_name + 'output0.png'
            ext = int(name[:-12])
        elif name[-11:-4] == 'output1':
            new_name = new_name + 'output1.png'
            ext = int(name[:-12])

        os.rename(os.path.join(dir_rcan,name),os.path.join(dir_rcan,new_name))

class RCAN_Dataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform

    def __len__(self):
        self._set_dir()
        return len([name for name in os.listdir('.') if os.path.isfile(name)])//3

    def __getitem__(self,idx):
        self._set_dir()

        x_image = io.imread('%s_input0.png'%(idx))

        y_images = []
        for i in range(2):
            y_images.append(io.imread('%s_output%s.png'%(idx,i)))

        if self.transform:
            x_image = self.transform(x_image)
            y_images[0] = self.transform(y_images[0])
            y_images[1] = self.transform(y_images[1])

        sample = {'input': x_image, 'output':y_images}
        return sample

    def _set_dir(self):
        home = os.path.expanduser('~')
        dir = os.path.join(home,self.dir)
        os.chdir(dir)

class RCAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv7 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv8 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv9 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv10 = nn.Conv2d(2048, 2048, 1)
        self.conv11 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.conv12 = nn.Conv2d(2048, 2048, 1)
        self.conv13 = nn.Conv2d(2048, 512, 3, padding=1)
        self.conv14 = nn.Conv2d(1024, 1024, 1)
        self.conv15 = nn.Conv2d(1024, 256, 3, padding=1)
        self.conv16 = nn.Conv2d(512, 512, 1)
        self.conv17 = nn.Conv2d(512, 128, 3, padding=1)
        self.conv18 = nn.Conv2d(256, 256, 1)
        self.conv19 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv20 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv21 = nn.Conv2d(64, 32, 7, padding=3)
        self.conv22 = nn.Conv2d(32, 4, 7, padding=3)

    def forward(self, x0):
        x1 = F.relu(F.instance_norm(self.conv1(x0)))
        x2 = F.relu(F.instance_norm(self.conv2(x1)))
        x3 = F.relu(F.instance_norm(self.conv3(x2)))
        x4 = F.avg_pool2d(F.relu(F.instance_norm(self.conv4(x3))), 2)
        x5 = F.avg_pool2d(F.relu(F.instance_norm(self.conv5(x4))), 2)
        x6 = F.avg_pool2d(F.relu(F.instance_norm(self.conv6(x5))), 2)
        x7 = F.avg_pool2d(F.relu(F.instance_norm(self.conv7(x6))), 2)
        x8 = F.avg_pool2d(F.relu(F.instance_norm(self.conv8(x7))), 2)
        x9 = F.interpolate(F.relu(F.instance_norm(self.conv9(x8))), scale_factor=2, mode='bilinear', align_corners=False)
        x10 = F.interpolate(F.relu(F.instance_norm(self.conv10(torch.cat([x9, x7], dim=1)))), scale_factor=2, mode='bilinear', align_corners=False)
        x11 = F.relu(F.instance_norm(self.conv11(x10)))
        x12 = F.interpolate(F.relu(F.instance_norm(self.conv12(torch.cat([x11, x6], dim=1)))), scale_factor=2, mode='bilinear', align_corners=False)
        x13 = F.relu(F.instance_norm(self.conv13(x12)))
        x14 = F.interpolate(F.relu(F.instance_norm(self.conv14(torch.cat([x13, x5], dim=1)))), scale_factor=2, mode='bilinear', align_corners=False)
        x15 = F.relu(F.instance_norm(self.conv15(x14)))
        x16 = F.interpolate(F.relu(F.instance_norm(self.conv16(torch.cat([x15, x4], dim=1)))), scale_factor=2, mode='bilinear', align_corners=False)
        x17 = F.relu(F.instance_norm(self.conv17(x16)))
        x18 = F.interpolate(F.relu(F.instance_norm(self.conv18(torch.cat([x17, x3], dim=1)))), scale_factor=2, mode='bilinear', align_corners=False)
        x19 = F.relu(F.instance_norm(self.conv19(x18)))
        x20 = F.interpolate(F.relu(F.instance_norm(self.conv20(x19))), scale_factor=2, mode='bilinear', align_corners=False)
        x21 = F.relu(F.instance_norm(self.conv21(x20)))
        x22 = (self.conv22(x21))  # TODO: CPU fix waiting on https://github.com/pytorch/pytorch/issues/20583
        return x22  # 8x downscaled?

def main():
    order_dataset()
    training_dir = os.path.join(os.path.expanduser('~'),'robotics_drl/data/rcan_training' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resize = T.Compose([T.ToTensor()])

    resize_test = T.Compose([T.ToPILImage(),
                        T.Resize((128,128)),
                        T.ToTensor()])

    dataset = RCAN_Dataset('robotics_drl/data/rcan_data', transform=T.Compose([T.ToTensor()]))
    dataset_size = len(dataset)
    valid_split = 0.01
    EPOCH = 5
    BATCH_SIZE = 8
    TOTAL_STEPS = EPOCH * int(np.floor(dataset_size/BATCH_SIZE))
    # for i_batch, sampled_batch in enumerate(dataloader):
    #     print(i_batch,sampled_batch['input'].size(), sampled_batch['output'][0].size(), sampled_batch['output'][1].size())

    # Test real images
    test_images = torch.tensor([])
    test_path = os.path.join(os.path.expanduser('~'),'robotics_drl')
    for i in range(1,4):
        test_images = torch.cat((test_images,(resize_test(io.imread(test_path + '/test_rcan%s.jpg'%(i)))).view(1,-1,128,128)), dim=0)

    # Valid simulation images
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    np.random.seed(40)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)

    net = RCAN().to(device)
    optimiser = optim.Adam(net.parameters(), lr=0.001)

    steps=0

    with tqdm(total=TOTAL_STEPS) as pbar:
        for i in range(EPOCH):
            for i_batch, sampled_batch in enumerate(train_loader):
                steps += 1
                pbar.update(1)
                Y = net(sampled_batch['input'].to(device))
                loss = F.smooth_l1_loss(Y, torch.cat((sampled_batch['output'][0],sampled_batch['output'][1]), 1).to(device))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                if steps % 60 == 0 and steps != 0:
                    # pbar.set_description('Loss: %s' %(str(loss.item())[:6]))
                    # save_image(Y.squeeze()[0:3,:,:].view(-1,3,128,128),'%s_canonical.png'%(i),normalize=True)
                    # save_image(Y.squeeze()[-1,:,:].view(-1,1,128,128),'%sdepth.png'%(i),normalize=True)
                    net.eval()
                    # Simulation image validation
                    valid_loss = 0
                    for i_batch_valid, sampled_batch_valid in enumerate(valid_loader):
                        with torch.no_grad():
                            output_valid = net(sampled_batch_valid['input'].to(device))
                            valid_loss += F.smooth_l1_loss(output_valid, torch.cat((sampled_batch_valid['output'][0],sampled_batch_valid['output'][1]), 1).to(device))

                    # Real image validation
                    output_test = net(test_images.to(device))
                    canonical_imgs = output_test.view(-1,4,128,128)[:,0:3,:,:].cpu()
                    depth_imgs = output_test.view(-1,4,128,128)[:,-1,:,:].view(-1,1,128,128).expand(-1,3,128,128).cpu()
                    os.chdir(training_dir)
                    save_image(torch.cat((test_images, canonical_imgs, depth_imgs),dim=0), '%s_real_valid.png'%(steps), nrow=3, normalize=False)

                    torch.save(net.state_dict(),os.path.join(training_dir,'model.pt'))

                    print('Loss: %s' %(str(loss.item())[:6]))
                    print('Validation Loss: %s' %(str(valid_loss.item())[:6]))

                    net.train()
if __name__ == "__main__":
    main()
