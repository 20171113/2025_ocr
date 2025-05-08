import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader, Dataset
import pandas as pd
import cv2
import numpy as np

class GaussianNoise_(object):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, img):
        img_array = np.array(img)
        img_final = img_array + np.random.normal(loc = 0, scale = self.sigma, size = img_array.shape)
        img_final = np.clip(img_final.astype(np.uint8), a_min=0, a_max=255)
        img_final = Image.fromarray(img_final.astype(np.uint8))
        return img_final
    
class MotionBlur_(object):
    def __init__(self, size=10, angle=0):
        self.size = size
        self.angle = angle

    def __call__(self, img):
        img_ = np.array(img)
        img_ = self.motion_blur(img_, self.size, self.angle)
        img_final = Image.fromarray(img_)
        
        return img_final
    
    def motion_blur(self, img, size=None, angle=None):
        k = np.zeros((size, size), dtype=np.float32)
        k[(size-1)//2, :] = np.ones(size, dtype=np.float32)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size/2-0.5, size/2-0.5), angle, 1.0), (size, size))
        k = k * (1.0/np.sum(k))
        return cv2.filter2D(img, -1, k)

class DigitData_TTAug_v1(Dataset):
    '''
    digit_data를 pytorch에서 사용 가능한 형태로 변경
    '''
    def __init__(self, path='../../../Data5/sujin/2025_OCR', size=64, transform_list = None): # transform 추가
        '''
        path: digit_data의 경로
        size: image input의 크기 (default: 64 -> 64 x 64 image를 입력으로 사용)
        split: train, validation 구분
        '''
        self.path = path
        self.size = (size, size)

        # 각 instance 별 경로 읽음
        self.image_files = pd.read_csv(os.path.join(path, 'data/dataset/dataset.csv'))

        if transform_list is not None:
            self.transform0 = transform_list[0]
            self.transform1 = transform_list[1]
            self.transform2 = transform_list[2]
            self.transform3 = transform_list[3]
            self.transform4 = transform_list[4]
        else:
            self.transform0 = transforms.Compose([transforms.Resize((64,64)), 
                                             transforms.ToTensor()])
            self.transform1 = transforms.Compose([GaussianNoise_(2),
                                             transforms.Resize((64,64)), 
                                             transforms.RandomAffine(0, shear=(27, 33)),
                                             transforms.ToTensor()])
            self.transform2 = transforms.Compose([GaussianNoise_(2),
                                             transforms.Resize((64,64)), 
                                             transforms.RandomAffine(degrees = 15, shear=(27, 33)),
                                             transforms.ToTensor()])
            self.transform3 = transforms.Compose([GaussianNoise_(2),
                                             transforms.Resize((64,64)), 
                                             transforms.RandomAffine(degrees = 0, shear=(-30, 30)),
                                             MotionBlur_(),
                                             transforms.ToTensor()])
            self.transform4 = transforms.Compose([GaussianNoise_(2),
                                             transforms.Resize((64,64)), 
                                             transforms.RandomAffine(degrees = 0, shear=(-30, 30)),
                                             transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                             transforms.ToTensor()])

    def __len__(self):
        # len 함수로 표시되는 output
        # 총 data instance의 수
        return len(self.image_files)

    def __getitem__(self, idx):
        # indexing을 하였을 때 나오는 output (예시 Data[5] 등의 output)
        path = os.path.join(self.path, self.image_files.iloc[idx,0])
        img = Image.open(path).convert('RGB') # image 읽기
        img0 = self.transform0(img) # image를 뉴럴 네트워크의 input으로 사용하기 위하여 transformation
        img1 = self.transform1(img) # image를 뉴럴 네트워크의 input으로 사용하기 위하여 transformation
        img2 = self.transform2(img) # image를 뉴럴 네트워크의 input으로 사용하기 위하여 transformation
        img3 = self.transform3(img) # image를 뉴럴 네트워크의 input으로 사용하기 위하여 transformation
        img4 = self.transform4(img) # image를 뉴럴 네트워크의 input으로 사용하기 위하여 transformation
        target = int(self.image_files.iloc[idx, 1])
        img = [img0, img1, img2, img3, img4]
        return img, target