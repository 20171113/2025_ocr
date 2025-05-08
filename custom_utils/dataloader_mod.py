'''
KAMP guidebook 코드 변경한 것
'''
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

class DigitData(Dataset):
    '''
    digit_data를 pytorch에서 사용 가능한 형태로 변경
    '''
    def __init__(self, path='../../../Data5/sujin/2025_OCR', size=64, transform=None): # transform 추가
        '''
        path: digit_data의 경로
        size: image input의 크기 (default: 64 -> 64 x 64 image를 입력으로 사용)
        split: train, validation 구분
        '''
        self.path = path
        self.size = (size, size)

        # 각 instance 별 경로 읽음
        self.image_files = pd.read_csv(os.path.join(path, 'data/dataset/dataset.csv'))

        # 이미지를 뉴럴 네트워크의 input으로 사용하기 위하여 transformation
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor()
            ])

    def __len__(self):
        # len 함수로 표시되는 output
        # 총 data instance의 수
        return len(self.image_files)

    def __getitem__(self, idx):
        # indexing을 하였을 때 나오는 output (예시 Data[5] 등의 output)
        path = os.path.join(self.path, self.image_files.iloc[idx,0])
        img = Image.open(path).convert('RGB') # image 읽기
        img = self.transform(img) # image를 뉴럴 네트워크의 input으로 사용하기 위하여 transformation
        target = int(self.image_files.iloc[idx, 1])
        return img, target

def make_dataloader(path='../../../Data5/sujin/2025_OCR', size=64, batch_size=64, transform=None, shuffle = True):
    '''
    DigitData를 사용하여 뉴럴네트워크를 학습할 때 데이터의 순서나 한 번의 iteration에 사용되는 batch를 생성
    path: digit_data의 경로
    size: image input의 크기
    batch_size: 한 번의 iteration에 사용될 데이터의 수
    '''
    # Dataset 생성
    dataset = DigitData(path, size=size, transform=transform)
    # Batch 별로 input을 나누어 사용할 수 잇도록 data loader 생성
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


if __name__ == '__main__':
    data = DigitData()
    loader = DataLoader(data, 2, True)
    for (img, target) in loader:
        print(img)
        print(target)
        break
        
        
        
def get_transforms(data_loader):
    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data, _ in tqdm(data_loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]

        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    custom_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    
    return custom_transform