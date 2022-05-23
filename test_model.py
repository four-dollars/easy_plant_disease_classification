from all_package import *
import os.path
from emc_module import CLASSES,DEVICE,SAVE_PATH

IMG_PATH='E:/大四下行動邊緣/期末專題/test_dataset/'
transform = transforms.Compose( #定義資料集的transformation
    [transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def test_model():
    model=models.resnet34(pretrained=True)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    model=model.to(DEVICE)
    torch.no_grad()
    if(os.path.exists(IMG_PATH+IMG_NAME+'.jpg')):
        img=Image.open(IMG_PATH+IMG_NAME+'.jpg')
        img=transform(img).unsqueeze(0)
        img_=img.to(DEVICE)
        output=model(img_)
        _,predicted=torch.max(output,1)
        print('this picture maybe:',CLASSES[predicted[0]])
    else:
        print('this file not exist in folder')

if __name__=='__main__':
    while(True):
        IMG_NAME=input('img name:')
        test_model()
