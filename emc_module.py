from all_package import *

transform_unsplit_data = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # [transforms.Resize(255),
    #  transforms.CenterCrop(224),
    #  transforms.ToTensor()]
)
DATASET_PATH="E:/大四下行動邊緣/期末專題/train_dataset"
DATASET=datasets.ImageFolder( #len()：資料總筆數
    DATASET_PATH,
    transform_unsplit_data
    )
DATASET_SIZE=len(DATASET)
# print('number of elements in dataset= ',len(DATASET))
# print('type of dataset= ',type(DATASET))
CLASSES=DATASET.classes #label list
CLASS_NUM=len(CLASSES)
print('list of labels= ',CLASSES)
print('number of labels= ',CLASS_NUM)
# indices=list(range(len(DATASET)))
# split = int(numpy.floor(0.85 * len(DATASET)))
# validation = int(numpy.floor(0.70 * split))
# numpy.random.shuffle(indices)
# train_indices, validation_indices, test_indices = (
#     indices[:validation],
#     indices[validation:split],
#     indices[split:],
# )
# print('indices= ',indices)
# print('train size= ',split)
# print('validation= ',validation)
# print('train index= ',train_indices)
# print('validation index= ',validation_indices)
# print('test index= ',test_indices)
BATCH_SIZE=64
INDICES=list(range(DATASET_SIZE))
SPLIT=int(numpy.floor(0.85*len(DATASET)))
VALIDATION=int(numpy.floor(0.70*SPLIT))
numpy.random.shuffle(INDICES)
TRAIN_INDICES,VALIDATION_INDICES,TEST_INDICES=(
    INDICES[:VALIDATION],
    INDICES[VALIDATION:SPLIT],
    INDICES[SPLIT:],
)
TRAIN_SAMPLER=torch.utils.data.sampler.SubsetRandomSampler(TRAIN_INDICES)
VALIDATION_SAMPLER=torch.utils.data.sampler.SubsetRandomSampler(VALIDATION_INDICES)
TEST_SAMPLER=torch.utils.data.sampler.SubsetRandomSampler(TEST_INDICES)
TRAIN_LOADER = torch.utils.data.DataLoader(
    DATASET, batch_size=BATCH_SIZE, sampler=TRAIN_SAMPLER
)
TEST_LOADER = torch.utils.data.DataLoader(
    DATASET, batch_size=BATCH_SIZE, sampler=TEST_SAMPLER
)
VALIDATION_LOADER = torch.utils.data.DataLoader(
    DATASET, batch_size=BATCH_SIZE, sampler=VALIDATION_SAMPLER
)
# print('train loader= ',TRAIN_LOADER)
# print('test loader= ',TEST_LOADER)
# print('validation loader= ',VALIDATION_LOADER)

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #若 CUDA 環境可用，則使用 GPU 計算，否則使用 CPU
MODEL=models.resnet34(pretrained=True) #pytorch官方提供的預訓練模型
MODEL.to(DEVICE) #convert model parameters and buffers to CPU or Cuda
CRITERION=nn.CrossEntropyLoss() #loss function
OPTIMIZER=optim.Adam(MODEL.parameters())
SAVE_PATH="E:/save_module.pth"
EPOCH_TIMES=30

def train_model():
    print('started trainging')
    #pbar=tqdm(TRAIN_LOADER)
    MODEL.train()
    for epoch in tqdm(range(EPOCH_TIMES)):
        trainLoss=0.0

        for idx,(img,label) in tqdm(enumerate(TRAIN_LOADER)):
            img,label=Variable(img.to(DEVICE)),Variable(label.to(DEVICE))
            OPTIMIZER.zero_grad()
            output=MODEL(img)
            loss=CRITERION(output,label)
            loss.backward()
            OPTIMIZER.step()

            trainLoss+=loss.item()
            if idx % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, idx + 1, trainLoss / 1000))
                trainLoss=0.0

    print('Finished Training')

def save_model():
    torch.save(MODEL.state_dict(),SAVE_PATH) #保存模型參數而非整個模型
    #torch.save(MODEL,SAVE_PATH) #保存整個模型

if __name__ == "__main__":

    #Image.register_extension = register_extension
    #Image.register_extensions = register_extensions

    #設定模型參數
    #set_model_parameters()

    #訓練模型
    train_model()

    save_model()

    #測試模型 
    #test_model()