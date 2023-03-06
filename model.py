import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
class_name = ['abyssinian(アビシニアン)',
 'american_bulldog(ブルドッグ)',
 'american_pit_bull_terrier(ピットブル)',
 'basset_hound(バセットハウンド)',
 'beagle(ビーグル)',
 'bengal(ベンガル)',
 'birman(バーマン)',
 'bombay(ボンベイ)',
 'boxer(ボクサー)',
 'british_shorthair(ブリティッシュショートヘアー)',
 'chihuahua(チワワ)',
 'egyptian_mau(エジプシャンマウ)',
 'english_cocker_spaniel(イングリッシュコッカースパニエル)',
 'english_setter(イングリッシュセター)',
 'german_shorthaired(ジャーマンショートヘアード)',
 'great_pyrenees(グレートピレニーズ)',
 'havanese(ハバニーズ)',
 'japanese_chin(ちん)',
 'keeshond(キースホンド)',
 'leonberger(レオンベルガー)',
 'maine_coon(メインクーン)',
 'miniature_pinscher(ミニチュアピンシャー)',
 'newfoundland(ニューファンドランド)',
 'persian(ペルシャ猫)',
 'pomeranian(ポメラニアン)',
 'pug(パグ)',
 'ragdoll(ラグドール)',
 'russian_blue(ロシアンブルー)',
 'saint_bernard(セントバーナード)',
 'samoyed(サモエド)',
 'scottish_terrier(スコティッシュテリア)',
 'shiba_inu(柴犬)',
 'siamese(シャム猫)',
 'sphynx(スフィンクス)',
 'staffordshire_bull_terrier(スタフォードブルテリア)',
 'wheaten_terrier(ウィートンテリア)',
 'yorkshire_terrier(ヨークシャーテリア)']

model = torchvision.models.resnet18(weights=True)
model.fc = nn.Linear(512, 37)
param = torch.load("pet_is.pth", map_location=torch.device('cpu'))
model.load_state_dict(param)

def predict(path):
    #im = Image.open(path)
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(IMG_MEAN, IMG_STD)])
    x = transform(path)
    model.eval()
    imx = torch.unsqueeze(x, 0)
    outputs = model(imx)
    #y = torch.argmax(outputs)
    dog = torch.squeeze(outputs)
    s, idx = torch.sort(dog, descending=True)
    a, b, c = idx[0:3].tolist()
    return class_name[a],class_name[b],class_name[c]

#ans = predict("main_53582_8e4bf_detail.jpg")
#print(ans)