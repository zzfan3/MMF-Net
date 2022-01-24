import io
import requests
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import cv2
import os

from modeling import build_model
from config import cfg

# input image

# LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'
# jsonfile = r'D:\python\Camtest\labels.json'
# with open(jsonfile, 'r') as load_f:
#     load_json = json.load(load_f)

# networks such as googlenet, resnet, densenet already use global average pooling at the end,
# so CAM could be used directly
#
# model_id = 1
# if model_id == 1:
#     net = models.squeezenet1_1(pretrained=False)
#     pthfile = r'E:\anaconda\app\envs\luo\Lib\site-packages\torchvision\models\squeezenet1_1.pth'
#     net.load_state_dict(torch.load(pthfile))
#     finalconv_name = 'features'  # this is the last conv layer of the network
# elif model_id == 2:
#     net = models.resnet18(pretrained=False)
#     finalconv_name = 'layer4'
# elif model_id == 3:
#     net = models.densenet161(pretrained=False)
#     finalconv_name = 'features'
num_classes = 751
image_path = 'test_image'
image_name = '0688_c1s4_032081_01.jpg'
n = 2
if n == 1:
    model_path = '/home/dell/D/dell/fjw/practice/ReID/MMF-Net/output/market1501/baseline10_all/resnet50_model_89280.pth'
    model_idx = 'all'
    w_s = 181
else:
    model_path = '/home/dell/D/dell/fjw/practice/ReID/MMF-Net/output/market1501/baseline13_base/resnet50_model_89280.pth'
    model_idx = 'base'  # all\base
    w_s = 161 # 181\161

model = build_model(num_classes)
#model.load_param(cfg.TEST.WEIGHT)
model_state_dict = torch.load(model_path)
# model_state_dict = model_state_dict['model']
now_state_dict = model.state_dict()
now_state_dict.update(model_state_dict)
model.load_state_dict(now_state_dict)



model.eval()
# print(model)

# hook the feature extractor
# features_blobs = []     # 目标特征图（b,c,h,w)


# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())
#
#
# net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(model.parameters())

weight_softmax = np.squeeze(params[w_s].data.detach().numpy())    # 分类全连接层的权重


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 128)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    normalize
])

# response = requests.get(IMG_URL)
# img_pil = Image.open(io.BytesIO(response.content))
# img_pil.save('test_image/1.jpg')
img_pil = Image.open(os.path.join(image_path, image_name))
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
features_blobs, logit = model(img_variable)
features_blobs = features_blobs.detach().numpy()
# download the imagenet category list
# classes = {int(key): value for (key, value)
#            in requests.get(LABELS_URL).json().items()}
# classes = {int(key): value for (key, value)
#           in load_json.items()}


# 结果有1000类，进行排序
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)      #预测分类结果从概率大到小排序
probs = probs.numpy()
idx = idx.numpy()

# output the prediction 取前5
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], idx[i]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs, weight_softmax, [idx[0]])     #大小和输入图像大小一致

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s' % idx[0])
img = cv2.imread(os.path.join(image_path, image_name))
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite(image_path + '/' + model_idx + "_CAM_" + image_name, result)