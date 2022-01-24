import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        #init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        #init.constant(m.bias.data, 0.0)


class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1 = nn.Sequential(nn.Conv2d(in_c, c1, kernel_size=1), nn.BatchNorm2d(c1), nn.ReLU())
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2 = nn.Sequential(nn.Conv2d(in_c, c2, kernel_size=1), nn.BatchNorm2d(c2), nn.ReLU(),
                                nn.Conv2d(c2, c2, kernel_size=3), nn.BatchNorm2d(c2), nn.ReLU())
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3 = nn.Sequential(nn.Conv2d(in_c, c3, kernel_size=1), nn.BatchNorm2d(c3), nn.ReLU(),
                                nn.Conv2d(c3, c3, kernel_size=5), nn.BatchNorm2d(c3), nn.ReLU())

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        return p1, p2, p3  # 在通道维上连结输出

class bn_class(nn.Module):
    def __init__(self, input_dim, class_num, fc_feats=False):
        super(bn_class, self).__init__()
        self.bottleneck = nn.BatchNorm1d(input_dim)
        # self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.clss = nn.Linear(input_dim, class_num, bias=False)
        if fc_feats:
            self.clss.apply(weights_init_kaiming)
        else:
            self.clss.apply(weights_init_classifier)

    def forward(self, x):
        x1 = self.bottleneck(x)
        x2 = self.clss(x1)
        return x1, x2

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck,bias = False)] 
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)

        self.add_block1 = add_block1
        self.add_block2 = add_block2

    def forward(self, x):
        x1 = self.add_block1(x)
        x2 = self.add_block2(x1)
        return x1, x2

#ft_net_50_1
class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=False)
        self.feats = 512

        pretrained_state_dict = torch.load("/home/dell/D/dell/fjw/practice/models/resnet50-19c8e357.pth")
        now_state_dict = model_ft.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model_ft.load_state_dict(now_state_dict)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.maxpool_1 = nn.AdaptiveMaxPool2d((1, 1))


        self.avgpool_2 = nn.AvgPool2d((8, 8))
        self.maxpool_2 = nn.MaxPool2d((8, 8))
        
        self.avgpool_3 = nn.AvgPool2d((7, 6))
        self.maxpool_3 = nn.MaxPool2d((7, 6))

        self.avgpool_4 = nn.AvgPool2d((4, 4))
        self.maxpool_4 = nn.MaxPool2d((4, 4))
        
        # self.avgpool_5 = nn.AvgPool2d((2, 2))
        # self.maxpool_5 = nn.MaxPool2d((2, 2))

        self.avgpool_x4 = nn.AvgPool2d((16, 8))
        self.maxpool_x4 = nn.MaxPool2d((16, 8))

        self.inception = Inception(2048, 512, 512, 512)

        self.bn_class1 = bn_class(2048, class_num)
        self.bn_class2_0 = bn_class(512, class_num)
        self.bn_class2_1 = bn_class(512, class_num)
        # self.bn_class2_t = bn_class(512 * 2, self.feats, True)
        self.bn_class3_0 = bn_class(512, class_num)
        self.bn_class3_1 = bn_class(512, class_num)
        # self.bn_class3_t = bn_class(512 * 2, self.feats, True)
        self.bn_class4_0 = bn_class(512, class_num)
        self.bn_class4_1 = bn_class(512, class_num)
        self.bn_class4_2 = bn_class(512, class_num)

        self.bn_class_x4 = bn_class(512, class_num)

        self.bn_class_cat = bn_class(2048+512*3, class_num)

        self.classifier_2 = ClassBlock(512*2, num_bottleneck=512)
        self.classifier_3 = ClassBlock(512*2, num_bottleneck=512)
        self.classifier_4 = ClassBlock(512 * 3, num_bottleneck=512)
        self.classifier_x4 = ClassBlock(1024, num_bottleneck=512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x1 = self.model.maxpool(x)
        x2 = self.model.layer1(x1)
        x3 = self.model.layer2(x2)
        x4 = self.model.layer3(x3)
        x5 = self.model.layer4(x4)

        # inception
        p2, p3, p4 = self.inception(x5)

        # p1
        p1_00 = self.model.avgpool(x5)
        p1_01 = self.maxpool_1(x5)
        p1_1 = torch.squeeze((p1_00+p1_01)/2)
        p1_f, p1_id = self.bn_class1(p1_1)

        # p2
        p2_00 = self.avgpool_2(p2)
        p2_01 = self.maxpool_2(p2)
        p2 = (p2_00+p2_01)/2
        p2_0 = torch.squeeze(p2[:, :, 0:1, :])
        p2_1 = torch.squeeze(p2[:, :, 1:2, :])
        p2_flat = p2.view(p2.size(0), -1)

        _, p2_0_id = self.bn_class2_0(p2_0)
        _, p2_1_id = self.bn_class2_1(p2_1)
        _, p2_f = self.classifier_2(p2_flat)

        # p3
        p3_00 = self.avgpool_3(p3)
        p3_01 = self.maxpool_3(p3)
        p3 = (p3_00 + p3_01) / 2
        p3_0 = torch.squeeze(p3[:, :, 0:1, :])
        p3_1 = torch.squeeze(p3[:, :, 1:2, :])
        p3_flat = p3.view(p3.size(0), -1)

        _, p3_0_id = self.bn_class3_0(p3_0)
        _, p3_1_id = self.bn_class3_1(p3_1)
        _, p3_f = self.classifier_3(p3_flat)

        # p4
        p4_00 = self.avgpool_4(p4)
        p4_01 = self.maxpool_4(p4)
        p4 = (p4_00 + p4_01) / 2
        p4_0 = torch.squeeze(p4[:, :, 0:1, :])
        p4_1 = torch.squeeze(p4[:, :, 1:2, :])
        p4_2 = torch.squeeze(p4[:, :, 2:3, :])
        p4_flat = p4.view(p4.size(0), -1)

        _, p4_0_id = self.bn_class4_0(p4_0)
        _, p4_1_id = self.bn_class4_1(p4_1)
        _, p4_2_id = self.bn_class4_2(p4_2)
        _, p4_f = self.classifier_4(p4_flat)

        # x4
        x4_1 = self.avgpool_x4(x4)
        x4_0 = self.maxpool_x4(x4)
        x4_flat = torch.squeeze((x4_0 + x4_1)/2)
        x4_f0, _, = self.classifier_x4(x4_flat)
        x4_f, x4_id = self.bn_class_x4(x4_f0)

        p_cat = torch.cat([p1_f, p2_f, p3_f, p4_f], dim=1)
        p_cat, p_cat_id = self.bn_class_cat(p_cat)

        features = torch.cat([x4_f, p_cat], dim=1)
        return features, [x4_id, p1_id, p2_0_id, p2_1_id, p3_0_id, p3_1_id, p4_0_id, p4_1_id, p4_2_id, p_cat_id], [x4_f, p1_f, p2_f, p3_f, p4_f, p_cat]
