import torch
import torch.nn.functional as F
from torch import nn

from config import cfg
from loss.triplet_loss import TripletLoss, CrossEntropyLabelSmooth

class Make_loss(nn.Module):
    def __init__(self, cfg, num_classes):
        super(Make_loss, self).__init__()
        self.triplet = TripletLoss(cfg.SOLVER.MARGIN)
        self.num_classes = num_classes

    def forward(self, logits, train_f, labels):
        loss_tpl = []
        for i in range(len(train_f)):
            loss_tpl.append(self.triplet(train_f[i], labels)[0])

        loss_cls = []
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            loss_label_smooth = CrossEntropyLabelSmooth(num_classes=self.num_classes)
            for i in range(len(logits)): #i为每个scale
                loss_cls.append(loss_label_smooth(logits[i], labels))      
        else:
            for i in range(len(logits)): #i为每个scale
                loss_cls.append(F.cross_entropy(logits[i], labels))

        loss_x4 = 0.2 * (loss_cls[0] + loss_tpl[0])
        loss_p1 = 0.25 * (loss_cls[1] + loss_tpl[1])
        loss_p2 = 0.25 * (sum(loss_cls[2:4]) / 2 + loss_tpl[2])
        loss_p3 = 0.25 * (sum(loss_cls[4:6]) / 2 + loss_tpl[3])
        loss_p4 = 0.25 * (sum(loss_cls[6:9]) / 3 + loss_tpl[4])
        loss_cat = 0.3 * (loss_cls[9] + loss_tpl[5])
        total_loss = loss_x4 + loss_p1 + loss_p2 + loss_p3 + loss_p4 + loss_cat

        return total_loss
