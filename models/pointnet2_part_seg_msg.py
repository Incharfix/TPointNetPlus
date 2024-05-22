import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation

from models_sn.utils import Transformer


# 主要的 详细讲文件
#核心： 多尔半径组， 组成一个特征向量



class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        # 主题： 多半径的特征提取
        #    找中心，画圈，
        #   512                 选择512个中心点
        #   [0.1, 0.2, 0.4]     以中心点为目标，圈多少个点
        #   [32, 64, 128]       每组的个数分别有多少
        ## [[32, 32, 64], ，，， 指的是每次卷积后的channel，这里经历了三次卷积

        #备注: 128+128+6 是上一层输入的channel
        #      sa3 没有使用多半径的写法
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.2, 0.3, 0.4], [64, 64, 128], 3 + additional_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512,
                                             [0.4, 0.8],
                                             [64, 128],
                                             128 + 128 + 64,
                                             [[128, 128, 256], [128, 196, 256]])

        self.sa3 = PointNetSetAbstractionMsg(128,       # 最后输出的采样点数量
                                             [0.4, 0.8],
                                             [64, 128],
                                             512,  #输入的通道数
                                             [[128, 128, 256], [128, 196, 256], [128, 256, 512]])

        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3,
                                          mlp=[256, 512, 1024], group_all=True)


        self.transformer_1 = Transformer(320, dim=64)
        self.transformer_2 = Transformer(512, dim=64)


        self.fp4 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=448, mlp=[256, 128])
        #self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=138, mlp=[128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction laye

        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

                 # #320 =128+128+64
        # sa2  input   output
        # sa3  input   output
        # new_xyz, new_points_concat

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #320 =128+128+64
        l1_xyz = l1_xyz.contiguous()
        l1_points = l1_points.contiguous()
        l1_points = self.transformer_1(l1_points, l1_xyz)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # 640 = 256 + 256 + 128 ？
        l2_xyz = l2_xyz.contiguous()
        # l2_points = self.transformer_2(l2_points, l2_xyz)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) #

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) #


        # Feature Propagation layers
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) #

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) #
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)


        cls_label_one_hot = cls_label.view(B,1,1).repeat(1,1,N)  # 原先 num_classes = 16 (B,16,1).
        ttemp = torch.cat([cls_label_one_hot,l0_xyz,l0_points],1)
        l0_points = self.fp1(l0_xyz, l1_xyz,ttemp , l1_points)

       #  [1, 256, 128] = fp3 [1, 3, 128], [1, 3, 1],, [1, 512, 128], [1, 1024, 1]
       #  [1, 128, 512] =fp2  [1, 3, 512], [1, 3, 128], [1, 320, 512], [1, 256, 128]
       #  [1, 128, 1024] = fp1 [1, 3, 2048], [1, 3, 512], XXX，[1, 320, 512]

  #fp3 执行：
   # l3_points ：[1, 128，1024]  =   [1, 1024, 1]  循环128次补齐，
   #       cat（ l2_points变更，l3_points变更 ） =  [1,128,1536]
   #     conv1（1536，256 ，1*1）     conv2（256，256 ，1*1）

# fp2 执行：
        # l3_points ：[1, 512, 256] =    [1, 256, 128]  多次变换
        #       cat（ l2_points变更，l3_points变更 ） =   [1,512,576]  576 = 320 + 256
        #     conv1（576，256 ，1*1）     conv2（256，128 ，1*1）


        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points   # [1,2048,50]  [1,1024,1]


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss