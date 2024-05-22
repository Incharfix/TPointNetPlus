import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

# normal_channel  是使用法向量

# 三大综合-整合包

# PointNetSetAbstraction
# PointNetSetAbstractionMsg
# PointNetFeaturePropagation



############ 公用的 东西  ########################
#################################################


# 点云归一化：       pc_normalize（）
# 最远点采样：        farthest_point_sample(xyz， Num)   return 那个点是中心点
# 抽离出512点tensor：index_points(points, idx):
# 512点-半径形成512个组 ：query_ball_point（）
#原点与采点的矩阵值：  square_distance（）      #一个距离矩阵：原始所有点到 每个下采样点的距离 [1024行  512列]

# sample_and_group_all

#####################################





def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

# 归一化点云，
# 以radius 为中心的坐标， 球半径为1

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


# # 得到B N M （就是N个点中每一个和M中每一个的欧氏距离）
#一个距离矩阵：原始所有点到 每个下采样点的距离 [1024行  512列]

def square_distance(src, dst): # 名字上得知，求欧式距离的
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist




# 作用： 将 之前有索引的点---抽出来 --- 成一个完整的 tensor格式（1024点 抽出512点-缩影的意思）
#输入： 点，但是带索引。
#输出： 实际得到的点 tensor 格式
#举例： id为[5，566，100，2000]， 在2048个点中 将这四个点(序号的值)抽出来，组成一个新的tensor
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

#最远点才采样 ： 得到的点与点之间足够的远
#返回： npoint个采样点在原始点云中的索引，即： 以后采用的中心点
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) #8*512 的tensor

    distance = torch.ones(B, N).to(device) * 1e10       #距离 8*1024 # 一个空壳  8个batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#第一个点随机，后边的是依据第一个
    batch_indices = torch.arange(B, dtype=torch.long).to(device)


    for i in range(npoint):        #第一个采样点选随机初始化的索引，后边需要for-512次
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)# 得到当前采样点的坐标 B*3
        dist = torch.sum((xyz - centroid) ** 2, -1)            # 计算当前采样点与其他点的距离
        mask = dist < distance                                 # 选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]                            # mask  true 还是flase
        farthest = torch.max(distance, -1)[1]#重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids


# 目标：  以radius为中心， 圈点 成球----成为1组
# 输入： 下采样后的点，进行圈点工作的必要数据
# 返回:  一共有512个点，那圈点就512个组  （# 返回的是索引）

# radius  ： 是半径多大 。 目的：为了得到稳定的特征。
# nsample ： 是指定附近点个数： 512个点采完后， 以指定半径为中心，指定半径范围，范围内的个数，：做均值。。
# xyz     : 原始的所有点
# new_xyz ： 经过下采用后的中心点（512个）
def query_ball_point(radius, nsample, xyz, new_xyz):   # 球状的点
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]  # batch  sample   sample_n_ball
    Return:
        group_idx: grouped points index, [B, S, nsample]  # （8*512）*16  # 512个点 一组内有16个点
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)             # 得到B N M （就是N个点中每一个和M中每一个的欧氏距离）

    group_idx[sqrdists > radius ** 2] = N                # 找到距离大于给定半径的设置成一个N值（1024）索引   # 这个点就不是我的， 我的是0-1023
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]# 当组内点很多，做升序排序，只要最近的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])#当组内点不足，就直接用第一个点复制来代替了。。。
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx



# sample_and_group(本函数)： 将整个点云 分成 多个局部的group # 先下采样， 然后各自点形成各自的group (ball球)
# sample_and_group_all   ： 将已经下采样的点，形成一个组group (ball球)
#

# npoint ： 最远下采样后的点个数
# radius ： 半径
# nsample： 半径内取几个，组成一个组 group

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    # 最远点采样得到序号      ：farthest_point_sample
    # 采完后序号形成新tensor ：index_points
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()

    # 每个采样点形成球group（的点序号） query_ball_point
    # 球内序号转成新tensor               index_points
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()

    # ？ 球内点 - 球中心点 = 中心值？
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()


     # 目的： 用于 cat（点特征数据 + 点坐标数据）
     #        如果每个点有新维度，则拼接
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm



    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points



# sample_and_group            ： 将整个点云 分成 多个局部的group # 先下采样， 然后各自点形成各自的group (ball球)
# sample_and_group_all(本函数) ： 将已经下采样的点，形成一个组group (ball球)
#
#  即 将所有点作为1个 group ，  npoints = 1


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# 目标;  Set  Abstraction
#
# 步骤： 1. sample_and_group 形成每个采样点的 局部的group
# 步骤   2. 对局部group中每个点-做MLP操作，
#         3.  ----> 局部最大池化 --->得到局部的特征

# npoint       128
# radius       0.4
# nsample      64
# in_channel   128+3
# mlp          [128,128,256]
# group_all    FALSE
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)       #  [8,512,3]
        print(xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1)   # [8,640,128]  --->  [8,128,640]  #128点 640个特征
        print(points.shape)


        # 目标 ： sample_and_group
        # [8,1,128,643]  8 batch 1个组 128个点 643个特征  # 643 = 640+3
        # 1次采样的所有点，现在当成一个整体 - 进入神经网络
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

         # 进入 神经网络中
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        print(new_points.shape)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        print(new_points.shape)


        # 目标 ： 对局部的 group 中每一个点最 MLP操作
        #
        # MLP利用 1*1的卷积相当于把灭个 group 当成一个通道，共npoint通道
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        print(new_points.shape)


        # 目标 ：局部最大化，得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]
        print(new_points.shape)
        new_xyz = new_xyz.permute(0, 2, 1)
        print(new_xyz.shape)
        return new_xyz, new_points  # [8,1024,1]   8个batch样本中， 得到1024个特征


# 实现 Msg 方法的 Set Abstraction
#
# 对不同半径的 ball query， 将不同半径下的点云特征保存在 New-point-list中，最后拼接到一起。

# npoint         512
# radius_list    [0.1, 0.2, 0.4]
# nsample_list   [16, 32, 128]
# in_channel     in_channel
# mlp_list       [[32, 32, 64], [64, 64, 128], [64, 96, 128]]

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S] # 不同半径采样
        """
        xyz = xyz.permute(0, 2, 1) #就是坐标点位置特征
        print(xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1) ##就是额外提取的特征，第一次的时候就是那个法向量特征
        print(points.shape)
        B, N, C = xyz.shape
        S = self.npoint


        # 最远点采样得到序号      ：farthest_point_sample
        # 采完后序号形成新tensor ：index_points
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))#采样后的点
        print(new_xyz.shape)


        # 将不同半径下的点云特这个保存在 new_points_list[]
        # for 循环，同一组点，radius半径变了，得到值也不一样
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]                      # 自己定义的值:定值 [16,18,20]

            # 每个采样点形成球group（的点序号） query_ball_point
            # 球内序号转成新tensor               index_points
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)              #去均值 # new_xyz相当于簇的中心点


             # cat（特征数据 + 坐标数据）
            if points is not None:                       # 用512个点组，做特征
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                print(grouped_points.shape)    #【8，512，16，6】 batch  512组 1个组内16点，  每个点6个值
            else:
                grouped_points = grouped_xyz


        #  开始卷积  提特征。
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            print(grouped_points.shape)    #【8，512，16，6】  ----->  [8,6,16,512]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            print(grouped_points.shape)      #[8,64,16,512]  上方的每个点有3个特征，变成 得到64个特征



            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S] 就是pointnet里的maxpool操作
            print(new_points.shape)                 # [8,64,512] 在以前的16维度中， ，只留1个最大值，所以消失了 1个维度
            new_points_list.append(new_points)

        #以上是同一个采样点，有3种半径的组，
        #以下是，将这3个cat
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        print(new_points_concat.shape)  # 320 = 128 + 128 +64
        return new_xyz, new_points_concat



#目标：通过线性差值 + MLP完成

# 当点的个数只有1，采用repeat直接复制N个点
# 当点个数大于1，  采用线性差值的方式进行上采样
# ---> 拼接上下采样对应得SA层特征，再对拼接后的 每一个点做 MLP
# in_channel=1536, mlp=[256, 256]

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        print(xyz1.shape)
        print(xyz2.shape)

        points2 = points2.permute(0, 2, 1)
        print(points2.shape)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape


        # 当点的个数只有1，采用repeat直接复制N个点
        # 当点个数大于1，  采用线性差值的方式进行上采样
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
            print(interpolated_points.shape)
        else:
            dists = square_distance(xyz1, xyz2)
            print(dists.shape)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)       # 距离越远，权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 对每个点的权重在做一个全局的归一化
            print(weight.shape)
            print(index_points(points2, idx).shape)

            #获得差值点
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            print(interpolated_points.shape)



        # 拼接上下采样前对应点  SA层的特征
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        print(new_points.shape)
        new_points = new_points.permute(0, 2, 1)
        print(new_points.shape)

        #拼接后，对每个做MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        print(new_points.shape)
        return new_points

