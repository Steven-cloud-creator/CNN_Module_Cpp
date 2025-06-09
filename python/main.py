from torch import nn
import torch

if __name__ == "__main__":
    # 创建卷积层
    features = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3, stride=2, padding=1,groups=1,bias=False),
                             nn.BatchNorm2d(3),
                             nn.ReLU())
    # 初始化权重为常数（方便测试）
    with torch.no_grad():
        features[0].weight.fill_(1.0)  # 初始化卷积层权重
        # 初始化BatchNorm层的参数
        features[1].weight.fill_(1.0)  # gamma设为1
        features[1].bias.fill_(0.0)    # beta设为0
        features[1].running_mean.fill_(5.4444)  # 均值设为0
        features[1].running_var.fill_(1.3334)   # 方差设为1
    
    # 创建全1输入张量（批次大小=1，通道=1，尺寸=5x5）
    input_tensor = torch.ones((1, 1, 5, 5), dtype=torch.float32)

    features.eval()
    # 前向传播
    output_tensor = features(input_tensor)

    # 打印输入输出形状和值
    print("输入形状:", input_tensor.shape)
    print("输入数据（前3x3区域）:\n", input_tensor[0, 0, :5, :5])

    print("\n输出形状:", output_tensor.shape)
    print("输出数据（第一个样本的第一个通道）:\n", output_tensor[0, 0, :, :])


