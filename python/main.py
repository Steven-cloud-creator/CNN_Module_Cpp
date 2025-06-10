from torch import nn
import torch

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        self.out_channels = out_planes
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def ConvBNReLU_Test():
    layers = []
    in_channels = 1
    out_channels = 3
    kernel_size = 3
    stride = 2
    groups = 1
    layers.append(ConvBNReLU(in_channels,out_channels,kernel_size,stride,groups))

    with torch.no_grad():
        layers[0][0].weight.fill_(1.0) #初始化卷积层权重
        # 初始化BatchNorm层的参数
        layers[0][1].weight.fill_(1.0)  # gamma设为1
        layers[0][1].bias.fill_(0.0)    # beta设为0
        layers[0][1].running_mean.fill_(5.4444)  # 均值设为0
        layers[0][1].running_var.fill_(1.3334)   # 方差设为1

    # 创建模型
    model = nn.Sequential(*layers)
    # 创建全1输入张量（批次大小=1，通道=1，尺寸=5x5）
    input_tensor = torch.ones((1, 1, 5, 5), dtype=torch.float32)

    layers[0].eval()
    # 前向传播
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 打印输入输出形状和值
    print("输入形状:", input_tensor.shape)
    print("输入数据（前3x3区域）:\n", input_tensor[0, 0, :5, :5])

    print("\n输出形状:", output_tensor.shape)
    print("输出数据（第一个样本的第一个通道）:\n", output_tensor[0, 0, :, :])
def ExtendedChannel_Test():
    layers = []
    in_channels = 1    # 输入通道数
    out_channels = 3   # 输出通道数（增加通道数）
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
    
    # 初始化卷积层权重
    with torch.no_grad():
        layers[0].weight.fill_(1.0)  # 所有权重初始化为1
    
    # 创建模型
    model = nn.Sequential(*layers)
    model.eval()  # 设置为评估模式
    
    # 创建全1输入张量 (批次大小=1, 通道=1, 尺寸=5x5)
    input_tensor = torch.ones((1, 1, 5, 5), dtype=torch.float32)
    
    # 前向传播
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 打印输入信息
    print("输入形状:", input_tensor.shape)
    print("输入数据:")
    print(input_tensor[0, 0, :, :])  # 打印第一个通道的内容
    
    # 打印输出信息
    print("\n输出形状:", output_tensor.shape)
    print(f"输出有 {out_channels} 个通道，每个通道的内容相同:")
    for i in range(out_channels):
        print(f"\n通道 {i+1}:")
        print(output_tensor[0, i, :, :])
    
    # 验证计算
    print("\n验证计算:")
    print(f"对于1x1卷积且权重全为1，每个输出通道的值等于输入值乘以权重之和")
    print(f"这里输入全为1，权重和为1，所以每个输出值等于 1x1 = 1")
    
    return output_tensor
if __name__ == "__main__":
#测试Conv_BN_ReLU模块
    #ConvBNReLU_Test()
#测试1×1通道数扩展
    ExtendedChannel_Test()

