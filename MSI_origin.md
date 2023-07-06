```
class Msi2Delta(nn.Module):
    # 本身就是默认的
    def __init__(self, input_c, output_c, ngf=64, n_res=3, useSoftmax=True):
        super(Msi2Delta, self).__init__()

        # Initial Convolution Layer

        self.init_conv = nn.Conv2d(input_c, ngf * 2, 1, 1, 0)

        # Dense Blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 4)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf * 4 + ngf * 2, ngf * 8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 8)
        )

        # Final Convolution Layer
        self.final_conv = nn.Conv2d(ngf * 8 + ngf * 4 + ngf * 2, output_c, 1, 1, 0)

        self.softmax = nn.Softmax(dim=1)
        self.usefostmax = useSoftmax

    def forward(self, x):
        # Initial Convolution Layer
        out = self.init_conv(x)

        # Dense Blocks
        out_block1 = self.block1(out)
        # 第二个卷积的部分等于第一个卷积的部分相加
        out_block2 = self.block2(torch.cat((out_block1, out), dim=1))
        # 然后这是第二个连接块
        out = torch.cat((out_block2, out_block1, out), dim=1)

        # 因为最后的卷积部分是直接把所有的相加了，所以这里是final_conv
        # Final Convolution Layer
        out = self.final_conv(out)

        if self.usefostmax == True:
            return self.softmax(out)
        elif self.usefostmax == False:
            return out
```