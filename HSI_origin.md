```
class Lr2Delta(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, n_res=3, useSoftmax=True):
        super(Lr2Delta, self).__init__()

        # Initial convolution
        self.conv0 = nn.Conv2d(input_c, ngf, kernel_size=3, stride=1, padding=1, bias=False)

        # Dense blocks with residual connections
        self.dense_block1 = self._make_dense_block(ngf, ngf * 2, n_res)
        self.transition1 = self._make_transition_block(ngf * 2, ngf * 4)

        self.dense_block2 = self._make_dense_block(ngf * 4, ngf * 4, n_res)
        self.transition2 = self._make_transition_block(ngf * 4, ngf * 8)

        self.dense_block3 = self._make_dense_block(ngf * 8, ngf * 8, n_res)

        # Final convolutional layer
        self.conv_final = nn.Conv2d(ngf * 8, output_c, kernel_size=1, stride=1, bias=False)

        self.usesoftmax = useSoftmax

    def _make_dense_block(self, in_channels, out_channels, n_res):
        layers = []
        for i in range(n_res):
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            in_channels += out_channels
        return nn.Sequential(*layers)

    def _make_transition_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.dense_block1(out)
        out = self.transition1(out)
        out = self.dense_block2(out)
        out = self.transition2(out)
        out = self.dense_block3(out)
        out = F.relu(out, inplace=True)
        out = self.conv_final(out)

        if self.usesoftmax:
            out = F.softmax(out, dim=1)

        return out

```