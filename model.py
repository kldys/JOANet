
import torch
import torch.nn as nn
from config import opt


class SISR(nn.Module):
    def __init__(self):
        super(SISR, self).__init__()
        self.up_factor = opt.scaling_factor

        self.Feature_Encoder1_fe = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder1_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.Feature_Encoder2_fe = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder2_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.Feature_Encoder3_fe = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder3_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.Feature_Encoder4_fe = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.Feature_Encoder4_down = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU())

        self.encoder_end = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())


        self.decoder_up4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.decoder_up3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.decoder_up2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.decoder_up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Feature_Decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1))

        self.Feature_Decoder_end = nn.Conv2d(64, 1, 3, padding=1)

        self.delta_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_5 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_5 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.delta_6 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.eta_6 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.delta_1.data = torch.tensor(0.1)
        self.eta_1.data = torch.tensor(0.9)
        self.delta_2.data = torch.tensor(0.1)
        self.eta_2.data = torch.tensor(0.9)
        self.delta_3.data = torch.tensor(0.1)
        self.eta_3.data = torch.tensor(0.9)
        self.delta_4.data = torch.tensor(0.1)
        self.eta_4.data = torch.tensor(0.9)
        self.delta_5.data = torch.tensor(0.1)
        self.eta_5.data = torch.tensor(0.9)
        self.delta_6.data = torch.tensor(0.1)
        self.eta_6.data = torch.tensor(0.9)

    def forward(self, input):
        x = torch.nn.functional.interpolate(input, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        y = input

        for i in range(6):
            f1 = self.Feature_Encoder1_fe(x)
            down1 = self.Feature_Encoder1_down(f1)

            f2 = self.Feature_Encoder2_fe(down1)
            down2 = self.Feature_Encoder2_down(f2)

            f3 = self.Feature_Encoder3_fe(down2)
            down3 = self.Feature_Encoder1_down(f3)

            f4 = self.Feature_Encoder4_fe(down3)
            down4 = self.Feature_Encoder2_down(f4)

            media_end = self.encoder_end(down4)


            up4 = self.decoder_up4(media_end, output_size=f4.shape)
            concat4 = torch.cat([up4, f4], dim=1)
            decoder4 = self.Feature_Decoder4(concat4)

            up3 = self.decoder_up3(decoder4, output_size=f3.shape)
            concat3 = torch.cat([up3, f3], dim=1)
            decoder3 = self.Feature_Decoder3(concat3)

            up2 = self.decoder_up2(decoder3, output_size=f2.shape)
            concat2 = torch.cat([up2, f2], dim=1)
            decoder2 = self.Feature_Decoder2(concat2)

            up1 = self.decoder_up1(decoder2, output_size=f1.shape)
            concat1 = torch.cat([up1, f1], dim=1)
            decoder1 = self.Feature_Decoder1(concat1)

            conv_out = self.Feature_Decoder_end(decoder1)

            v = conv_out + x

            x = self.reconnect(v, x, y, i)

        return x

    def reconnect(self, v, x, y, i):

        i = i + 1
        if i == 1:
            delta = self.delta_1
            eta = self.eta_1
        if i == 2:
            delta = self.delta_2
            eta = self.eta_2
        if i == 3:
            delta = self.delta_3
            eta = self.eta_3
        if i == 4:
            delta = self.delta_4
            eta = self.eta_4
        if i == 5:
            delta = self.delta_5
            eta = self.eta_5
        if i == 6:
            delta = self.delta_6
            eta = self.eta_6

        down = torch.nn.functional.interpolate(x, scale_factor=1/self.up_factor, mode='bicubic', align_corners=False)
        err1 = torch.nn.functional.interpolate(down-y, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        err2 = x - v
        recon = x - delta*(err1 + eta*err2)

        return recon


if __name__ == '__main__':

    input1 = torch.rand(1, 1, 64, 64)

    net = SISR()
    out = net(input1)
    print(out.size())
