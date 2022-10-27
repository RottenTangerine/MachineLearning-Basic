import torch.nn as nn
import torchvision.models as models


class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        base_model = models.vgg16()
        self.vgg = nn.Sequential(*(list(base_model.features)[:-1]))
        self.rpn = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True)
        )
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.ReLU(inplace=True)
        )

        self.cls_score = nn.Conv2d(512, 2 * 10, 1)
        self.coordinate = nn.Conv2d(512, 2 * 10, 1)

    def forward(self, x):
        x = self.vgg(x)
        x = self.rpn(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        x, _ = self.brnn(x)
        x = x.view(b[0], x.size(0), x.size(1), 256)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.fc(x)

        cls_score = self.cls_score(x)
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous()
        cls_score = cls_score.view(cls_score.size(0), cls_score.size(1) * cls_score.size(2) * 10, 2)

        vertical_pred = self.coordinate(x)
        vertical_pred = vertical_pred.permute(0, 2, 3, 1).contiguous()
        vertical_pred = vertical_pred.view(vertical_pred.size(0), vertical_pred.size(1) * vertical_pred.size(2) * 10, 2)

        return cls_score, vertical_pred


if __name__ == '__main__':
    import torch
    from icecream import ic

    t = torch.rand(1, 3, 700, 700)
    ctpn = CTPN()
    output = ctpn(t)
    ic(output[0].shape, output[1].shape)
