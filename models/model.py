import torch.nn as nn
from .FSRA import make_transformer_model


class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False):
        super(two_view_net, self).__init__()
        self.model_1 = make_transformer_model(num_class=class_num, block=block,return_f=return_f)


    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_1(x2)
        return y1, y2





def make_model(opt):
    model_path = "pretrain_model/vit_small_p16_224-15ec54c9.pth"                 #预训练地址
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block,return_f=opt.triplet_loss)
        # load pretrain param
        model.model_1.transformer.load_param(model_path)

    return model

