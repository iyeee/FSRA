from kiwisolver import Variable
import torch
import torch.nn as nn
from models.FSRA import make_transformer_model
from thop import profile

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


class three_view_net(nn.Module):
    def __init__(self, class_num, share_weight = False,block=4,return_f=False):
        super(three_view_net, self).__init__()
        self.share_weight = share_weight
        self.model_1 = make_transformer_model(num_class=class_num, block=block, return_f=return_f)


        if self.share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = make_transformer_model(num_class=class_num,  block=block, return_f=return_f)



    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_2(x2)

        if x3 is None:
            y3 = None
        else:
            y3 = self.model_1(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            y4 = self.model_2(x4)
        return y1, y2, y3, y4


def make_model(opt):
    model_path = "pretrain_model/vit_small_p16_224-15ec54c9.pth"
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block,return_f=opt.triplet_loss)
        # load pretrain param
        model.model_1.transformer.load_param(model_path)

    elif opt.views == 3:
        model = three_view_net(opt.nclasses, share_weight=opt.share,block=opt.block,return_f=opt.triplet_loss)

        # load pretrain param
        model.model_1.transformer.load_param(model_path)
        model.model_2.transformer.load_param(model_path)

    return model


if __name__ == '__main__':


    net = two_view_net(class_num=751,block=3)
    net.cuda()

    input1 =torch.FloatTensor(8, 3, 256, 256).cuda()
    output1,output1 =net(input1,input1)
    # print(output1)

    # modelData = "/data/modanqi/projects/geo_location/model/new_4090_lr001_three_view_long_share_d0.75_256_s1_bz8_160epoch_vanB3/net_139.pth" 
    # torch.onnx.export(net, input1,input1, modelData) 
    # netron.start(modelData) 


    flops,params=profile(net,inputs=(input1,input1,))
    print('flops: ', flops/1000000, 'params: ', params/1000000)
    # print('net output size:')
    # print(output2.shape)
