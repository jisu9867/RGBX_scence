import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from engine.logger import get_logger

logger = get_logger()


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion = None, criterion_seg=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                 norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

            print("+++++++++++++use segformer-b2+++++++++++++++")
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            # from .decoders.MLPDecoder import DecoderHead
            # self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes,
            #                                norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
            from .decoders.MLPDecoder import fc_decoder
            self.decode_head = fc_decoder(in_channels=self.channels, num_classes=cfg.num_classes,
                                           norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)

        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer,
                                        channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes,
                                       norm_layer=norm_layer)

        self.criterion = criterion
        self.criterion_seg = criterion_seg
        if self.criterion or self.criterion_seg:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
            # self.init_weights(cfg, pretrained=None)

        self.softmax = nn.Softmax(dim=1)


    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)

        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out
    def encode_decode_seg(self, rgb, pred_vpd, pred_zoe):
        orisize = rgb.shape
        x = self.backbone(rgb)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)

        prob = self.softmax(out)

        # for rgb segmentation
        out_depth = prob[:, 0, :, :] * pred_vpd + prob[:, 1, :, :] * pred_zoe

        # for scence selection
        # sel = prob.argmax(dim=1)
        # if torch.sum(sel == 0) >= torch.sum(sel == 1):
        #     out_depth = pred_vpd
        # else:
        #     out_depth = pred_zoe

        return out_depth, out


    def encode_decode_cls_decoder(self, rgb):
        x = self.backbone(rgb)
        out = self.decode_head.forward(x)
        return out
    def classification_encoder(self, rgb):
        orisize = rgb.shape
        x, logits = self.backbone(rgb)
        # print("logits: ", logits.shape)

        return logits
    def forward(self, rgb, pred_vpd = None, pred_zoe=None,depth = None,label=None):
        # out_depth, out = self.encode_decode_seg(rgb, pred_vpd, pred_zoe)
        orisize = rgb.shape
        # out = self.encode_decode_cls(rgb)
        # out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)  # for pixel segmentation
        # print(out.shape)
        num_classes = 2
        out = self.classification_encoder(rgb)
        if label is not None:
            # loss = self.criterion(out_depth, depth) + self.criterion_seg(out, label.long())
            # loss = self.criterion(out_depth, depth)
            # loss = torch.sqrt(loss)
            # loss = self.criterion_seg(out, label)
            loss = self.criterion_seg(out.view(-1,num_classes), label.long().view(-1))

            return loss
        return out
        # return out_depth, out




    # for hha
    # def forward(self, rgb, hha = None, label=None):
    #     out = self.encode_decode(rgb, hha)
    #     if label is not None:
    #         loss = self.criterion_seg(out, label.long())
    #         return loss
    #     return out
