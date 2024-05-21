import torch
from torch import nn
from torch.functional import F
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv
from .proposal_conv import build_proposal_conv_fusion



class MMN(nn.Module):
    def __init__(self, cfg):
        super(MMN, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.proposal_conv_fusion = build_proposal_conv_fusion(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.MMN.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.MMN.TEXT_ENCODER.NAME

        self.Linear_1 = nn.Linear(512,128)
        self.Linear_2 = nn.Linear(512,128)

    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # backbone
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]
        feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        # map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        sent_feat, sent_feat_iou ,sent_feat_iou_fusion= self.text_encoder(batches.queries, batches.wordlens)
        # import pdb;pdb.set_trace()
        
        B, D, N = feats.shape
        x1 = feats.view(B*D, N, 1)
        x2 = feats.view(B*D, 1, N)
        MAP = torch.bmm(x1, x2)
        map2d = MAP.view(B, D, N, N)
        map2d, map2d_iou = self.proposal_conv(map2d)

        iou_fusion_scores = []
        for batch_tag, quries_single_video in enumerate(sent_feat_iou_fusion):
            # import pdb;pdb.set_trace()
            # 如何减少参数量
            num_sent = quries_single_video.shape[0]

            single_video_broadcast = feats[batch_tag].reshape(1,D,N).repeat(num_sent,1,1).permute(2,0,1)
            # [N,num_sent,D]
            single_video_broadcast = self.Linear_1(single_video_broadcast.reshape(-1,D)).reshape(N,num_sent,-1)
            quries_single_video = self.Linear_2(quries_single_video.reshape(-1,D)).reshape(num_sent,-1)
            
            Dimension = 128
            
            # [N,num_sent,D]
            fusion_feature = torch.mul(single_video_broadcast,quries_single_video)
            fusion_x1 = fusion_feature.permute(1,2,0).view(num_sent*Dimension,N,1)
            fusion_x2 = fusion_feature.permute(1,2,0).view(num_sent*Dimension,1,N)
            fusion_map2d = torch.bmm(fusion_x1, fusion_x2).reshape(num_sent, Dimension, N, N)
            # import pdb;pdb.set_trace()
            _, map2d_iou_fusion = self.proposal_conv_fusion(fusion_map2d)
            map2d_iou_fusion = map2d_iou_fusion.reshape(num_sent, N, N)
            iou_fusion_scores.append((map2d_iou_fusion*10).sigmoid() * self.feat2d.mask2d)

            
        

        # inference
        contrastive_scores = []
        iou_scores = []
        _, T, _ = map2d[0].size()
        for i, sf_iou in enumerate(sent_feat_iou):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            vid_feat_iou = map2d_iou[i]  # C x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou_norm = F.normalize(sf_iou, dim=1)
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)

        # loss
        if self.training:
            # import pdb;pdb.set_trace()
            loss_iou = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_iou_fusion = self.iou_score_loss(torch.cat(iou_fusion_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_vid, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d, batches.moments)
            return loss_vid, loss_sent, loss_iou, loss_iou_fusion
        else:
            for i, sf in enumerate(sent_feat):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores, iou_fusion_scores  # first two maps for visualization
