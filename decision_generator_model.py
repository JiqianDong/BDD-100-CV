from imports import *


class MHSA(nn.Module):
    def __init__(self,
            emb_dim,
            kqv_dim,
            num_heads=1):
        super(MHSA, self).__init__()
        self.emb_dim = emb_dim
        self.kqv_dim = kqv_dim
        self.num_heads = num_heads

        self.w_k = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_q = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_v = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_out = nn.Linear(kqv_dim * num_heads, emb_dim)

    def forward(self, x):

        b, t, _ = x.shape
        e = self.kqv_dim
        h = self.num_heads
        keys = self.w_k(x).view(b, t, h, e)
        values = self.w_v(x).view(b, t, h, e)
        queries = self.w_q(x).view(b, t, h, e)

        keys = keys.transpose(2, 1)
        queries = queries.transpose(2, 1)
        values = values.transpose(2, 1)

        dot = queries @ keys.transpose(3, 2)
        dot = dot / np.sqrt(e)
        dot = nn.functional.softmax(dot, dim=3)

        out = dot @ values
        out = out.transpose(1,2).contiguous().view(b, t, h * e)
        out = self.w_out(out)
        return out

class MHSA2(nn.Module):
    def __init__(self,
            emb_dim,
            kqv_dim,
            output_dim=10,
            num_heads=8):
        super(MHSA2, self).__init__()
        self.emb_dim = emb_dim
        self.kqv_dim = kqv_dim
        self.num_heads = num_heads

        self.w_k = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_q = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_v = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
        self.w_out = nn.Linear(kqv_dim * num_heads, output_dim)

    def forward(self, x):

        b, t, _ = x.shape
        e = self.kqv_dim
        h = self.num_heads
        keys = self.w_k(x).view(b, t, h, e)
        values = self.w_v(x).view(b, t, h, e)
        queries = self.w_q(x).view(b, t, h, e)

        keys = keys.transpose(2, 1)
        queries = queries.transpose(2, 1)
        values = values.transpose(2, 1)

        dot = queries @ keys.transpose(3, 2)
        dot = dot / np.sqrt(e)
        dot = nn.functional.softmax(dot, dim=3)

        out = dot @ values
        out = out.transpose(1,2).contiguous().view(b, t, h * e)
        out = self.w_out(out)
        return out


class DecisionGenerator(nn.Module):
    def __init__(self,faster_rcnn_model,device,batch_size,select_k=2,action_num=4,explanation_num=21,freeze_rcnn=True):
        super().__init__()

        self.rcnn = faster_rcnn_model
        self.batch_size = batch_size

        if freeze_rcnn:
            for param in self.rcnn.parameters():
                param.requires_grad = False
                self.rcnn.eval()
        self.object_attention = MHSA(1024, kqv_dim=10, num_heads=8)

        self.roi_pooling_conv = nn.Conv1d(in_channels=1000,out_channels=select_k,kernel_size=1)

        self.action_branch = nn.Sequential(
                                nn.Linear(select_k*1024, 1024),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(1024, action_num))

        self.explanation_branch = nn.Sequential(
                                nn.Linear(select_k*1024, 1024),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(1024, explanation_num))


        self.action_loss_fn, self.reason_loss_fn = self.loss_fn(device)

    def loss_fn(self,device):
        class_weights = [1, 1, 2, 2]
        w = torch.FloatTensor(class_weights).to(device)
        action_loss = nn.BCEWithLogitsLoss(pos_weight=w).to(device)
        explanation_loss = nn.BCEWithLogitsLoss().to(device)
        return action_loss,explanation_loss


    def forward(self,images,targets=None):
        if self.training:
            assert targets is not None
            target_reasons = torch.stack([t['reason'] for t in targets])
            target_actions = torch.stack([t['action'] for t in targets])

        with torch.no_grad():
            self.rcnn.eval()
            batch_size = len(images)
            images,_ = self.rcnn.transform(images)
            features = self.rcnn.backbone(images.tensors)
            proposals, _ = self.rcnn.rpn(images, features)

            box_features = self.rcnn.roi_heads.box_roi_pool(features,proposals,images.image_sizes)
            box_features = self.rcnn.roi_heads.box_head(box_features).view(batch_size, -1, 1024)  #(B, num_proposal, 1024)
        
        box_features = self.object_attention(box_features) #(B, num_proposal, 1024)

        # feature_polled,_ = torch.max(box_features,1)
        feature_polled = self.roi_pooling_conv(box_features)
        # print(feature_polled.shape)
        feature_polled = torch.flatten(feature_polled,start_dim=1)
        # print(feature_polled.shape)

        actions = self.action_branch(feature_polled)
        reasons = self.explanation_branch(feature_polled)

        if self.training:
            action_loss = self.action_loss_fn(actions, target_actions)
            reason_loss = self.reason_loss_fn(reasons, target_reasons)
            loss_dic = {"action_loss":action_loss, "reason_loss":reason_loss}
            return loss_dic
        else:
            return {"action":torch.sigmoid(actions),"reasons":torch.sigmoid(reasons)}


class DecisionGenerator_v1(nn.Module):
    def __init__(self,faster_rcnn_model,device,batch_size,action_num=4,explanation_num=21,freeze_rcnn=True):
        super().__init__()

        self.rcnn = faster_rcnn_model
        self.batch_size = batch_size

        if freeze_rcnn:
            for param in self.rcnn.parameters():
                param.requires_grad = False
                self.rcnn.eval()
        self.object_attention = MHSA(1024, kqv_dim=10, num_heads=8)
        self.action_branch = nn.Linear(1024,action_num)

        self.explanation_branch = nn.Linear(1024, explanation_num)
        self.action_loss_fn, self.reason_loss_fn = self.loss_fn(device)

    def loss_fn(self,device):
        class_weights = [1, 1, 2, 2]
        w = torch.FloatTensor(class_weights).to(device)
        action_loss = nn.BCEWithLogitsLoss(pos_weight=w).to(device)
        explanation_loss = nn.BCEWithLogitsLoss().to(device)
        return action_loss,explanation_loss


    def forward(self,images,targets=None):
        if self.training:
            target_reasons = torch.stack([t['reason'] for t in targets])
            target_actions = torch.stack([t['action'] for t in targets])

        with torch.no_grad():
            self.rcnn.eval()
            batch_size = len(images)
            images,_ = self.rcnn.transform(images)
            features = self.rcnn.backbone(images.tensors)
            proposals, _ = self.rcnn.rpn(images, features)

            box_features = self.rcnn.roi_heads.box_roi_pool(features,proposals,images.image_sizes)
            box_features = self.rcnn.roi_heads.box_head(box_features).view(batch_size, -1, 1024)  #(B, num_proposal, 1024)
        
        box_features = self.object_attention(box_features) #(B, num_proposal, 1024)
        feature_polled,_ = torch.max(box_features,1)

        actions = self.action_branch(feature_polled)
        reasons = self.explanation_branch(feature_polled)
        if self.training:
            action_loss = self.action_loss_fn(actions, target_actions)
            reason_loss = self.reason_loss_fn(reasons, target_reasons)

            loss_dic = {"action_loss":action_loss, "reason_loss":reason_loss}
            return loss_dic
        else:
            return {"action":torch.sigmoid(actions),"reasons":torch.sigmoid(reasons)}


class DecisionGenerator_v3(nn.Module): # attention with only one layer
    def __init__(self,faster_rcnn_model,device,batch_size,select_k=10,action_num=4,explanation_num=21,freeze_rcnn=True):
        super().__init__()

        self.rcnn = faster_rcnn_model
        self.batch_size = batch_size
        self.select_k = select_k

        if freeze_rcnn:
            for param in self.rcnn.parameters():
                param.requires_grad = False
                self.rcnn.eval()

        self.attention_score = nn.Sequential(nn.Linear(1024,512),
                                             nn.ReLU(),
                                             nn.Linear(512,1),
                                             nn.Softmax(dim=1))

        self.roi_pooling_conv = nn.Conv1d(in_channels=1000,out_channels=select_k,kernel_size=1)

        self.action_branch = nn.Sequential(
                                nn.Linear(select_k*1024, 1024),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(1024, action_num))

        self.explanation_branch = nn.Sequential(
                                nn.Linear(select_k*1024, 1024),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(1024, explanation_num))


        self.action_loss_fn, self.reason_loss_fn = self.loss_fn(device)

    def loss_fn(self,device):
        class_weights = [1, 1, 2, 2]
        w = torch.FloatTensor(class_weights).to(device)
        action_loss = nn.BCEWithLogitsLoss(pos_weight=w).to(device)
        explanation_loss = nn.BCEWithLogitsLoss().to(device)
        return action_loss,explanation_loss


    def forward(self,images,targets=None):
        if self.training:
            assert targets is not None
            target_reasons = torch.stack([t['reason'] for t in targets])
            target_actions = torch.stack([t['action'] for t in targets])

        with torch.no_grad():
            self.rcnn.eval()
            batch_size = len(images)
            images,_ = self.rcnn.transform(images)
            features = self.rcnn.backbone(images.tensors)
            proposals, _ = self.rcnn.rpn(images, features)

            box_features = self.rcnn.roi_heads.box_roi_pool(features,proposals,images.image_sizes)
            box_features = self.rcnn.roi_heads.box_head(box_features).view(batch_size, -1, 1024)  #(B, num_proposal, 1024)
        
        score = self.attention_score(box_features) #(B, num_proposal, 1024)
        _,ind = torch.topk(score,k=self.select_k,dim=1)
        ### cnn for dimensional reduction
        # box_features = box_features * score
        # feature_polled = self.roi_pooling_conv(box_features)

        feature_polled = torch.gather(box_features,1,ind.expand(ind.size(0),ind.size(1),box_features.size(2))) #select_top_k

        feature_polled = torch.flatten(feature_polled,start_dim=1)

        actions = self.action_branch(feature_polled)
        reasons = self.explanation_branch(feature_polled)

        if self.training:
            action_loss = self.action_loss_fn(actions, target_actions)
            reason_loss = self.reason_loss_fn(reasons, target_reasons)
            loss_dic = {"action_loss":action_loss, "reason_loss":reason_loss}
            return loss_dic
        else:
            return {"action":torch.sigmoid(actions),"reasons":torch.sigmoid(reasons)}


class DecisionGenerator_v4(nn.Module):
    def __init__(self,faster_rcnn_model,device,batch_size,select_k=2,action_num=4,explanation_num=21,freeze_rcnn=True):
        super().__init__()

        self.rcnn = faster_rcnn_model
        self.batch_size = batch_size

        if freeze_rcnn:
            for param in self.rcnn.parameters():
                param.requires_grad = False
                self.rcnn.eval()
        self.object_attention = MHSA2(1024, kqv_dim=10, num_heads=8, output_dim=10)

        self.action_branch = nn.Sequential(
                                nn.Linear(10000, 1024),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(1024, action_num))

        self.explanation_branch = nn.Sequential(
                                nn.Linear(10000, 1024),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(1024, explanation_num))


        self.action_loss_fn, self.reason_loss_fn = self.loss_fn(device)

    def loss_fn(self,device):
        class_weights = [1, 1, 2, 2]
        w = torch.FloatTensor(class_weights).to(device)
        action_loss = nn.BCEWithLogitsLoss(pos_weight=w).to(device)
        explanation_loss = nn.BCEWithLogitsLoss().to(device)
        return action_loss,explanation_loss


    def forward(self,images,targets=None):
        if self.training:
            assert targets is not None
            target_reasons = torch.stack([t['reason'] for t in targets])
            target_actions = torch.stack([t['action'] for t in targets])

        with torch.no_grad():
            self.rcnn.eval()
            batch_size = len(images)
            images,_ = self.rcnn.transform(images)
            features = self.rcnn.backbone(images.tensors)
            proposals, _ = self.rcnn.rpn(images, features)

            box_features = self.rcnn.roi_heads.box_roi_pool(features,proposals,images.image_sizes)
            box_features = self.rcnn.roi_heads.box_head(box_features).view(batch_size, -1, 1024)  #(B, num_proposal, 1024)
        
        box_features = self.object_attention(box_features) #(B, num_proposal, 10)

        # feature_polled,_ = torch.max(box_features,1)
        # print(feature_polled.shape)
        feature_polled = torch.flatten(box_features,start_dim=1)
        # print(feature_polled.shape)

        actions = self.action_branch(feature_polled)
        reasons = self.explanation_branch(feature_polled)

        if self.training:
            action_loss = self.action_loss_fn(actions, target_actions)
            reason_loss = self.reason_loss_fn(reasons, target_reasons)
            loss_dic = {"action_loss":action_loss, "reason_loss":reason_loss}
            return loss_dic
        else:
            return {"action":torch.sigmoid(actions),"reasons":torch.sigmoid(reasons)}



class DecisionGenerator_whole_attention(nn.Module):
    def __init__(self, encoder, encoder_dims, device, num_heads=8, \
                 attention_out_dim=10, action_num=4, explanation_num=21):
        super().__init__()

        """
        encoder_dims = (F,H,W) 
            F:Feature shape (1280 for mobile net, 2048 for resnet)
            H,W = image feature height, width 
        """
        self.encoder = encoder

        assert len(encoder_dims) == 3, "encoder_dims has to be a triplet with shape (F,H,W)" 

        F,H,W = encoder_dims

        self.MHSA = MHSA2(emb_dim=F,kqv_dim=10,output_dim=attention_out_dim,num_heads=num_heads)

        T = H*W
        self.action_branch = nn.Sequential(
                                nn.Linear(attention_out_dim*T,64),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(64,action_num))

        self.explanation_branch = nn.Sequential(
                                nn.Linear(attention_out_dim*T,64),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(64, explanation_num))

        self.action_loss_fn, self.reason_loss_fn = self.loss_fn(device)

    def loss_fn(self,device):
        class_weights = [1, 1, 2, 2]
        w = torch.FloatTensor(class_weights).to(device)
        action_loss = nn.BCEWithLogitsLoss(pos_weight=w).to(device)
        explanation_loss = nn.BCEWithLogitsLoss().to(device)
        return action_loss,explanation_loss


    def forward(self,images,targets=None):
        images = torch.stack(images)
        if self.training:
            assert targets is not None
            target_reasons = torch.stack([t['reason'] for t in targets])
            target_actions = torch.stack([t['action'] for t in targets])
        # print(images.shape)
        features = self.encoder(images) # 
        # print(features.shape)

        B,F,H,W = features.shape

        # print(features.view(B,F,H*W).transpose(1,2).shape)
        
        transformed_feature = self.MHSA(features.view(B,F,H*W).transpose(1,2)) #(B, H, T, 10)
        # print(transformed_feature.shape)
        feature_polled = torch.flatten(transformed_feature,start_dim=1)

        # print(feature_polled.shape)

        actions = self.action_branch(feature_polled)
        reasons = self.explanation_branch(feature_polled)

        if self.training:
            action_loss = self.action_loss_fn(actions, target_actions)
            reason_loss = self.reason_loss_fn(reasons, target_reasons)
            loss_dic = {"action_loss":action_loss, "reason_loss":reason_loss}
            return loss_dic
        else:
            return {"action":torch.sigmoid(actions),"reasons":torch.sigmoid(reasons)}


class DecisionGenerator_no_attention(nn.Module):
    def __init__(self, encoder, encoder_dims, device, action_num=4, explanation_num=21):
        super().__init__()

        """
        encoder_dims = (F,H,W) 
            F:Feature shape (1280 for mobile net, 2048 for resnet)
            H,W = image feature height, width 
        """
        self.encoder = encoder

        assert len(encoder_dims) == 3, "encoder_dims has to be a triplet with shape (F,H,W)" 

        F,H,W = encoder_dims
        in_dim = H*W*F
        self.action_branch = nn.Sequential(
                                nn.Linear(in_dim,12),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(12,action_num))

        self.explanation_branch = nn.Sequential(
                                nn.Linear(in_dim,12),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Linear(12, explanation_num))

        self.action_loss_fn, self.reason_loss_fn = self.loss_fn(device)

    def loss_fn(self,device):
        class_weights = [1, 1, 2, 2]
        w = torch.FloatTensor(class_weights).to(device)
        action_loss = nn.BCEWithLogitsLoss(pos_weight=w).to(device)
        explanation_loss = nn.BCEWithLogitsLoss().to(device)
        return action_loss,explanation_loss


    def forward(self,images,targets=None):
        images = torch.stack(images)
        if self.training:
            assert targets is not None
            target_reasons = torch.stack([t['reason'] for t in targets])
            target_actions = torch.stack([t['action'] for t in targets])
        # print(images.shape)
        features = self.encoder(images) # 
        # print(features.shape)

        B,F,H,W = features.shape

        # print(features.view(B,F,H*W).transpose(1,2).shape)
        # print(transformed_feature.shape)
        feature_polled = torch.flatten(features,start_dim=1)
        # print(feature_polled.shape)

        # print(feature_polled.shape)

        actions = self.action_branch(feature_polled)
        reasons = self.explanation_branch(feature_polled)

        if self.training:
            action_loss = self.action_loss_fn(actions, target_actions)
            reason_loss = self.reason_loss_fn(reasons, target_reasons)
            loss_dic = {"action_loss":action_loss, "reason_loss":reason_loss}
            return loss_dic
        else:
            return {"action":torch.sigmoid(actions),"reasons":torch.sigmoid(reasons)}