from imports import *

class ReasonDecoder(nn.Module):
    def __init__(self, image_f_dim, embedding_dim, hidden_dim, dict_size, device='cpu', null_index=0, start_index = 3, end_index=2,using_gate=True):
        super().__init__()
        self._NULL_INDEX = null_index
        self._START_INDEX = start_index
        self._DICT_SIZE = dict_size
        self._END_INDEX = end_index

        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim
        self._image_f_dim = image_f_dim
        
        self.using_gate = using_gate
        self.device = device
        
        self.embedding_layer = nn.Embedding(num_embeddings=self._DICT_SIZE,embedding_dim=self._embedding_dim)
        self.image_affine_layers = nn.Sequential(nn.Linear(self._image_f_dim,self._hidden_dim),nn.ReLU()) 
        self.init_c_layer = nn.Linear(self._hidden_dim,self._hidden_dim)
        self.init_h_layer = nn.Linear(self._hidden_dim,self._hidden_dim)
        self.lstm_step = nn.LSTMCell(input_size=hidden_dim*2,hidden_size=hidden_dim)
        self.score_layer = nn.Linear(hidden_dim, self._DICT_SIZE)      
        if self.using_gate:
            self.gate = nn.Sequential(nn.Linear(self._hidden_dim, self._hidden_dim), nn.Sigmoid())
        self.loss = self.loss_fn()

    def loss_fn(self):
        reason_loss = nn.CrossEntropyLoss().to(self.device)
        return reason_loss

    def dot_product_attention(self, prev_h, A):
        """
        dot product between the hidden state and image embeddings
        A: B,F,H,W
        prev_h: B,F
        
        (F: hidden dimension)
        """
        B,F,H,W = A.shape
        attention_score = torch.bmm(prev_h.view(B,1,F),A.view(B,F,H*W)).squeeze(1).div(F**0.5) # B, H*W
        attention_weights = nn.functional.softmax(attention_score,dim=1) # B, H*W
        attended_features = torch.bmm(A.view(B,F,H*W), attention_weights.view(B,H*W,1)).squeeze(2) # B, F
        attention_weights = attention_weights.view(B,H,W)

        return attention_weights, attended_features

    def init_hidden_state(self, A):
        mean_image = A.mean(dim=(2,3)) # N,image_feature_dim
        h0 = self.init_h_layer(mean_image)
        c0 = self.init_c_layer(mean_image)
        return h0, c0   

    def forward(self, image_feature, reason_batch):

        image_f = self.image_affine_layers(image_feature.permute(0,2,3,1)).permute(0,3,1,2) # B , F , H , W 

        ## Manipulate the reason batch
        padded = nn.utils.rnn.pad_sequence(reason_batch,padding_value=self._NULL_INDEX)
        embedded = self.embedding_layer(padded)
        embedded_input = embedded[:-1,:,:]
        padded_output = padded[1:,:]
        T,B,E = embedded_input.shape
        
        h,c = self.init_hidden_state(image_f)
        
        hs = []
        scores = []
        attention_weights = []
        
        for t in range(T):
            attention_weight, attended_features = self.dot_product_attention(h,image_f)
            # print(attended_feature.shape)
            if self.using_gate:
                gate_val = self.gate(h)
                # print(gate_val.shape)
                attended_features = attended_features*gate_val
            
            lstm_input = torch.cat([attended_features,embedded_input[t]],axis=1)
            h,c = self.lstm_step(lstm_input,(h,c))
            score = self.score_layer(h)
            hs.append(h)
            scores.append(score)
            attention_weights.append(attention_weight)
        hs, scores, attention_weights = torch.stack(hs), torch.stack(scores) ,torch.stack(attention_weights) 

        if self.training:
            loss = self.loss(scores.view(-1,self._DICT_SIZE), padded_output.view(-1))
            return loss,scores, attention_weights, hs
        else:
            return scores, attention_weights, hs #hs: T B F, attention weights: T, B, H, W 
    
    def generate_reason(self, image_feature, max_length=33):
        image_f = self.image_affine_layers(image_feature.permute(0,2,3,1)).permute(0,3,1,2) # B , F , H , W 

        B = image_f.shape[0]
        
        prev_word = torch.LongTensor([self._START_INDEX]*B).to(self.device)

        h0,c0 = self.init_hidden_state(image_f)
        h = h0
        c = c0
        hs = []
        scores = []
        attention_weights = []
        reasons = []
        
        for t in range(max_length):
            prev_embedding = self.embedding_layer(prev_word)
            attention_weight, attended_features = self.dot_product_attention(h,image_f)
            # print(attended_feature.shape)
            if self.using_gate:
                gate_val = self.gate(h)
                # print(gate_val.shape)
                attended_features = attended_features*gate_val
            
            lstm_input = torch.cat([attended_features,prev_embedding],axis=1)
            h,c = self.lstm_step(lstm_input,(h,c))
            score = self.score_layer(h)
            prev_word = torch.max(score,1)[1]
            hs.append(h)
            scores.append(score)
            attention_weights.append(attention_weight)
            reasons.append(prev_word)
        hs, scores, attention_weights, reasons = torch.stack(hs), torch.stack(scores),torch.stack(attention_weights),torch.stack(reasons) 

        return hs, scores, attention_weights, reasons
