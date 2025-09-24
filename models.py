import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_conv1d(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(basic_conv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return x

class basic_dwconv1d(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=out_ch)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return x

class var_dwconv1d(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size=3, expand = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, in_ch*expand, kernel_size = 1),
            nn.BatchNorm1d(in_ch*expand),            
            nn.GELU(),
            nn.Conv1d(in_ch*expand, in_ch*expand, kernel_size = kernel_size, padding=kernel_size//2, groups=in_ch*expand),
            nn.BatchNorm1d(in_ch*expand),            
            nn.GELU(),
            nn.Conv1d(in_ch*expand, out_ch, kernel_size = 1),
            )

    def forward(self, x):
        x = self.conv(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, out_ch, heads, freq_hidden):
        super().__init__()
        self.scalemod = scale_attention(out_ch, ca_num_heads=heads)
        self.fus = nn.Conv1d(out_ch*3, out_ch, kernel_size=1)
        self.freqmod = freq_attention(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_mod = self.scalemod(x) + x
        x_mod = self.norm1(x_mod)
        x_freq = self.freqmod(x) + x
        x_freq = self.norm2(x_freq)
        x_fus = self.fus(((torch.cat((x_mod, x_freq, x), dim=1))))
        
        return x_fus

class basic_conv1d_trans(nn.Module):
    def __init__(self, in_ch, out_ch, heads = 1, kernel_size = 3, first_layer = False, block_num = 2, freq_hidden = 64):
        super(basic_conv1d_trans, self).__init__()
        self.firstlayer = first_layer
        self.block_num = block_num
        if first_layer:
            self.conv1x1 = basic_conv1d(in_ch, out_ch, kernel_size=kernel_size)
        self.enc_block = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(0, block_num):
            self.enc_block.append(encoder_block(out_ch, heads, freq_hidden))
            self.norm.append(nn.BatchNorm1d(out_ch))
        
        self.norm1 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        if self.firstlayer:
            x = self.conv1x1(x)
        for i in range(0, self.block_num):
            x = self.enc_block[i](x)

        return x

class freq_attention(nn.Module):
    def __init__(self, dim, hidden_dim = 64):
        super().__init__()
        hidden_dim = dim
        self.split = [hidden_dim//2]*2
        self.head1 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=7,padding=3, groups=hidden_dim//2)
        self.head2 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=5,padding=2, groups=hidden_dim//2)
        self.agg = nn.Conv1d(hidden_dim, dim, kernel_size=1, padding=0)
        self.agg1 = nn.Conv1d(hidden_dim, dim, kernel_size=1, padding=0)
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
        
    def forward(self, x):

        x_glo = x
        x_fft = torch.fft.rfftn(x_glo, norm='ortho', dim=-1)
        mag = torch.abs(x_fft)      
        phase = torch.angle(x_fft)       
        x1, x2 = torch.split(mag, self.split, dim=1)
        x1 = self.head1(x1)
        x2 = self.head2(x2)
        mag = torch.cat((x1, x2), dim=1)
        x_mod = self.agg(self.gelu(mag))
        x_fft_out = torch.polar(x_mod, phase)
        x_fft_out = torch.fft.irfftn(x_fft_out, norm='ortho', dim=-1)
        x_fft_out = self.agg1(self.gelu(x_fft_out) * self.gelu(x_glo))

        return x_fft_out

class scale_attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, expand_ratio=2):
        super().__init__()

        self.dim = dim 
        self.ca_num_heads = ca_num_heads # Number of conv heads
        self.act = nn.GELU() 
        self.split_groups=self.dim//ca_num_heads # Number of groups
        self.groups_channel = [dim//ca_num_heads]*ca_num_heads

        for i in range(self.ca_num_heads): 
            local_conv = nn.Conv1d(self.split_groups, self.split_groups, kernel_size=5, padding=(2*i), stride=1, groups=self.split_groups, dilation=i)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        
        self.proj0 = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm1d(dim)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU()

    
    def forward(self, x):
        B, C, N = x.shape
        
        s = torch.split(x, self.groups_channel, dim=1)

        # Conv within each head
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]
            s_i = local_conv(s_i)
            s_i = s_i.reshape(B, self.split_groups, -1, N)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out,s_i],2) 
        s_out = s_out.reshape(B, C, N)
        
        # Group aggregation
        s_out = self.proj0(s_out)
        s_out = self.act(self.bn1(s_out))
        s_out = self.bn2(self.proj1(s_out))

        return s_out

class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten() 
        self.fc = nn.Linear(dim, num_classes) 
 
    def forward(self, x):
        x = self.global_avg_pool(x)  
        x = self.flatten(x)  
        x = self.fc(x)  
        return x

class MLP(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_ch, hidden_ch)
        self.fc2 = nn.Linear(hidden_ch, out_ch)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.layernorm1 = nn.LayerNorm(hidden_ch)
        self.layernorm2 = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layernorm1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layernorm2(x)
        x = self.dropout(x)
        return x

class MultiGatedFusionNetwork(nn.Module):
    def __init__(self, input_size, gate_num):
        super(MultiGatedFusionNetwork, self).__init__()
        self.gate_num = gate_num
        self.receivegate = nn.Sequential(
            nn.LayerNorm(input_size * 2),
            nn.Linear(input_size * 2, input_size),
            nn.LayerNorm(input_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )
        self.sendgate = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )
        self.layernorm1 = nn.LayerNorm(input_size)
        self.layernorm2 = nn.LayerNorm(input_size)

    def forward(self, current_input, pre_input):

        out1 = current_input
        out2 = pre_input

        fused = torch.cat((out1, out2), dim=-1)
        gate = self.receivegate(fused)

        gated_output = (0 + gate) * current_input + (1 - gate) * pre_input
        return gated_output

class graph_fusion_block(nn.Module):
    def __init__(self, current_dim, higher_dim, normalize=True, hidden_dim = 128):
        super(graph_fusion_block, self).__init__()
        self.normalize=normalize
        self.gate_num = len(higher_dim)
        self.gate = MultiGatedFusionNetwork(hidden_dim, self.gate_num)


    def forward(self, x_highdim, x_lowdim, current_scale):
        
        x_lowdim = x_lowdim.transpose(1,2)
        x_lowdim_store = x_lowdim

        x_highdim = x_highdim.transpose(1,2)

        if self.normalize:
            x_lowdim = F.normalize(x_lowdim, p=2, dim=-1)
            x_highdim = F.normalize(x_highdim, p=2, dim=-1)
        
        affinity = torch.bmm(x_lowdim, x_highdim.transpose(1,2))
        affinity_matrix = F.softmax(affinity, dim=2)
        x_highdim = torch.bmm(affinity_matrix, x_highdim)

        fused_features = self.gate(x_lowdim_store, x_highdim)
        fused_features = fused_features.transpose(1,2)
    
        return fused_features

class PatchMerging1D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.conv = basic_conv1d(2 * dim, 2 * dim, kernel_size = 3)

    def forward(self, x):
        """
        x: (B, C, H)
        """
        B, C, H = x.shape
        assert H % 2 == 0, f"H size ({H}) cannot be divided by 2"
        
        x0 = x[:,:,0::2]
        x1 = x[:,:,1::2]
        x = torch.cat((x0,x1),dim=1)
        
        x = self.conv(x)
        
        return x

class multiscale_HRRPencoder(nn.Module):
    def __init__(self, scale, channel_dim, conv_heads, num_classes, HRRP_len, freq_hidden, block_num):
        super(multiscale_HRRPencoder, self).__init__()
        self.scale = scale
        self.conv1d = nn.ModuleList()
        self.HRRPnet = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        self.conv1d.append(basic_conv1d_trans(1, channel_dim[0], heads=conv_heads[0],kernel_size=7, first_layer=True, freq_hidden=freq_hidden, block_num=block_num))
        self.patch_merging.append(PatchMerging1D(channel_dim[0]))
        for i in range(0, scale-1):
            self.conv1d.append(basic_conv1d_trans(channel_dim[i], channel_dim[i+1], heads=conv_heads[i], freq_hidden=freq_hidden, block_num=block_num))
            self.patch_merging.append(PatchMerging1D(channel_dim[i]))
        
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
            
        encoder_feature = []
        x_temp = x
        x_temp = self.conv1d[0](x_temp)
        encoder_feature.append(x_temp)

        for i in range(1, self.scale):
            x_temp = self.patch_merging[i](x_temp)
            x_temp  = self.conv1d[i](x_temp) 
            encoder_feature.append(x_temp)     

        return encoder_feature

class multiscale_HRRPdecoder(nn.Module):
    def __init__(self, scale, channel_dim):
        super(multiscale_HRRPdecoder, self).__init__()
        self.conv1d = nn.ModuleList()
        self.de_conv = nn.ModuleList()
        self.ds_conv = nn.ModuleList()
        self.de_conv.append(nn.ConvTranspose1d(channel_dim[0]*2, channel_dim[0], kernel_size=3, stride=2, padding=1,output_padding=1))
        self.conv1d.append(var_dwconv1d(channel_dim[0], 1))
        self.ds_conv.append(nn.Conv1d(channel_dim[0], 1, kernel_size=3, padding=1))
        for i in range(1,scale):
            self.de_conv.append(nn.ConvTranspose1d(channel_dim[i]*2, channel_dim[i-1]*2, kernel_size=3, stride=2, padding=1,output_padding=1))
            self.conv1d.append(var_dwconv1d(channel_dim[i], channel_dim[i-1]))
            self.ds_conv.append(nn.Conv1d(channel_dim[i], 1, kernel_size=3, padding=1))
        self.last_fea = var_dwconv1d(channel_dim[-1], channel_dim[-1], expand=1)
        self.scale = scale        

    def forward(self, encoder_feature):
        
        decoder_feature = [None]*(self.scale)
        for i in range(self.scale-1, -1, -1):
            x_temp = encoder_feature[i]
            decoder_feature[i] = torch.sigmoid(self.ds_conv[i](x_temp))
            
        return decoder_feature

class multiscale_graphencoder(nn.Module):
    def __init__(self, scale, channel_dim, num_classes, HRRP_len, gate_hidden):
        super(multiscale_graphencoder, self).__init__()
        self.scale = scale
        self.HRRPnet = nn.ModuleList()
        self.graph_fuser = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(0,scale):
            self.graph_fuser.append(graph_fusion_block(channel_dim[i], channel_dim, hidden_dim = gate_hidden))
            self.conv.append(nn.Sequential(nn.Conv1d(channel_dim[i], gate_hidden, kernel_size=1),
                                    nn.BatchNorm1d(gate_hidden)))
            
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                     nn.Flatten())
        
    def forward(self, x, mask):
            
        graph_fea = []
        channel_fea = []
        for i in range(0, self.scale):
            x_temp = x[i]*mask[i]
            x_temp = self.conv[i](x_temp)
            graph_fea.append(x_temp)

        graph_pre = graph_fea[-1]
        channel_fea.append(self.avgpool(graph_pre))

        for i in range(self.scale-2, -1, -1):
            graph_pre = self.graph_fuser[i](graph_fea[i], graph_pre, i)
            channel_fea.append(self.avgpool(graph_pre))


        return channel_fea

class HRRP_net(nn.Module):
    def __init__(self, num_classes, emb_dim, N, scale, gate_hidden, freq_hidden, block_num):
        super(HRRP_net,self).__init__()
        channel_dim = []
        conv_heads = [4]*scale
        for i in range(0, scale):
            channel_dim.append(emb_dim*(2**i))
        self.HRRPencoder = multiscale_HRRPencoder(scale, channel_dim, conv_heads,num_classes,N, freq_hidden, block_num)
        self.graphencoder = multiscale_graphencoder(scale, channel_dim, num_classes,N, gate_hidden=gate_hidden)
        self.HRRPdecoder = multiscale_HRRPdecoder(scale, channel_dim)
        self.fc = nn.Linear((scale)*gate_hidden, num_classes)

    def forward(self, HRRP):

        enc_fea = self.HRRPencoder(HRRP)
        dec_fea = self.HRRPdecoder(enc_fea)
        channel_fea = self.graphencoder(enc_fea, dec_fea)

        class_hrrp = torch.cat(channel_fea, dim=-1)
        
        class_out =self.fc(class_hrrp)
        
        return class_out, class_hrrp

