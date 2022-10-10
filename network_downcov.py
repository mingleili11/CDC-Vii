import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_c=3, embed_dim=256, img_size=128, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        B, C, H, W = x.shape
        a = self.proj(x)
        #b = self.proj(x).flatten(2)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        x = self.downConv(x)
        #x = self.downConv(x.permute(0, 2, 1))
        #x = self.downConv(x)

        x = self.activation(x)
        x = self.norm(x)
        x = self.maxPool(x.transpose(1,2)).transpose(1,2)
        #x = x.transpose(1,2)
        return x

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=2, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        # aa = K.unsqueeze(-3)bxb
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)#
        index_sample1 = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))
        #index_sample1 = torch.randint(1, (L_Q, sample_k))
        #cc = torch.arange(L_Q).unsqueeze(1)
        #dd = torch.arange(L_Q)
        aa = 2 * torch.arange(L_Q / 2)
        index_sample = aa.unsqueeze(0).expand(L_Q, aa.shape[0]).long()
        #ww = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), :]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample]
        # aa = Q.unsqueeze(-2)
        # bb = K_sample.transpose(-2, -1)
        # Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1))
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # find the Top_k query with sparisty measurement
        # aa = torch.div(Q_K_sample.sum(-1), L_K)
        # bb = Q_K_sample.sum(-1)
        # cc = Q_K_sample.max(-1)[0]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]  # ????????????????n_top????
        # use the reduced Q to calculate Q_K
        # aa = torch.arange(B)[:, None, None]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)  # ?????
            # aa = V_sum.unsqueeze(-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)  # ??cumsum
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, attn)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.transpose(2, 1).contiguous(), attn

class AttentionLayer(nn.Module):  #????????attention
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        self.inner_attention = ProbAttention()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        #queries = self.query_projection(queries)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        return self.out_projection(out),attn

class Attention(nn.Module):#fullattention
    def __init__(self,
                 dim,   # ??token?dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.1,
                 proj_drop_ratio=0.1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        B, N, C = x.shape
        #aa =  self.qkv(x)
        #qkv1 = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #q=k=v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,  act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn_full= Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn = AttentionLayer(attention=ProbAttention, d_model=256, n_heads=8,d_keys=None, d_values=None)
        self.drop = nn.Dropout(drop_path_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.conv = ConvLayer(c_in=65)

    def forward(self, x):
        x_out,attn =self.attn(self.norm1(x), self.norm1(x), self.norm1(x), None)
        #x_out = self.drop(x_out)
        #x_out,attn = self.attn_full(self.norm1(x))
        x = x + x_out
        x = x + self.mlp(self.norm2(x))
        x = self.conv(x)
        return x,attn

class Block1(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block1, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn_full= Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn = AttentionLayer(attention=ProbAttention, d_model=256, n_heads=8,d_keys=None, d_values=None)
        self.drop = nn.Dropout(drop_path_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.conv = ConvLayer(c_in=65)

    def forward(self, x):
        x_out,attn =self.attn(self.norm1(x), self.norm1(x), self.norm1(x), None)
        #x_out = self.drop(x_out)
        #x_out,attn = self.attn_full(self.norm1(x))
        x = x + x_out
        x = x + self.mlp(self.norm2(x))
        x = self.conv(x)

        return x,attn

class Block2(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block2, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn_full = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn = AttentionLayer(attention=ProbAttention, d_model=256, n_heads=8, d_keys=None, d_values=None)
        self.drop = nn.Dropout(drop_path_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.conv = ConvLayer(c_in=32)

    def forward(self, x):
        x_out, attn = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), None)
        # x_out = self.drop(x_out)
        # x_out,attn = self.attn_full(self.norm1(x))
        x = x + x_out
        x = x + self.mlp(self.norm2(x))
        x = self.conv(x)

        return x, attn

class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=256, depth=1, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0.1, drop_path_ratio=0.1, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=nn.GELU):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer( patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block1(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        self.fc1 = nn.Linear(257*768, 2560)
        self.fc2 = nn.Linear(2560, 768)
        self.fc3 = nn.Linear(768, 256)
        self.fc4 = nn.Linear(256, 1)
        self.fc = nn.Linear(768, 1)
        self.act = act_layer()
        self.sigmoid = nn.Sigmoid()
        self.attn2s = []
        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward_features(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x,attn1 = self.blocks(x)
        x, attn2 = self.blocks(x)
        #attn2s = []
        #attn2s.append(attn2.tolist())
        x = self.norm(x)
        return x,attn2
    def forward(self, x):
        x ,attn= self.forward_features(x)
        x = x[:, 0]
        x = self.sigmoid(self.fc4(x))
        x = x[:, 0]
        return x ,attn

class Net(nn.Module):
    def __init__(self, inchannels=1, size=128):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 2560)
        self.fc2 = nn.Linear(2560, 768)
        self.fc3 = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = x[:, 0]
        return x


class cascade_VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=256, depth=1, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0.1, drop_path_ratio=0.1, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=nn.GELU):
        super(cascade_VisionTransformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer( patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks1 = nn.Sequential(*[
            Block1(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.blocks2 = nn.Sequential(*[
            Block2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=21, stride=1, padding=10)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256, 1)
        # Classifier head(s)
        self.act = act_layer()
        self.sigmoid = nn.Sigmoid()
        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward_features(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x,attn1 = self.blocks1(x)
        x, attn2 = self.blocks2(x)
        x = self.norm(x)
        return x ,attn1
    def forward(self, x):

        # x_1 = F.relu(self.conv1(x))
        x_1 = F.leaky_relu(self.conv1(x), negative_slope=0.01, inplace=False)
        # x_2 = F.relu(self.conv1(x_1))
        x_2 = F.leaky_relu(self.conv1(x_1), negative_slope=0.01, inplace=False)
        x = x+x_2
        """
        x = x.cpu().detach().numpy()
        x = x[0, 0, :, :]
        sns.heatmap(x)
        plt.show()
        np.save('x.npy', x)
        """
        #x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x ,attn= self.forward_features(x)
        x = x[:, 0]
        x = self.sigmoid(self.fc(x))
        x = x[:, 0]
        return x,attn