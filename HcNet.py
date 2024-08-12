

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class HC_Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=3 // 2, groups=hidden_features, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
         
        x = self.fc1(x)
        
        x = self.act(x)
        x = self.dwconv(x)
        
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x



class HcRaBlock(nn.Module):

    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=GroupNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.sigmoid = nn.Sigmoid()
        #------------------------------0------------------------------------------
        self.pool_0 = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=True)
        self.fc_0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #------------------------------------------------------------------------
        mean_num = torch.ones(1, 1, input_resolution[0], input_resolution[1], requires_grad=False)
        mean_num = self.pool_0(mean_num) * 9.0 - 1.0
        self.register_buffer("mean_num", mean_num)
        # ------------------------------------------------------------------------
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = HC_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        H_, W_ = self.input_resolution
        B, C, H, W = x.shape
        assert H == H_, "input feature has wrong size"

        #x = self.norm1(x)
        #----------------------------------0--------------------------------------
        shortcut_0 = x

        x_avg_0 = x

        x_mean_0 = self.pool_0(x_avg_0)
        x_sum_0 = x_mean_0 * 9.0
        x_around_sum_0 = x_sum_0 - x_avg_0

        x_around_mean_0 = x_around_sum_0/self.mean_num

        second_Der_0 = x_around_mean_0 - x_avg_0

        # -----------------------
        input_Dependency_0 = self.avg_pool(x)
        alpha_DeltaT_0 = self.fc_0(input_Dependency_0)
        alpha_DeltaT_0 = self.sigmoid(alpha_DeltaT_0)
        # -----------------------

        x = shortcut_0 + self.drop_path(alpha_DeltaT_0 * second_Der_0)
        # -------------------------------------------------------------------------

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, outdim, norm_layer=GroupNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim #
        self.reduction = nn.Conv2d(dim, outdim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = norm_layer(outdim)

    def forward(self, x):

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, mlp_ratio=4., drop=0., drop_path=0., norm_layer=GroupNorm, downsample=None, downsample_outdim=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HcRaBlock(dim=dim, input_resolution=input_resolution,
                                 mlp_ratio=mlp_ratio,
                                 drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, outdim=downsample_outdim,norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class StemLayer(nn.Module):

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer= nn.GELU,
                 norm_layer= GroupNorm):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = norm_layer(out_chans // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = norm_layer(out_chans)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class HcNet(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=[64, 128, 320, 512], depths=[2, 2, 6, 2], mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=GroupNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)

        self.embed_dim = embed_dim

        self.ape = False
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = StemLayer(in_chans=in_chans, out_chans=embed_dim[0])

        patches_resolution = [img_size // patch_size, img_size // patch_size]

        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim[i_layer]),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               downsample_outdim=int(embed_dim[i_layer+1]) if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean([-2, -1])

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda')

    model = HcNet(img_size=224,
                    patch_size=4,
                    in_chans=3,
                    num_classes=1000,
                    embed_dim=[64, 128, 320, 512],
                    depths=[4, 4, 12, 4],
                    mlp_ratio=4.,
                    drop_rate=0.0,
                    drop_path_rate=0.2,
                    ape=False,
                    norm_layer=GroupNorm,
                    patch_norm=True,
                    use_checkpoint=False).to(device)


    tensor = torch.randn(2, 3, 224, 224).to(device)

    B, C, W, H = tensor.shape
    pre = model(tensor)
    print(pre.shape)
