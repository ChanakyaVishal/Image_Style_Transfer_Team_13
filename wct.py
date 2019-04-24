from __future__ import division
import torch
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
from model import decoder1
from model import encoder1
import torch.nn as nn

class WCT(nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        # Load pre-trained network
        vgg1 = load_lua(args.vgg1)
        decoder1_torch = load_lua(args.decoder1)

        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)

    def transform(self, content_features, style_features):

        content_features = content_features.double()
        style_features = style_features.double()
            
        content_features_view = content_features.view(content_features.size(0),-1)
        style_features_view = style_features.view(content_features.size(0),-1)

        content_features_view_size = content_features_view.size()
        style_features_view_size = style_features_view.size()

        # Subtract Mean from content image data
        content_mean = torch.mean(content_features_view,1).unsqueeze(1).expand_as(content_features_view)
        content_features_view = content_features_view - content_mean

        # Obtain the SVD of the covariance matrix
        contentConv = torch.mm(content_features_view, content_features_view.t()).div(content_features_view_size[1]-1) + torch.eye(content_features_view_size[0]).double()
        c_u, c_e, c_v = torch.svd(contentConv,some=False)

        # Ignore the elmenents of the diagonla matrix which are lesser than the thershold
        k_c = content_features_view_size[0]
        for i in range(content_features_view_size[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        # Subtract Mean from style image data
        style_mean = torch.mean(style_features_view,1)
        style_features_view = style_features_view - style_mean.unsqueeze(1).expand_as(style_features_view)
        
        # Obtain the SVD of the covariance matrix
        styleConv = torch.mm(style_features_view,style_features_view.t()).div(style_features_view_size[1]-1)
        s_u, s_e, s_v = torch.svd(styleConv,some=False)

        k_s = style_features_view_size[0]
        for i in range(style_features_view_size[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_e_after_diag = c_e[0:k_c]
        c_v_after_diag = c_v[:,0:k_c]
        c_d = (c_e_after_diag).pow(-0.5)

        # Whitening process of content image
        eigen_vector_times_value = torch.mm(c_v_after_diag, torch.diag(c_d))
        final_equation = torch.mm(eigen_vector_times_value, (c_v_after_diag.t()))
        whiten_content_features_view = torch.mm(final_equation, content_features_view)

        s_d = (s_e[0:k_s]).pow(0.5)
        
        # Coloring process
        s_v_after_diag = s_v[:,0:k_s]
        recon_1 = torch.mm(s_v_after_diag, torch.diag(s_d))
        recon_2 = torch.mm(recon_1, (s_v_after_diag.t()))
        target_feature = torch.mm(recon_2, whiten_content_features_view)
        target_feature = target_feature + style_mean.unsqueeze(1).expand_as(target_feature)    
        target_feature = target_feature.view_as(content_features)


