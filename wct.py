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

	def transform(self,content_features):

        content_features = content_features.double()
            
        content_features_view = content_features.view(content_features.size(0),-1)

        content_features_view_size = content_features_view.size()

        # Subtract mean from content image data
        content_mean = torch.mean(content_features_view,1).unsqueeze(1).expand_as(content_features_view)
        content_features_view = content_features_view - content_mean

        # Obtain the corresponding SVD for the content image
        contentConv = torch.mm(content_features_view, content_features_view.t()).div(content_features_view_size[1]-1) + torch.eye(content_features_view_size[0]).double()
        c_u,c_e,c_v = torch.svd(contentConv,some=False)

        # Find the diagonal element that are extremly small, and consider that to be the end point when reading the matrix
        k_c = content_features_view_size[0]
        for i in range(content_features_view_size[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break
		
	c_e_after_diag = c_e[0:k_c]
        c_v_after_diag = c_v[:,0:k_c]
	c_d = (c_e_after_diag).pow(-0.5)
		
	# Whitening process of content image
        eigen_vector_times_value = torch.mm(c_v_after_diag, torch.diag(c_d))
        final_equation = torch.mm(eigen_vector_times_value, (c_v_after_diag.t()))
        whiten_content_features_view = torch.mm(final_equation, content_features_view)

