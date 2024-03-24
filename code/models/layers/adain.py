import torch

def calc_vector_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    feat_var = feat.var(dim=1) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=1)
    return feat_mean, feat_std

def calc_tensor_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_vector_mean_std(style_feat)
    content_mean, content_std = calc_tensor_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.view(style_std.shape[0],1,1,1).expand(size) + style_mean.view(style_mean.shape[0],1,1,1).expand(size)

def adaptive_instance_normalization2(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_vector_mean_std(style_feat)
    content_mean, content_std = calc_vector_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.unsqueeze(-1).expand(
        size)) / content_std.unsqueeze(-1).expand(size)
    return normalized_feat * style_std.unsqueeze(-1).expand(size) + style_mean.unsqueeze(-1).expand(size)