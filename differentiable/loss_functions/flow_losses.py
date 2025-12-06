import torch
import math

# def mixture_of_laplace_loss(output, flow_gt, valid, gamma=0.8, max_flow=None):
#     """ Loss function defined over sequence of flow predictions """
#     if max_flow is None:
#         flow_h, flow_w = flow_gt.shape[-2:]
#         max_flow = (flow_h ** 2 + flow_w ** 2) ** 0.5
#     n_predictions = len(output['flow'])
#     flow_loss = 0.0
#     # exlude invalid pixels and extremely large diplacements
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()
#     valid = (valid >= 0.5) & (mag < max_flow)
#     for i in range(n_predictions):
#         i_weight = gamma ** (n_predictions - i - 1)
#         loss_i = output['nf'][i]
#         final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
#         flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
#     return flow_loss


def mixture_of_laplace_loss(pred_motions, pred_infos, true_motion, valid_mask,
                            gamma=0.85, 
                            use_var=True, var_min=0, var_max=10, 
                            epsilon=1e-6):
    device, dtype = true_motion.device, true_motion.dtype
    B, _, H, W = true_motion.shape
    
    # mixture of laplace loss in each scale
    num_preds = len(pred_motions)
    total_loss = 0.0

    if not use_var:
        var_max = var_min = 0
        
    for i in range(len(pred_infos)):
        raw_b = pred_infos[i][:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = pred_infos[i][:, :2]
        # Large b Component                
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        # term2: [N, 2, m, H, W]
        term2 = ((true_motion - pred_motions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        # term1: [N, m, H, W]
        term1 = weight - math.log(2) - log_b
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        # Add to total loss
        weight = gamma ** (num_preds - i - 1)
        final_mask = (~torch.isnan(nf_loss.detach())) & (~torch.isinf(nf_loss.detach())) & valid_mask
        total_loss += weight * ((final_mask * nf_loss).sum() / (final_mask.sum() + epsilon))
    return total_loss