from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def initialize_raft(freeze_weights=True):
    # Initialize the RAFT model for optical flow estimation
    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights, progress=False)
    transforms = weights.transforms()
    if freeze_weights:
        for parameter in raft.parameters():
            parameter.requires_grad = False
    return raft, transforms