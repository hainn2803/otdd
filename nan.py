import torch 

dist_path = "saved_split_tiny_imagenet/dist_pairwise/sotdd_distance.pt"

dist_tensor = torch.load(dist_path, map_location='cpu')

for i in range(10):
    for j in range(i+1, 10):
        dist