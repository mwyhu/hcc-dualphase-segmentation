import torch

# Original single-channel checkpoint
input_path = "/Users/michellehu/Desktop/hcc_active_learning/src/corrections/checkpoint_final.pth"

# New inflated checkpoint
output_path = "/Users/michellehu/Desktop/hcc_active_learning/src/corrections/checkpoint_final_2channel.pth"

ckpt = torch.load(
    input_path,
    map_location="cpu",
    weights_only=False
)

state_dict = ckpt["network_weights"]

modified = 0

for key, weight in state_dict.items():

    # Find first convolution layers with 1 input channel
    if (
        isinstance(weight, torch.Tensor)
        and len(weight.shape) == 5
        and weight.shape[1] == 1
    ):
        print("\nInflating:", key)
        print("Old:", weight.shape)

        # Copy single-channel filters to two channels
        # divide by 2 to keep activation scale stable
        new_weight = torch.cat([weight, weight], dim=1) / 2

        state_dict[key] = new_weight

        print("New:", new_weight.shape)

        modified += 1

print("\nLayers modified:", modified)

ckpt["network_weights"] = state_dict

torch.save(
    ckpt,
    output_path
)

print("\nSaved:", output_path)