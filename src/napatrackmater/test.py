import torch


def sample_points(data, num):
    pos, face = data["pos"], data["face"]

    # Check if the input mesh data has the expected dimensions
    if pos.size(1) != 3 or face.size(0) != 3:
        raise ValueError("Input mesh data has incorrect dimensions.")

    pos_max = pos.abs().max()
    pos = pos / pos_max

    area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    sample = torch.multinomial(prob, num, replacement=True)
    face = face[:, sample]

    frac = torch.rand(num, 2)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]

    pos_sampled = pos[face[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    pos_sampled = pos_sampled * pos_max

    return pos_sampled


# Example usage of the modified sample_points function:
data = {
    "pos": torch.randn(100, 3),  # Replace with your mesh vertex data
    "face": torch.randint(0, 100, (3, 100)),  # Replace with your mesh face data
}
num_points = 1000  # Replace with the desired number of sampled points
sampled_points_value = sample_points(data, num_points)
print(sampled_points_value)
