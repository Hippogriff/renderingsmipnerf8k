import numpy as np
import imageio
import matplotlib.pyplot as plt

fname = "data/texture.npy"
texdata = np.load(fname, mmap_mode="r")
num_lobes=6
sigma = texdata[:, :, -1]
alpha = (1 - np.exp(-sigma * 0.005))

diffuse = texdata[:, :, :3]
features = texdata[..., 3:-1]
print(features.shape)
# lobes = np.reshape(features, (features.shape[0], features.shape[1], 6, 7))
lobes = np.reshape(features, (features.shape[0], features.shape[1], num_lobes, 7))
axes = lobes[..., :3]
lambdas = np.abs(lobes[..., 3])
c = lobes[..., 4:]

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

print(lobes.shape)
print(axes.shape)
print(lambdas.shape)
print(c.shape)

def compress_polar_coordinates(vectors):
    vectors = vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-6)
    azimuth = (np.arctan2(vectors[..., 1], vectors[..., 0]) * 128 / np.pi + 128).astype(np.uint8)
    elevation = (np.arccos(vectors[..., 2]) * 256 / np.pi).astype(np.uint8)
    return azimuth, elevation

azimuth, elevation = compress_polar_coordinates(axes)

# plt.hist(elevation.flatten(), bins=256)
# plt.show()

# quit()
# print(np.linalg.norm(axes, axis=-1))

log_lambda = np.log(np.clip(lambdas, 1e-5, np.inf))
compressed_lambda = np.clip((log_lambda + 2.5) / 7.5, 0.0, 1.0)
compressed_lambda = np.array(255 * compressed_lambda, dtype=np.uint8)

def write_color(filename, data):
    data = np.array(np.clip(data * 255, 0, 255), dtype=np.uint8)
    imageio.imwrite(filename, data)

write_color("data/diffuse.png", sigmoid(diffuse))

for i in range(num_lobes):
    lambda_axis = np.stack([
        compressed_lambda[..., i],
        azimuth[..., i],
        elevation[..., i]
    ], axis=-1)
    imageio.imwrite(f"data/lambda_axis_{i}.png", lambda_axis)

    write_color(f"data/color_{i}.png", sigmoid(c[..., i, :]))

# plt.hist(compressed_lambda.flatten(), bins=np.linspace(0.0, 1.0, 256))
# plt.show()

# print(compressed_lambda.shape)

def write(filename, data):
    data = np.array(np.clip(data * 255, 0, 255), dtype=np.uint8)
    imageio.imwrite(filename, data)

write("data/alpha.png", alpha)

quit()


# def spherical_gaussian(x, direction):
#     axis = x[..., :3]
#     # normalize axis
#     axis = axis / np.linalg.norm(axis, dim=-1, keepdim=True)
#     lambda_ = np.abs(x[..., 3])
#     a = np.abs(x[..., 4])
#     return a * np.exp(-lambda_ * (1 - np.sum(axis * direction, -1)))

# def spherical_gaussian_mixture(self, x, direction):
#     # Split x into number of lobes
#     x = torch.chunk(x, self.num_g_lobes, dim=-1)
#     rgb = torch.zeros((x[0].shape[0], 3)).cuda()
#     for x_ in x:
#         # split x_ into 3 parts corresponding to RGB
#         x_ = torch.chunk(x_, 3, dim=-1)
#         # compute the spherical gaussian
#         rgb_ = torch.cat([self.spherical_gaussian(x_[i], direction).unsqueeze(1) for i in range(3)], dim=-1)
#         rgb = rgb + rgb_
#     return rgb

# def features(self, x):
#     density, embedding = self.query_density(x, return_feat=True)
#     h = embedding.reshape(-1, self.geo_feat_dim)
#     features = (
#         self.mlp_head(h)
#         .reshape(list(embedding.shape[:-1]) + [self.num_g_lobes * 3 * (3 + 1 + 1) + 3])
#         .to(embedding)
#     )
#     features = np.concatenate([features, density], dim=-1)
#     return features


def features_to_rgb(features, dir):
    diffuse_color = features[:, :, :3]
    # rgb = sigmoid(diffuse_color + spherical_gaussian_mixture(features[:, 3:], dir))
    rgb = sigmoid(diffuse_color)
    return rgb

features = texdata[:, :, :-1]
rgb = features_to_rgb(features, np.array([0, 0, 1]))

print(rgb.shape)

write("color.png", rgb)
