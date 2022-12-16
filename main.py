from selfAttention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformer import *
selfAttention = MultiHeadedSelfAttention(heads=8,tokens=7,embeddingDimension=5)
bruh = FeedForward(5,30)
#number of tokens is 7
#embedding dimension is 5

input = torch.randn(7,5)

data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
#print(poses)

testimg, testpose = images[101], poses[101]
images = images[:100,...,:3]
poses = poses[:100]

# Plot a random image from the dataset for visualization.
#plt.imshow(images[np.random.randint(low=0, high=num_images)])
#plt.show()

# tokens, vocabSize, embeddingDimension, selfAttentionHeads, selfAttentionDimension
encoder = Encoder(6, 100, 7, 8 , 7)
decoder = Decoder(6, 100, 7, 8 , 7)

encoderInput = torch.tensor(
    [1,2,3,4,5,6]
)
decoderInput = torch.tensor(
    [0]
)

encoded = encoder(encoderInput)
decoded = decoder(decoderInput, encoded)
print(decoded)