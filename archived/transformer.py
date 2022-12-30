import torch
import torch.nn as nn
import torch.nn.functional as F

from selfAttention import *

class Encoder(nn.Module):
    def __init__(self, tokens, vocabSize, embeddingDimension, selfAttentionHeads, selfAttentionDimension):
        super().__init__()

        self.positionalEncoding = EmbeddingLayer(vocabSize, embeddingDimension)
        self.selfAttention = MultiHeadedSelfAttention(selfAttentionHeads, tokens, selfAttentionDimension)
        self.feedForward = FeedForward(selfAttentionDimension, embeddingDimension)
    def forward(self, input):
        input = self.positionalEncoding(input)
        input = self.selfAttention(input, input, input)
        input = self.feedForward(input)
        return input

class Decoder(nn.Module):
    def __init__(self, tokens, vocabSize, embeddingDimension, selfAttentionHeads, selfAttentionDimension):
        super().__init__()
        self.positionalEncoding = EmbeddingLayer(vocabSize, embeddingDimension)
        self.selfAttention1 = MultiHeadedSelfAttention(selfAttentionHeads, tokens, selfAttentionDimension, masked=True)
        self.selfAttention2 = MultiHeadedSelfAttention(selfAttentionHeads, tokens, selfAttentionDimension, masked=False)
        self.feedForward = FeedForward(selfAttentionDimension, embeddingDimension)
        self.linear = nn.Linear(tokens*embeddingDimension, vocabSize)
    def forward(self, input, encoderInput):
        input = self.positionalEncoding(input)
        input = self.selfAttention1(input, input, input)
        output = self.selfAttention2(encoderInput, encoderInput, input)
        output = self.feedForward(output)
        #flatten operation
        output = output.view((-1))

        #compute logits
        output = self.linear(output)
        #softmax
        return F.softmax(output)
