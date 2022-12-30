import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embeddingDimension, linearDimension, masked = False):
        super().__init__()

        self.masked = masked
        self.queries = nn.Linear(embeddingDimension, linearDimension)
        self.keys = nn.Linear(embeddingDimension, linearDimension)
        self.values = nn.Linear(embeddingDimension, linearDimension)
        
    def forward(self,x,y,z):
        toQueries = self.queries(x)
        toKeys = self.keys(y)
        toValues = self.values(z)

        attentionFilter = torch.matmul(toQueries, torch.transpose(toKeys,0,1))
        if self.masked:
            infinities = torch.full(attentionFilter.size(), float('-inf')) 
            infinities = torch.triu(infinities, diagonal=1)
            attentionFilter = attentionFilter + infinities

        attentionFilter = F.softmax(attentionFilter / math.sqrt(attentionFilter.size()[0]))

        filteredValue = torch.matmul(attentionFilter, toValues)
        return filteredValue

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, heads, tokens, embeddingDimension, masked=False):
        super().__init__()
        finalDimension = tokens
        linearDimension = embeddingDimension

        self.heads = heads
        self.selfAttentionHeads = [SelfAttention(embeddingDimension, linearDimension, masked) for _ in range(heads)]
        
        self.finalLinear = nn.Linear(heads*tokens, finalDimension)
    def forward(self,x,y,z):
        outputs = [self.selfAttentionHeads[a](x,y,z) for a in range(self.heads)]
        concatVectors = torch.cat(outputs, dim=0)

        #final: finalDimension by linearDimension = tokens by embedding dimensions
        final = torch.transpose(self.finalLinear(torch.transpose(concatVectors,0,1)),0,1)

        #residual connection
        #add
        residual = final + z

        #norm
        stats = []
        for x in residual:
            stats.append((np.mean(x.detach().numpy()), np.std(x.detach().numpy())))
        
        for x in range(len(residual)):
            for y in range(len(residual[0])):
                mean = stats[x][0]
                std = stats[x][1]
                residual[x][y] = (residual[x][y] - mean) / math.sqrt(std*std + 0.0001)
        return residual

class FeedForward(nn.Module):
    def __init__(self, features, embedding):
        super().__init__()
        #input: 7 by 5
        self.linear1 = nn.Linear(features,embedding)

        #256 by 5
        self.linear2 = nn.Linear(embedding, features)

        #output: 7 by 5
    def forward(self, x):
        final = F.relu(self.linear1(x))

        #residual connection
        #add
        residual = final + x
        
        #norm
        stats = []
        for x in residual:
            stats.append((np.mean(x.detach().numpy()), np.std(x.detach().numpy())))
        
        for x in range(len(residual)):
            for y in range(len(residual[0])):
                mean = stats[x][0]
                std = stats[x][1]
                residual[x][y] = (residual[x][y] - mean) / math.sqrt(std*std + 0.0001)
        return residual

class EmbeddingLayer(nn.Module):
    def __init__(self, vocabSize, embeddingDimension):
        super().__init__()
        self.embeddingDimension = embeddingDimension
        self.embedding = nn.Embedding(vocabSize, embeddingDimension)
    
    def forward(self, input):
        input = self.embedding(input)

        #positional encoding
        tokens = input.size()[0]
        d = self.embeddingDimension
        positionalEncoding = torch.zeros((tokens, self.embeddingDimension))
        for x in range(tokens):
            for y in range(self.embeddingDimension):
                if x % 2 == 0: #position is even
                    positionalEncoding[x][y] = math.sin(x/(10000**((2*y)/d)))
                if x % 2 == 1: #position is odd
                    positionalEncoding[x][y] = math.cos(x/(10000**((2*y)/d)))

        return input + positionalEncoding