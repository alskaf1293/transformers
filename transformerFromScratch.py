import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embeddingSize, heads):
        super(SelfAttention, self).__init__()
        self.embeddingSize = embeddingSize
        self.heads = heads
        self.head_dimensions = embeddingSize // heads
        #input will be of size N, sequence length, embeddingDimension
        
        assert (self.head_dimensions * self.heads == embeddingSize), "embedding size not a multiple of heads"
        #these are matrix multiplications on the same input, but retain the same dimensions after the operations
        self.toQueries = nn.Linear(embeddingSize, embeddingSize)
        self.toValues = nn.Linear(embeddingSize, embeddingSize)
        self.toKeys = nn.Linear(embeddingSize, embeddingSize)

        self.finalLinear = nn.Linear(self.heads*self.head_dimensions,self.heads*self.head_dimensions)
    def forward(self, queries, keys, values, mask):
        N, sequenceLength, _ = queries.shape

        valueLength, keyLength, queryLength = values.shape[1], keys.shape[1], queries.shape[1]
        #print(valueLength, keyLength, queryLength)
        queries = self.toQueries(queries)
        keys = self.toKeys(keys)
        #print(N, keys.shape)
        #print(queries.shape, keys.shape, values.shape)
        values = self.toValues(values)

        queries = queries.reshape(N, queryLength, self.heads, self.head_dimensions)
        keys = keys.reshape(N, keyLength, self.heads, self.head_dimensions)
        values = values.reshape(N, valueLength, self.heads, self.head_dimensions)

        #multiplying matrix of size 
        # N, queries sequence length, heads, head_dimensions
        # N, keys sequence length, heads, head_dimensions
        # to
        # N, heads, queries sequence length, keys sequence length
        attention = torch.einsum("nqhl,nkhl->nhqk", [queries, keys])
        #print(N, sequenceLength, attention.shape)
        if mask != None:
            #print(attention.shape,mask.shape)
            attention = attention.masked_fill(mask==0, float('-1e20'))
        
        attention = F.softmax(attention / keys.size()[1])

        #multiplying matrix of size
        # N, heads, queries sequence length, keys sequence length
        # N, values sequence length, heads, head_dimensions
        # N, queries sequence length, heads, head_dimensions

        output = torch.einsum("nhqk,nvhd->nqhd", [attention, values])
        #print(N, sequenceLength, output.shape)
        output = output.reshape(N,sequenceLength,self.embeddingSize)
        return self.finalLinear(output)

class TransformerBlock(nn.Module):
    def __init__(self, embeddingSize, heads, linearDimension):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embeddingSize,heads)
        self.norm1 = nn.LayerNorm(embeddingSize)
        self.norm2 = nn.LayerNorm(embeddingSize)

        self.linear1 = nn.Linear(embeddingSize, linearDimension*embeddingSize)
        self.linear2 = nn.Linear(linearDimension*embeddingSize, embeddingSize)
    def forward(self, queries, keys, values, mask):
        output = self.attention(queries,keys,values,mask)
        x = self.norm1(output+queries)
        output = self.linear2(F.relu(self.linear1(output)))

        return self.norm2(output+x)

class Encoder(nn.Module):
    def __init__(self, vocabSize, maxLength, embeddingSize, heads, linearDimension, numLayers, device):
        super(Encoder, self).__init__()
        self.numLayer = numLayers
        self.device = device

        self.inputEmbedding = nn.Embedding(vocabSize, embeddingSize)
        self.positionalEmbedding = nn.Embedding(maxLength, embeddingSize)

        self.layers = nn.ModuleList(
            [TransformerBlock(embeddingSize, heads, linearDimension) for _ in range(numLayers)]
        )

    def forward(self, x, mask):
        #N by sequenceLength
        N, sequenceLength = x.shape

        positions = torch.arange(0, sequenceLength).expand(N,sequenceLength).to(self.device)
        transformerInput = self.inputEmbedding(x).to(self.device) + self.positionalEmbedding(positions).to(self.device)

        for y in range(self.numLayer):
            transformerInput = self.layers[y](transformerInput,transformerInput,transformerInput,mask)
        return transformerInput

class DecoderBlock(nn.Module):
    def __init__(self, embeddingSize, heads, linearDimension):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embeddingSize,heads)
        self.norm = nn.LayerNorm(embeddingSize)
        self.transformerBlock = TransformerBlock(embeddingSize, heads, linearDimension)
    def forward(self, x, keys, values, srcMask, trgMask):
        output = self.attention(x, x, x, trgMask)
        output = self.norm(output)
        return self.transformerBlock(output, keys, values, srcMask)

class Decoder(nn.Module):
    def __init__(self, vocabSize, maxLength, embeddingSize, heads, linearDimension, numLayers, device):
        super(Decoder, self).__init__()
        self.vocabSize = vocabSize
        self.maxLength = maxLength
        self.numLayers = numLayers
        self.device = device

        self.inputEmbedding = nn.Embedding(vocabSize, embeddingSize)
        self.positionalEmbedding = nn.Embedding(maxLength, embeddingSize)

        self.layers = nn.ModuleList([DecoderBlock(embeddingSize, heads, linearDimension) for _ in range(numLayers)])
        self.fullyConnected = nn.Linear(embeddingSize, vocabSize)
    def forward(self, src, trg, srcMask, trgMask):
        N, sequenceLength = trg.shape
        positions = torch.arange(0, sequenceLength).expand(N,sequenceLength).to(self.device)
        output = self.inputEmbedding(trg) + self.positionalEmbedding(positions)
        for x in range(self.numLayers):
            #print(output.shape)
            output = self.layers[x](output, src, src, srcMask, trgMask)
        
        #output is sequenceLength, vocabSize
        return self.fullyConnected(output)

class Transformer(nn.Module):
    def __init__(self, srcVocabSize, trgVocabSize, srcPadIndex, trgPadIndex, device="cuda:0", embeddingSize=512, numLayers=4, linearDimension=6, heads=8, maxLength=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(srcVocabSize, maxLength, embeddingSize, heads, linearDimension, numLayers, device)

        self.decoder = Decoder(trgVocabSize, maxLength, embeddingSize, heads, linearDimension, numLayers, device)
        self.srcPadIndex = srcPadIndex
        self.trgPadIndex = trgPadIndex
        self.device = device
    def createSrcMask(self, src):
        #shape is of N, srcSequenceLength
        return (src == self.srcPadIndex).unsqueeze(1).unsqueeze(2).to(self.device)
    def createTrgMask(self, trg):
        #shape is of N, trgSequenceLength
        N, trgSequenceLength = trg.shape

        return torch.tril(torch.ones((trgSequenceLength, trgSequenceLength))).expand(N,1,trgSequenceLength,trgSequenceLength).to(self.device)
    def forward(self, src, trg):
        srcMask = self.createSrcMask(src)
        trgMask = self.createTrgMask(trg)

        encoderOutput = self.encoder(src, srcMask)
        output = self.decoder(encoderOutput, trg, srcMask, trgMask)
        return output
