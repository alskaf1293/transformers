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
    def forward(self, values, keys, queries, mask):
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
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        #print(N, sequenceLength, attention.shape)
        if mask != None:
            #print(attention.shape,mask.shape)
            attention = attention.masked_fill(mask==0, float('-1e20'))
        
        attention = torch.softmax(attention / (self.embeddingSize**(1/2)), dim=3)#F.softmax(attention / keys.size()[1])

        #multiplying matrix of size
        # N, heads, queries sequence length, keys sequence length
        # N, values sequence length, heads, head_dimensions
        # N, queries sequence length, heads, head_dimensions

        output = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        #print(N, sequenceLength, output.shape)
        output = output.reshape(N,sequenceLength,self.embeddingSize)
        return self.finalLinear(output)

class TransformerBlock(nn.Module):
    def __init__(self, embeddingSize, heads, dropout, linearDimension):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embeddingSize,heads)
        self.norm1 = nn.LayerNorm(embeddingSize)
        self.norm2 = nn.LayerNorm(embeddingSize)

        self.linear1 = nn.Linear(embeddingSize, linearDimension*embeddingSize)
        self.linear2 = nn.Linear(linearDimension*embeddingSize, embeddingSize)

        self.dropout = nn.Dropout(dropout)
    def forward(self, values, keys, queries, mask):
        output = self.attention(values,keys,queries,mask)
        x = self.dropout(self.norm1(output+queries))
        
        forward = self.linear2(F.relu(self.linear1(x)))

        return self.dropout(self.norm2(forward+x))

class Encoder(nn.Module):
    def __init__(self, vocabSize, embeddingSize, numLayers, heads, device, linearDimension, dropout, maxLength):
        super(Encoder, self).__init__()
        self.numLayer = numLayers
        self.device = device

        self.inputEmbedding = nn.Embedding(vocabSize, embeddingSize)
        self.positionalEmbedding = nn.Embedding(maxLength, embeddingSize)

        self.layers = nn.ModuleList(
            [TransformerBlock(embeddingSize, heads, dropout, linearDimension) for _ in range(numLayers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        #N by sequenceLength
        N, sequenceLength = x.shape

        positions = torch.arange(0, sequenceLength).expand(N,sequenceLength).to(self.device)
        transformerInput = self.dropout(
            (self.inputEmbedding(x).to(self.device) + self.positionalEmbedding(positions).to(self.device))
        )
        for y in self.layers:
            transformerInput = y(transformerInput,transformerInput,transformerInput,mask)
        return transformerInput

class DecoderBlock(nn.Module):
    def __init__(self, embeddingSize, heads, dropout, linearDimension):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embeddingSize,heads)
        self.norm = nn.LayerNorm(embeddingSize)
        self.transformerBlock = TransformerBlock(embeddingSize, heads, dropout, linearDimension)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, values, keys, srcMask, trgMask):
        output = self.attention(x, x, x, trgMask)
        output = self.dropout(self.norm(output+x))
        return self.transformerBlock(values, keys, output, srcMask)

class Decoder(nn.Module):
    def __init__(self, vocabSize, embeddingSize, numLayers, heads, linearDimension, dropout, device, maxLength):
        super(Decoder, self).__init__()
        self.vocabSize = vocabSize
        self.maxLength = maxLength
        self.numLayers = numLayers
        self.device = device

        self.inputEmbedding = nn.Embedding(vocabSize, embeddingSize)
        self.positionalEmbedding = nn.Embedding(maxLength, embeddingSize)

        self.layers = nn.ModuleList([DecoderBlock(embeddingSize, heads, dropout, linearDimension) for _ in range(numLayers)])
        self.fullyConnected = nn.Linear(embeddingSize, vocabSize)

        self.dropout = nn.Dropout(dropout)
    def forward(self, trg, src, srcMask, trgMask):
        N, sequenceLength = trg.shape
        positions = torch.arange(0, sequenceLength).expand(N,sequenceLength).to(self.device)
        output = self.dropout((self.inputEmbedding(trg) + self.positionalEmbedding(positions)))
        for y in self.layers:
            #print(output.shape)
            output = y(output, src, src, srcMask, trgMask)
        
        #output is sequenceLength, vocabSize
        return self.fullyConnected(output)

class Transformer(nn.Module):
    def __init__(self, srcVocabSize, trgVocabSize, srcPadIndex, trgPadIndex, device="cuda:0", embeddingSize=512, numLayers=3, linearDimension=4, heads=8, dropout=0.1,maxLength=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(srcVocabSize, embeddingSize, numLayers,heads, device, linearDimension, dropout, maxLength)

        self.decoder = Decoder(trgVocabSize, embeddingSize, numLayers, heads, linearDimension, dropout, device, maxLength)
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
        output = self.decoder(trg, encoderOutput, srcMask, trgMask)
        return output
