# Transformers from scratch
This is me implementing transformers from scratch. 
Given how ubiquitous these models have been over the past few years, it makes sense to both get a really good understanding of it
as well as know at a low level (Yes, Pytorch is low level for me) how it works, as since its inception, it largely renders RNN's and
LSTM's obselete because of its self-attention mechanism allowing limitless knowledge of previously seen inputs, as well as it's ability to be run in parallel, making it much easier for GPUs to train on.

## 12/30/2022 update:
Finished writing the model with https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson (Aladdin Persson) as a reference. I tried my best
to write the models based off my own knowledge, but often referenced Aladdin's github code and videos if needed a little push. https://www.youtube.com/watch?v=gJ9kaJsE78k&ab_channel=HeduAI , Hedu AI's videos provided a good high level yet practical overview of how transformers work, so I combined both of these
resources to write this code.

The model still doesn't work, haha. the transformer keeps returning the end token and never writes anything. I suspect it's due to the loss function, as
my model should have a high enough capacity to represent the mapping, but it could be that I wrote something seriously wrong in the model. Next time I
update this, I'll also clean up the code, explain more, and fix the bugs.

## 12/31/2022 update:
I fixed the bug. The model now converges to produce a German to English text translator. As a sample, I used the sentence "ein pferd geht unter einer brücke neben einem boot." to see the progress of the model during training. 

<p align = "center"><img src="images/translateSentence.PNG"></p>

After 20 epochs, which was around 12 minutes of training time on my RTX 3060 (see, Mom! I didn't just buy a graphics card to play video games!), you have a sentence that looks like this.

<p align = "center"><img src="images/20epochsCropped.png"></p>

The word ordering is different in comparison to Google Translate but I the meaning is still the same. The printed tensors is the loss printed after running through every batch. The structure of the model was 3 encoder and decoder layers, with a max length of 100 for the input sequences, 8 heads for every transformer block, and an embedding size of 512.