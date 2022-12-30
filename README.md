# Transformers from scratch
This is me implementing transformers from scratch. 
Given how ubiquitous these models have been over the past few years, it makes sense to both get a really good understanding of it
as well as know at a low level (Yes, Pytorch is low level for me) how it works, as since its inception, it largely renders RNN's and
LSTM's obselete because of its self-attention mechanism allowing limitless knowledge of previously seen inputs.

## 12/30/2022 update:
Finished writing the model with https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson (Aladdin Persson) as a reference. I tried my best
to write the models based off my own knowledge, but often referenced Aladdin's github code and videos if needed a little push. https://www.youtube.com/watch?v=gJ9kaJsE78k&ab_channel=HeduAI , Hedu AI's videos provided a good high level yet practical overview of how transformers work, so I combined both of these
resources to write some of this code.

The model still doesn't work, haha. the transformer keeps returning the end token and never writes anything. I suspect it's due to the loss function, as
my model should have a high enough capacity to represent the mapping, but it could be that I wrote something seriously wrong in the model. Next time I
update this, I'll also clean up the code, explain more, and fix the bugs.
