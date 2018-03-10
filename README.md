# myRNN

作业1：RNN反向传播小作业 

https://www.tinymind.com/u012160945/notebooks/10


作业2：word_embedding

https://gitee.com/MengNiMeia/myRNN/blob/master/word2vec_songci.py 

输出图片：
https://github.com/mengnimei/myRNN/blob/master/tsne.png

对embeding的理解：
传统的onehot向量表达法，可以很方便的给词进行编码运算，但是词与词之间的语义关系很难进行表达。
而WordEmbedding就是神经网路的方式训练出一个map，将词映射到空间上来表达。语义相近相关的词往往在空间上会表现出一定的相似性，而通过这些相似性可以让计算机看起来更理解自然语言。
这种语义关系一般通过文档样本的学习来实现，在本实验中，QuanSongCi.txt就充当了训练用的文档样本。

实验结果分析：
实验输出图片基本实现了词向量的训练，图中可见向二三四五八十百千等数字聚集在一起，说明这些词在语义上有明显的相关性，和实际情况相符合。
标点符号也出现了聚集，这也说明训练过程中找到了标点符号语义上的相关性。

作业3：RNN训练部分

训练结果：https://www.tinymind.com/executions/tqxe7m0q
对RNN的认识：通过W权重矩阵，在训练过程中产生hidden state，hidden state又作为下一个时刻的训练输入，所以使得RNN在时序上有了记忆的功能。

训练心得：
开始由于keep_prob的设置原因，一直看不到sample.py的验证log输出，经过调试后发现tinymind输出log无法以中文体现。
训练learningrate一直采用0.0001，loss可以看见明显的下降过程，学习率设定基本合理，但到了loss小于5以后就很难再降低。




# myRNN
# myRNN
