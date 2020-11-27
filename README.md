# DGNN
Pytorch implementation of "Streaming Graph Neural Network" 
(https://arxiv.org/abs/1810.10627)
(I'm not author just trying to reproduce the experiment results)


The best result of my implementation on UCI dataset is :
mrr: 0.0259, recall@20: 0.1276, recall@50:0.2078.

The result in orgin paper is :
mrr:0.0342 recall@20:0.1284  recall@50: 0.2547.

Any changes to the existing implementation are welcome.

The main difficulty is how to process several events in one batch(batch size > 1). I borrow the message aggregation idea from Tgn(http://arxiv.org/abs/2006.10637) to solve the problem. Besides, I have not used 4 propagation module as the paper does but 2 in my implementation.

