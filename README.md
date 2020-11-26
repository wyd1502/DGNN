# DGNN
Pytorch implementation of "Streaming Graph Neural Network" (https://dl.acm.org/doi/epdf/10.1145/3397271.3401092)(https://arxiv.org/abs/1810.10627)
(I'm not author just trying to reproduce the experiment results)
The best result of my implementation on UCI dataset is mrr: 0.02587241562385947, recall@20: 0.1276123188405797, recall@50:0.20776449275362321.
The result in orgin paper is mrr:0.0342 recall@20:0.1284  recall@50: 0.2547
The main difficulty is how to process several events in one batch(batch size > 1). I borrow the message aggregation idea from Tgn(http://arxiv.org/abs/2006.10637) to solve the problem. Besides, I only use two propagation module in my implementation not four in the origin paper.
Any changes to the existing implementation are welcome.
