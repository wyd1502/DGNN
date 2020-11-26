import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def choose_target(model,memory_s, memory_g, src_mem):
  u = model.memory_merge(memory_s[1], memory_g[1]) #[num_nodes,mem_d]
  u_norm = torch.norm(u, dim=1)  #[num_nodes, 1]
  u_normalized = u/u_norm.view(-1, 1)  #[num_nodes,mem_d]
  src_mem_norm = torch.norm(src_mem, dim=1)  #[batch_size, 1]
  src_mem_normalized = src_mem / src_mem_norm.view(-1, 1)  #[batch_size, mem_d]
  cos_similarity = torch.matmul(src_mem_normalized, u_normalized.t()) #[batch_size, num_nodes]
  cos_similarity, idx = torch.sort(cos_similarity, descending=True)
  return cos_similarity, idx

def recall(des_node, idx, top_k):
  bs = idx.shape[0]
  idx = idx[:, :top_k] #[bs,top_k]
  recall = np.array([a in idx[i] for i, a in enumerate(des_node)])#[bs,1]
  recall = recall.sum() / recall.size
  return recall

def MRR(des_node, idx):
  bs = idx.shape[0]
  mrr = np.array([float(np.where(idx[i].cpu() == a)[0] + 1) for i, a in enumerate(des_node)])#[bs,1]
  mrr = (1 / mrr).mean()
  return mrr


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_mrr, val_recall_20, val_recall_50 = [], [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      src_mem, des_mem = model(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, test=True)

      src_cos_sim, src_idx = choose_target(model, model.memory_s.memory, model.memory_g.memory, src_mem)
      des_cos_sim, des_idx = choose_target(model, model.memory_s.memory, model.memory_g.memory, des_mem)
      recall_20 = (recall(destinations_batch, src_idx, 20) + recall(sources_batch, des_idx, 20)) / 2
      recall_50 = (recall(destinations_batch, src_idx, 50) + recall(sources_batch, des_idx, 50)) / 2
      mrr = (MRR(destinations_batch, src_idx) + MRR(sources_batch, des_idx)) / 2
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_mrr.append(mrr)
      val_recall_20.append(recall_20)
      val_recall_50.append(recall_50)

  return np.mean(val_mrr), np.mean(val_recall_20), np.mean(val_recall_50)
