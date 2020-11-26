import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation import eval_edge_prediction
from models.dgnn import DGNN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics



### Argument and global variables
parser = argparse.ArgumentParser('DGGN training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='UCI-Msg')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')

parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')

parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')

parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--seed', type=int, default=0, help='random seed')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
DROP_OUT = args.drop_out
GPU = args.gpu
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
timenow = time.strftime("%Y-%m-%d_%H:%M:%S",time.localtime(time.time()))
fh = logging.FileHandler('log/{}.log'.format(str(args.prefix)+timenow))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  dgnn = DGNN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device, dropout=DROP_OUT,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)
  for a in dgnn.parameters():
    if a.ndim > 1:
      torch.nn.init.xavier_uniform_(a)

  dgnn.memory_s.__init_memory__(args.seed)
  dgnn.memory_g.__init_memory__(args.seed)

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(dgnn.parameters(), lr=LEARNING_RATE)
  dgnn = dgnn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_mrrs = []
  val_mrrs = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    dgnn.memory_s.__init_memory__(args.seed)
    dgnn.memory_g.__init_memory__(args.seed)

    # Train using only training graph
    dgnn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        dgnn = dgnn.train()
        pos_prob, neg_prob = dgnn(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch)

        loss += criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      dgnn.memory_s.detach_memory()
      dgnn.memory_g.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    dgnn.set_neighbor_finder(full_ngh_finder)

    # Backup memory at the end of training, so later we can restore it and use it for the
    # validation on unseen nodes
    train_memory_backup_s = dgnn.memory_s.backup_memory()
    train_memory_backup_g = dgnn.memory_g.backup_memory()

    val_mrr, val_recall_20, val_recall_50= eval_edge_prediction(model=dgnn, negative_edge_sampler=val_rand_sampler, data=val_data,
                                           n_neighbors=NUM_NEIGHBORS)

    val_memory_backup_s = dgnn.memory_s.backup_memory()
    val_memory_backup_g = dgnn.memory_g.backup_memory()
    # Restore memory we had at the end of training to be used when validating on new nodes.
    # Also backup memory after validation so it can be used for testing (since test edges are
    # strictly later in time than validation edges)
    dgnn.memory_s.restore_memory(train_memory_backup_s)
    dgnn.memory_g.restore_memory(train_memory_backup_g)

    # Validate on unseen nodes
    nn_val_mrr, nn_val_recall_20, nn_val_recall_50 = eval_edge_prediction(model=dgnn,negative_edge_sampler=val_rand_sampler,
                                                 data=new_node_val_data, n_neighbors=NUM_NEIGHBORS)

    # Restore memory we had at the end of validation
    dgnn.memory_s.restore_memory(val_memory_backup_s)
    dgnn.memory_g.restore_memory(val_memory_backup_g)

    new_nodes_val_mrrs.append(nn_val_mrr)
    val_mrrs.append(val_mrr)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_mrrs": val_mrrs,
      "new_nodes_val_aps": new_nodes_val_mrrs,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val mrr: {}, new node val mrr: {}'.format(val_mrr, nn_val_mrr))
    logger.info(
      'val recall 20: {}, new node val recall 20: {}'.format(val_recall_20, nn_val_recall_20))
    logger.info(
      'val recall 50: {}, new node val recall 50: {}'.format(val_recall_50, nn_val_recall_50))

    # Early stopping
    if early_stopper.early_stop_check(val_mrr):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      dgnn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      dgnn.eval()
      break
    else:
      torch.save(dgnn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
    val_memory_backup_s = dgnn.memory_s.backup_memory()
    val_memory_backup_g = dgnn.memory_g.backup_memory()

  ### Test
  dgnn.propagater_g.neighbor_finder = full_ngh_finder
  dgnn.propagater_s.neighbor_finder = full_ngh_finder
  test_mrr, test_recall_20, test_recall_50 = eval_edge_prediction(model=dgnn, negative_edge_sampler=test_rand_sampler,
                                           data=test_data, n_neighbors=NUM_NEIGHBORS)

  dgnn.memory_s.restore_memory(val_memory_backup_s)
  dgnn.memory_g.restore_memory(val_memory_backup_g)

  # Test on unseen nodes
  nn_test_mrr, nn_test_recall_20, nn_test_recall_50 = eval_edge_prediction(model=dgnn, negative_edge_sampler=nn_test_rand_sampler,
                                                 data=new_node_test_data, n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- mrr: {}, recall_20: {}, recall_50:{}'.format(test_mrr, test_recall_20,
                                                                                test_recall_50))
  logger.info(
    'Test statistics: New nodes -- mrr: {}, recall_20: {}, recall_50:{}'.format(nn_test_mrr, nn_test_recall_20,
                                                                                nn_test_recall_50))
  # Save results for this run
  pickle.dump({
    "val_aps": val_mrrs,
    "new_nodes_val_aps": new_nodes_val_mrrs,
    "test_ap": test_mrr,
    "new_node_test_ap": nn_test_mrr,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving DGNN model')
  # Restore memory at the end of validation (save a model which is ready for testing)
  dgnn.memory_s.restore_memory(val_memory_backup_s)
  dgnn.memory_g.restore_memory(val_memory_backup_g)
  torch.save(dgnn.state_dict(), MODEL_SAVE_PATH)
  logger.info('DGNN model saved')
