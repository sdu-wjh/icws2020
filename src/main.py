import argparse
import numpy as np
from data_loader import load_data
from train import train
import tensorflow as tf

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--f_p', type=int, default=16, help='f_p')
parser.add_argument('--num_history_item', type=int, default=32, help="num_history_item")
parser.add_argument('--num_relation', type=int, default=8, help="num_relation")
parser.add_argument('--num_relation_count', type=int, default=32, help="num_relation_count")
parser.add_argument('--use_avg', type=bool, default=True, help="use_avg")
parser.add_argument('--lambd', type=float, default=0.1, help="lambda")

'''
# default settings for Book-Crossing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
'''

a = tf.constant([1,2,3], dtype=tf.float32)
b = tf.constant([4,5,6], dtype=tf.float32)
# b = tf.constant([[7,8,9]])
# # c = tf.concat([a,b], axis=0)
# d = tf.stack([a,b], axis=1)
print(tf.Session().run(a+b))
print(tf.Session().run(tf.nn.leaky_relu(a, alpha=2.0)))

args = parser.parse_args()

show_loss = False
data_info = load_data(args)

print("num_history_item: " + str(args.num_history_item) +
      "\nnum_relation: " + str(args.num_relation) +
      "\nnum_relation_count: " + str(args.num_relation_count))
train(args, data_info, show_loss)
