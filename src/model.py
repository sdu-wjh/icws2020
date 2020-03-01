import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


class RippleNet(object):
    def __init__(self, args, n_entity, n_relation):
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.f_p = args.f_p
        self.num_history_item = args.num_history_item
        self.num_relation = args.num_relation
        self.num_relation_count = args.num_relation_count
        self.use_avg = args.use_avg
        self.lambd = args.lambd

    def _build_inputs(self):
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")

        # [batch size, num]
        # self.input_items = tf.placeholder(dtype=tf.int32, shape=[None, self.num], name="input_items")

        # self.memories_h = []
        # self.memories_r = []
        # self.memories_t = []
        #
        # for hop in range(self.n_hop):
        #     self.memories_h.append(
        #         tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
        #     self.memories_r.append(
        #         tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
        #     self.memories_t.append(
        #         tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

        self.inter_item_list = []
        for i in range(self.num_history_item):
            self.relation_catagory_list = []
            for j in range(self.num_relation):
                # [batch size, num_relation_count]
                head = tf.placeholder(dtype=tf.int32,
                                      shape=[None],
                                      name="head_" + str(i) + str(j) + "0")
                tail_list = tf.placeholder(dtype=tf.int32,
                                           shape=[None, self.num_relation_count],
                                           name="tail_" + str(i) + str(j) + "1")
                self.relation_catagory_list.append((head, tail_list))
            self.inter_item_list.append(self.relation_catagory_list)

    def _build_embeddings(self):
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

    def _gat(self, head_embeddings, tail_list_embeddings, w_matrix, a_vector, num):
        # ====gat====
        # [batch size, f_p] = [batch size, dim] * [dim, f_p]
        hi = tf.matmul(head_embeddings, w_matrix)
        # [batch size, num, f_p]
        hi_extended = tf.tile(tf.expand_dims(hi, axis=1), multiples=[1, num, 1])
        # [batch size, dim, f_p]
        w_matrix_extended = tf.tile(tf.expand_dims(w_matrix, axis=0), multiples=[1024, 1, 1])
        # [batch size, num, f_p] = [batch size, num, dim] * [batch size, dim, f_p]
        hj = tf.matmul(tail_list_embeddings, w_matrix_extended)
        # [batch size, num, 2*f_p]
        hij = tf.concat([hi_extended, hj], axis=2)
        # [batch size, 2*f_p, 1]
        a_vector_extended = tf.tile(tf.expand_dims(a_vector, axis=0), multiples=[1024, 1, 1])
        # [batch size, num] = [batch size, num, 2*f_p] * [batch size, 2*f_p, 1]
        ha = tf.squeeze(tf.matmul(hij, a_vector_extended))
        # [batch size, num]
        e = tf.nn.leaky_relu(tf.cast(ha, dtype=tf.float32))
        # [batch size, num]
        a = tf.nn.softmax(tf.squeeze(e))
        # [batch size, f_p] = [batch size, 1, num] * [batch size, num, f_p]
        # temp = tf.squeeze(
        #     tf.matmul(tf.expand_dims(tf.cast(a, dtype=tf.float64), axis=1), hj), axis=1)
        # return tf.sigmoid(temp)
        return tf.squeeze(
            tf.matmul(tf.expand_dims(tf.cast(a, dtype=tf.float64), axis=1), tail_list_embeddings), axis=1)

    def _build_model(self):
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.lambd)

        # [batch size, dim]
        # self.self_item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)
        # self.item_embeddings_v = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        # [dim, f_p]
        self.w_matrix_1 = tf.get_variable(name="w_matrix_1", shape=[self.dim, self.f_p], dtype=tf.float64,
                                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_regularizer)
        # [f_p*2, 1]
        self.a_vector_1 = tf.get_variable(name="a_vector_1", shape=[self.f_p * 2, 1], dtype=tf.float64,
                                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_regularizer)
        # [dim, f_p]
        self.w_matrix_2 = tf.get_variable(name="w_matrix_2", shape=[self.dim, self.f_p], dtype=tf.float64,
                                          initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_regularizer)
        # [f_p*2, 1]
        self.a_vector_2 = tf.get_variable(name="a_vector_2", shape=[self.f_p * 2, 1], dtype=tf.float64,
                                          initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_regularizer)

        # [self.num_relation * self.dim, self.dim]
        self.w_relation = tf.get_variable(name="w_relation", shape=[self.num_relation * self.dim, self.dim],
                                        dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(), regularizer=l2_regularizer)

        enriched_inter_item_embeddings_list = []
        for i in range(self.num_history_item):
            temp = []
            for j in range(self.num_relation):
                head_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.inter_item_list[i][j][0])
                tail_list_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.inter_item_list[i][j][1])
                gat_head_embeddings = self._gat(head_embeddings, tail_list_embeddings, self.w_matrix_1, self.a_vector_1, self.num_relation_count)
                temp.append(gat_head_embeddings)
            gat_head_embeddings = temp[0]
            for i in range(1, len(temp)):
                gat_head_embeddings = tf.concat((gat_head_embeddings, temp[i]), axis=1)

            # [batch size, 1, num_relation * dim]
            extended_gat_head_embeddings = tf.expand_dims(gat_head_embeddings, axis=1)

            # [batch size, num_relation * dim, dim]
            self.w_relation_extended = tf.tile(tf.expand_dims(self.w_relation, axis=0), multiples=[1024, 1, 1])

            # [batch size, dim] = [batch size, 1, num_relation * dim] * [batch size, num_relation * dim, dim]
            enriched_one_inter_item_embeddings = tf.squeeze(tf.matmul(extended_gat_head_embeddings, self.w_relation_extended))
            enriched_inter_item_embeddings_list.append(enriched_one_inter_item_embeddings)

        if self.use_avg:
            # 直接加权平均
            enriched_inter_item_embeddings = enriched_inter_item_embeddings_list[0]
            for i in range(1, len(enriched_inter_item_embeddings_list)):
                enriched_inter_item_embeddings = enriched_inter_item_embeddings + enriched_inter_item_embeddings_list[i]
            enriched_inter_item_embeddings = enriched_inter_item_embeddings / self.num_history_item
        else:
            # 使用注意力机制
            # [batch size, 1, dim]
            enriched_inter_item_embeddings = tf.expand_dims(enriched_inter_item_embeddings_list[0], axis=1)
            for i in range(1, len(enriched_inter_item_embeddings_list)):
                enriched_inter_item_embeddings = tf.concat([enriched_inter_item_embeddings,
                                                            tf.expand_dims(enriched_inter_item_embeddings_list[i], axis=1)], axis=1)
            enriched_inter_item_embeddings = self._gat(self.item_embeddings, enriched_inter_item_embeddings, self.w_matrix_2, self.a_vector_2, self.num_history_item)

        self.scores = tf.squeeze(self.predict(self.item_embeddings, enriched_inter_item_embeddings))
        self.scores_normalized = tf.sigmoid(self.scores)

    # def _key_addressing(self):
    #     o_list = []
    #     for hop in range(self.n_hop):
    #         # [batch_size, n_memory, dim, 1]
    #         h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)
    #
    #         # [batch_size, n_memory, dim]
    #         Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)
    #
    #         # [batch_size, dim, 1]
    #         v = tf.expand_dims(self.item_embeddings, axis=2)
    #
    #         # [batch_size, n_memory]
    #         probs = tf.squeeze(tf.matmul(Rh, v), axis=2)
    #
    #         # [batch_size, n_memory]
    #         probs_normalized = tf.nn.softmax(probs)
    #
    #         # [batch_size, n_memory, 1]
    #         probs_expanded = tf.expand_dims(probs_normalized, axis=2)
    #
    #         # [batch_size, dim]
    #         o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)
    #
    #         self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
    #         o_list.append(o)
    #     return o_list

    # def update_item_embedding(self, item_embeddings, o):
    #     if self.item_update_mode == "replace":
    #         item_embeddings = o
    #     elif self.item_update_mode == "plus":
    #         item_embeddings = item_embeddings + o
    #     elif self.item_update_mode == "replace_transform":
    #         item_embeddings = tf.matmul(o, self.transform_matrix)
    #     elif self.item_update_mode == "plus_transform":
    #         item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
    #     else:
    #         raise Exception("Unknown item updating mode: " + self.item_update_mode)
    #     return item_embeddings

    def predict(self, item_embeddings, user_embeddings):
        # [batch_size]
        scores = tf.reduce_sum(item_embeddings * user_embeddings, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
        reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.l2_loss = tf.add_n(reg_set) / 1024

        # self.kge_loss = 0
        # for hop in range(self.n_hop):
        #     h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
        #     t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
        #     hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
        #     self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        # self.kge_loss = -self.kge_weight * self.kge_loss
        #
        # self.l2_loss = 0
        # for hop in range(self.n_hop):
        #     self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
        #     self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
        #     self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
        #     if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
        #         self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        # self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, clip_norm=5)
                     for gradient in gradients]
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables))
        '''

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc
