import tensorflow as tf
import numpy as np

from .ACTAN import Flow

class Model(object):
    def __init__(self, hps):
        self.hps = hps
        self.flow = Flow(hps)

    def sample(self, x, b, m):
        B = tf.shape(x)[0]
        d = self.hps.dimension
        N = self.hps.num_samples
        x = tf.tile(tf.expand_dims(x,axis=1),[1,N,1])
        x = tf.reshape(x, [B*N,d])
        b = tf.tile(tf.expand_dims(b,axis=1),[1,N,1])
        b = tf.reshape(b, [B*N,d])
        m = tf.tile(tf.expand_dims(m,axis=1),[1,N,1])
        m = tf.reshape(m, [B*N,d])

        sam = self.flow.inverse(x, b, m)
        sam = tf.reshape(sam, [B,N,d])

        return sam 

    # def build(self, x, y, b, m):
    #     d = self.hps.dimension
    #     batch_size = self.hps.batch_size
    #     diag_mask = np.ones((d, d)) - np.eye(d)
    #     parent_mat = tf.get_variable(
    #         'sparsity_mat', shape=(self.hps.dimension, self.hps.dimension),
    #         initializer=tf.constant_initializer(diag_mask), trainable=True
    #     )
    #     x = tf.reshape(
    #         tf.tile(tf.expand_dims(x, 1), [1, d, 1]), [batch_size*d, -1]
    #     )
    #     # using a boolean operation here kills the gradient for parent_mat in ll term
    #     # but all we need here is to find out what the conditional mask is so why should there be a gradient
    #     # we can instead assume it conditions on everything and do soft thresholding w.r.t to the dag loss
    #     pos_parent_mat = tf.square(parent_mat * diag_mask)
    #     b = tf.cast(tf.tile(pos_parent_mat > self.hps.eps, [batch_size, 1]), tf.float32)
    #     m = tf.cast(tf.tile(pos_parent_mat > self.hps.eps  + tf.eye(d), [batch_size, 1]), tf.float32)
    #     self.log_likel = self.flow.forward(x, b, m)
    #     self.log_likel = tf.reduce_sum(tf.reshape(self.log_likel, [batch_size, d, -1]), 1)

    #     self.x_sam = self.sample(x, b, m)

    #     dag_loss = tf.linalg.trace(tf.linalg.expm(pos_parent_mat)) - d
    #     # loss
    #     self.loss = tf.reduce_mean(-self.log_likel) + self.hps.norm * dag_loss
    #     tf.summary.scalar('loss', self.loss)

    #     # train
    #     self.global_step = tf.train.get_or_create_global_step()
    #     learning_rate = tf.train.inverse_time_decay(
    #         self.hps.lr, self.global_step,
    #         self.hps.decay_steps, self.hps.decay_rate,
    #         staircase=True)
    #     tf.summary.scalar('lr', learning_rate)
    #     optimizer = tf.train.AdamOptimizer(
    #         learning_rate=learning_rate,
    #         beta1=0.9, beta2=0.999, epsilon=1e-08,
    #         use_locking=False, name="Adam")
    #     grads_and_vars = optimizer.compute_gradients(
    #         self.loss, tf.trainable_variables())
    #     grads, vars_ = zip(*grads_and_vars)
    #     if self.hps.clip_gradient > 0:
    #         grads, gradient_norm = tf.clip_by_global_norm(
    #             grads, clip_norm=self.hps.clip_gradient)
    #         gradient_norm = tf.check_numerics(
    #             gradient_norm, "Gradient norm is NaN or Inf.")
    #         tf.summary.scalar('gradient_norm', gradient_norm)
    #     capped_grads_and_vars = zip(grads, vars_)
    #     self.train_op = optimizer.apply_gradients(
    #         capped_grads_and_vars, global_step=self.global_step)

    #     # summary
    #     self.summ_op = tf.summary.merge_all()

    #     # metric
    #     self.metric = self.log_likel

    def build(self, x, y, b, m):
        d = self.hps.dimension
        batch_size = self.hps.batch_size
        parent_mat = tf.get_variable(
            'sparsity_mat', shape=(self.hps.dimension, self.hps.dimension),
            initializer=tf.constant_initializer(np.ones((d, d)) - np.eye(d)), trainable=False
        )
        x = tf.reshape(
            tf.tile(tf.expand_dims(x, 1), [1, d, 1]), [batch_size*d, -1]
        )

        pos_parent_mat = tf.square(parent_mat)
        b = tf.cast(tf.tile(tf.ones((d, d)) - tf.eye(d), [batch_size, 1]), tf.float32)
        m = tf.cast(tf.tile(tf.ones((d, d)), [batch_size, 1]), tf.float32)
        x = x * tf.tile(pos_parent_mat, [batch_size, 1]) * b + x * 1 - b
        self.log_likel = self.flow.forward(x, b, m)
        self.log_likel = tf.reduce_sum(tf.reshape(self.log_likel, [batch_size, d, -1]), 1)

        self.x_sam = self.sample(x, b, m)

        dag_loss = tf.linalg.trace(tf.linalg.expm(pos_parent_mat)) - d
        # loss
        loss = tf.reduce_mean(-self.log_likel) + self.hps.norm * dag_loss
        tf.summary.scalar('loss', loss)

        # train
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.inverse_time_decay(
            self.hps.lr, self.global_step,
            self.hps.decay_steps, self.hps.decay_rate,
            staircase=True)
        tf.summary.scalar('lr', learning_rate)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.9, beta2=0.999, epsilon=1e-08,
            use_locking=False, name="Adam")
        grads_and_vars = optimizer.compute_gradients(
            loss, tf.trainable_variables())
        grads, vars_ = zip(*grads_and_vars)
        if self.hps.clip_gradient > 0:
            grads, gradient_norm = tf.clip_by_global_norm(
                grads, clip_norm=self.hps.clip_gradient)
            gradient_norm = tf.check_numerics(
                gradient_norm, "Gradient norm is NaN or Inf.")
            tf.summary.scalar('gradient_norm', gradient_norm)
        capped_grads_and_vars = zip(grads, vars_)

        self.train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=self.global_step)

        self.norm = tf.Variable(self.hps.min_norm, trainable=False)
        self.norm_update = tf.assign(self.norm, tf.minimum(
            self.hps.norm_up * self.norm, self.hps.max_norm
        ))
        # update the beta matrix
        mask_grad = optimizer.compute_gradients(loss, [parent_mat])[0][0]
        self.mask_lr = tf.Variable(self.hps.mask_lr, trainable=False)
        d_mask = parent_mat - mask_grad
        new_mask = tf.sign(d_mask) * tf.maximum(0.0, tf.abs(d_mask) - self.norm)
        self.mask_op = tf.assign(parent_mat, new_mask)

        # summary
        self.summ_op = tf.summary.merge_all()

        # metric
        self.metric = self.log_likel
        self.loss = loss
