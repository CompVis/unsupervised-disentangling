import tensorflow as tf

from ops import get_features, part_map_to_mu_L_inv, feat_mu_to_enc, \
    fold_img_with_mu, fold_img_with_L_inv, prepare_pairs, AbsDetJacobian, augm_mu, get_img_slice_around_mu
from utils import define_scope, batch_colour_map, tf_summary_feat_and_parts, part_to_color_map
from transformations import ThinPlateSpline, make_input_tps_param
from architectures import decoder_map, encoder_map, discriminator_patch


class Model:
    def __init__(self, orig_img, arg, tps_param_dic):

        self.arg = arg

        self.train = (self.arg.mode == 'train')
        self.tps_par = tps_param_dic
        self.image_orig = orig_img
        self.encoder = encoder_map[arg.encoder]
        self.img_decoder = decoder_map[arg.decoder]
        self.discriminator = discriminator_patch

        self.image_in, self.image_rec, self.transform_mesh = None, None,None

        self.mu, self.mu_t,  self.stddev_t = None, None, None

        self.volume_mesh, self.features, self.L_inv, self.part_maps = None, None, None, None
        self.encoding_same_id, self.reconstruct_same_id = None, None

        self.heat_mask_l2, self.fold_img_squared = None, None

        # adverserial
        self.adverserial = self.arg.adverserial
        self.t_D, self.t_D_logits = None, None
        self.patches = None

        self.update_ops = None

        self.graph()
        self.optimize
        self.visualize()

    def graph(self):
        with tf.variable_scope("tps"):
            coord, vector = make_input_tps_param(self.tps_par)
            t_images, t_mesh = ThinPlateSpline(self.image_orig, coord, vector, self.arg.in_dim, self.arg.n_c)
            self.image_in, self.image_rec = prepare_pairs(t_images, self.arg.reconstr_dim, self.arg)
            self.transform_mesh = tf.image.resize_images(t_mesh, size=(self.arg.heat_dim, self.arg.heat_dim))
            self.volume_mesh = AbsDetJacobian(self.transform_mesh)

        with tf.variable_scope("encoding"):
            self.part_maps, raw_features = self.encoder(self.image_in, self.train,
                                                        self.arg.n_parts, self.arg.n_features,
                                                        self.arg.nFeat_1,
                                                        self.arg.nFeat_2)

            self.mu, self.L_inv = part_map_to_mu_L_inv(part_maps=self.part_maps, scal=self.arg.L_inv_scal)
            self.features = get_features(raw_features, self.part_maps, slim=True)

        with tf.variable_scope("transform"):
            integrant = tf.squeeze(tf.expand_dims(self.part_maps, axis=-1) * tf.expand_dims(self.volume_mesh, axis=-1))
            self.integrant = integrant / tf.reduce_sum(integrant, axis=[1, 2],
                                                       keepdims=True)

            self.mu_t = tf.einsum('aijk,aijl->akl', self.integrant, self.transform_mesh)
            transform_mesh_out_prod = tf.einsum('aijm,aijn->aijmn', self.transform_mesh, self.transform_mesh)
            mu_out_prod = tf.einsum('akm,akn->akmn', self.mu_t, self.mu_t)
            self.stddev_t = tf.einsum('aijk,aijmn->akmn', self.integrant, transform_mesh_out_prod) - mu_out_prod

            with tf.variable_scope("generation"):
                with tf.variable_scope("encoding"):
                    self.encoding_same_id = feat_mu_to_enc(self.features, self.mu, self.L_inv,
                                                           self.arg.rec_stages, self.arg.part_depths,
                                                           self.arg.feat_slices, n_reverse=2,
                                                           covariance=self.arg.covariance,
                                                           feat_shape=self.arg.average_features_mode,
                                                           heat_feat_normalize=self.arg.heat_feat_normalize,
                                                           static=self.arg.static)

                self.reconstruct_same_id = self.img_decoder(self.encoding_same_id, self.train, self.arg.reconstr_dim, self.arg.n_c)

        if self.adverserial:
            with tf.variable_scope("adverserial_on_patches"):
                flatten_dim = 2 * self.arg.bn * self.arg.n_parts
                part_map_last_layer = self.encoding_same_id[0][:, :, :, :self.arg.part_depths[0]]
                real_patches = get_img_slice_around_mu(tf.concat([self.image_rec, part_map_last_layer], axis=-1),
                                                       self.mu, self.arg.patch_size)
                real_patches = tf.reshape(real_patches,
                                          shape=[flatten_dim, self.arg.patch_size[0], self.arg.patch_size[1], -1])
                fake_patches_same_id = get_img_slice_around_mu(tf.concat(
                    [self.reconstruct_same_id, part_map_last_layer], axis=-1),  self.mu, self.arg.patch_size)
                fake_patches_same_id = tf.reshape(fake_patches_same_id, shape=[flatten_dim, self.arg.patch_size[0],
                                                                               self.arg.patch_size[1], -1])
                self.patches = tf.concat([real_patches, fake_patches_same_id], axis=0)
                self.t_D, self.t_D_logits = self.discriminator(self.patches, train=self.train)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    @define_scope
    def loss(self):
        mu_t_1, mu_t_2 = self.mu_t[:self.arg.bn], self.mu_t[self.arg.bn:]
        stddev_t_1, stddev_t_2 = self.stddev_t[:self.arg.bn], self.stddev_t[self.arg.bn:]
        transform_loss = tf.reduce_mean((mu_t_1 - mu_t_2) ** 2)

        precision_sq = (stddev_t_1 - stddev_t_2) ** 2

        eps = 1e-6
        precision_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(precision_sq, axis=[2, 3]) + eps))

        img_difference = self.reconstruct_same_id - self.image_rec

        if self.arg.L1:
            distance_metric = tf.abs(img_difference)

        else:
            distance_metric = img_difference ** 2

        if self.arg.fold_with_shape:
            fold_img_squared = fold_img_with_L_inv(
                distance_metric, tf.stop_gradient(self.mu), tf.stop_gradient(self.L_inv), self.arg.l_2_scal,
                visualize=False, threshold=self.arg.l_2_threshold, normalize=True)
        else:
            fold_img_squared, self.heat_mask_l2 = fold_img_with_mu(distance_metric, self.mu, self.arg.l_2_scal,
                                                                   visualize=False,threshold=self.arg.l_2_threshold,
                                                                   normalize=True)

        self.fold_img_squared = fold_img_squared
        tf.summary.image(name="l2_loss", tensor=fold_img_squared, max_outputs=4, family="reconstr")
        l2_loss = tf.reduce_mean(tf.reduce_sum(fold_img_squared, axis=[1, 2]))

        if self.adverserial:
            flatten_dim = 2 * self.arg.bn * self.arg.n_parts
            D, D_ = self.t_D[:flatten_dim], self.t_D[flatten_dim:]
            D_logits, D_logits_ = self.t_D_logits[:flatten_dim], self.t_D_logits[flatten_dim:]

            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
            d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
            d_loss = d_loss_real + d_loss_fake
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

        else:
            d_loss, g_loss = tf.constant(0.), tf.constant(0.)

        return transform_loss, precision_loss, l2_loss, d_loss, g_loss

    @define_scope
    def optimize(self):
        transform_loss, precision_loss, l2_loss, d_loss, g_loss = self.loss

        total_loss = self.arg.c_l2 * l2_loss + self.arg.c_trans * transform_loss\
                     + self.arg.c_precision_trans * precision_loss + self.arg.c_g * g_loss

        tf.summary.scalar(name="total_loss", tensor=total_loss)
        tf.summary.scalar(name="l2", tensor=l2_loss)
        tf.summary.scalar(name="transform_loss", tensor=transform_loss)
        tf.summary.scalar(name="precision_loss", tensor=precision_loss)
        if self.adverserial:
            tf.summary.scalar(name="g_loss", tensor=g_loss)
            tf.summary.scalar(name="d_loss", tensor=d_loss)

        tvar = tf.trainable_variables()
        adverserial_vars = [var for var in tvar if 'discriminator' in var.name]
        rest_vars = [var for var in tvar if 'discriminator' not in var.name]

        if self.arg.print_vars:
            if self.adverserial:
                print("adverserial_vars")
                for var in adverserial_vars:
                    print(var)

            print("normal_vars")
            for var in rest_vars:
                print(var)

        with tf.control_dependencies(self.update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.arg.lr)

            optimizer_d = tf.train.AdamOptimizer(learning_rate=self.arg.lr_d)

            if self.adverserial:
                return optimizer.minimize(total_loss, var_list=rest_vars), optimizer_d.minimize(d_loss, var_list=adverserial_vars)
            else:
                return optimizer.minimize(total_loss, var_list=rest_vars)

    def visualize(self):
        tf.summary.image(name="g_reconstr", tensor=self.image_rec, max_outputs=4, family="reconstr")

        normal = part_to_color_map(self.encoding_same_id, self.arg.part_depths, size=self.arg.in_dim)
        normal = normal / (1 + tf.reduce_sum(normal, axis=-1, keepdims=True))
        vis_normal = tf.where(tf.tile(tf.reduce_sum(normal, axis=-1, keepdims=True), [1, 1, 1, 3]) > 0.3, normal,
                              self.image_in)
        heat_mask_l2 = tf.image.resize_images(tf.tile(self.heat_mask_l2, [1, 1, 1, 3]), size=(self.arg.in_dim, self.arg.in_dim))
        vis_normal = tf.where(heat_mask_l2 > self.arg.l_2_threshold, vis_normal, 0.3 * vis_normal)
        tf.summary.image(name="gt_t_1", tensor=vis_normal[:self.arg.bn], max_outputs=4, family="t_1")
        tf.summary.image(name="gt_t_2", tensor=vis_normal[self.arg.bn:], max_outputs=4, family="t_2")
        tf.summary.image(name="part_maps", tensor=batch_colour_map(self.part_maps[:self.arg.bn]), max_outputs=4,
                         family="t_1")
        tf.summary.image(name="part_maps", tensor=batch_colour_map(self.part_maps[self.arg.bn:]), max_outputs=4,
                         family="t_2")
        # tf.summary.image(name="VolumeElement", tensor=self.volume_mesh, max_outputs=4, family="Volume")

        if self.adverserial:
            f_dim = 2 * self.arg.bn * self.arg.n_parts
            with tf.variable_scope("patch_real"):
                tf.summary.image(name="patch_real",
                                 tensor=self.patches[:f_dim, :, :, :self.arg.n_c], max_outputs=4)
            with tf.variable_scope("patch_fake"):
                tf.summary.image(name="fake_same",
                                 tensor=self.patches[f_dim: f_dim + f_dim // 2, :, :, :self.arg.n_c],
                                 max_outputs=4)


        with tf.variable_scope("reconstr_same_id"):
            tf.summary.image(name="same_id_reconstruction", tensor=self.reconstruct_same_id, max_outputs=4,
                             family="reconstr")

        with tf.variable_scope("normal"):
            tf_summary_feat_and_parts(self.encoding_same_id, self.arg.part_depths, visualize_features=False)
