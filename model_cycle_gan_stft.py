import numpy as np
import tensorflow as tf
import os
import time
import sys
import numpy.random as random
# from bn import VBN
from sklearn.externals import joblib

from scipy import interpolate
from scipy.signal import decimate, spectrogram
from scipy.signal import butter, lfilter
# from asrunet_loader import LoadData

import sys
import librosa

from generator import *
from discriminator import *
import timeit
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Model(object):
    def __init__(self, name='ReverbGAN'):
        self.name = name

    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, os.path.join(save_path, model_name), global_step=step)

    def load(self, save_path, model_path=None):
        if not os.path.exists(save_path):
            print ('checkpoint dir is not exist')
            return False
        print ('reading checkpoint')
        if model_path is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_path
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print ('restore ', ckpt_name)
        return True


class ReverbGAN(Model):
    def __init__(self, sess, args, infer=False, name='ReverbGAN'):
        super(ReverbGAN, self).__init__(name)
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        self.batch_size = args.batch_size
        self.skips = args.skips
        self.epoch = args.epoch
        self.disc_label_smooth = args.d_label_smooth
        self.z_dim = args.z_dim
        self.z_depth = args.z_depth
        self.deconv_type = args.deconv_type
        self.tfrecords = args.tfrecords
        self.tfrecords_val = args.tfrecords_val
        # use biases or not
        self.bias_downconv = args.bias_downconv
        self.bias_deconv = args.bias_deconv
        self.bias_D_conv = args.bias_D_conv
        self.d_clip_weights = False
        # apply VBN or regular BN?
        self.disable_vbn = False
        self.save_path = args.save_path
        # num of updates to be applied to D before G
        # this is k in original GAN paper (https://arxiv.org/abs/1406.2661)
        self.disc_updates = args.disc_updates
        # use pre emphsis or not
        self.pre_emphasis = args.pre_emphasis
        if self.pre_emphasis > 0:
            print ('apply pre-emphasis: ',self.pre_emphasis)
        else:
            print ('No pre-emphasis')

        self.canvas_size = args.canvas_size
        self.de_activated_noise = False

        self.keep_prob_var_valid = tf.Variable(self.keep_prob, trainable=False)
        self.keep_prob = 0.5
        self.keep_prob_var_train = tf.Variable(self.keep_prob, trainable=False)

        self.save_path = args.save_path
        # update ratio d and g
        self.generator_dilated_blocks = [1, 2 ,4 ,8 ,16 ,32 ,64 ,128 ,256]
        self.generator_encoder_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512]
        self.discriminator_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512]
        # noise_std
        self.init_noise_std = args.init_noise_std
        self.discriminator_noise_std = tf.Variable(self.init_noise_std, trainable=False)
        self.discriminator_noise_std_summ = scalar_summary('discriminator_noise_std', self.discriminator_noise_std)

        # generator's l2 loss
        self.l2_weight = args.l1_weight
        self.l2_weight = args.l1_weight
        self.l2_lambda = tf.Variable(self.l2_weight, trainable=False)
        self.d_weight = args.d_weight
        self.gan_lambda = tf.Variable(self.d_weight, trainable=False)
        self.de_activated_l2 = False
        self.discriminator = discriminator
        self.g_activation = args.g_activation
        ''''
        if args.g_type == 'com':
            self.generator = Generator_com(self)
        elif args.g_type == 'dec':
            self.generator = Generator_dec(self)
        '''
        self.generator = generator
        self.generator_for_loss = generator 
        self.build_model(args)

    def build_model(self, config):
        all_d_grads = []
        all_g_grads = []
        # d_opt = tf.train.RMSPropOptimizer(config.d_lr)
        # g_opt = tf.train.RMSPropOptimizer(config.g_lr)


        self.build_model_2()
        d_opt = tf.train.AdamOptimizer(config.d_lr, 0.9, 0.999)
        self.d_opt = d_opt.minimize(self.d_losses[-1], var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(config.g_lr, 0.9, 0.999)
        self.g_opt = g_opt.minimize(self.g_losses[-1], var_list=self.g_vars)

        '''
        g_grads = g_opt.compute_gradients(self.g_losses[-1], var_list=self.g_vars)
        all_g_grads.append(g_grads)
        #tf.get_variable_scope().reuse_variables()
        avg_g_grads = average_gradients(all_g_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)


        d_grads = d_opt.compute_gradients(self.d_losses[-1], var_list=self.d_vars)
        all_d_grads.append(d_grads)
        avg_d_grads = average_gradients(all_d_grads)
        self.d_opt = d_opt.apply_gradients(avg_d_grads)
        '''

    def emphasis(self, x, coeff=0.95):
        x0 = tf.reshape(x[0], [1,])
        diff = x[1:] - coeff * x[:-1]
        concat = tf.concat([x0, diff], axis=0)
        return concat

      
       
    def get_spectrum(self, x, n_fft=1024):
        # S = librosa.stft(x, n_fft)
        # p = np.angle(S)
        S = np.log(np.abs(x)+0.0001)
        return S

    def inv_magphase(self, mag, phase):
        phase = np.cos(phase) + 1.j * np.sin(phase)
        return mag * phase

    def parser(self, record):
        keys_to_features = {
            'X_reverb': tf.FixedLenFeature([], tf.string),
            'Y_origin': tf.FixedLenFeature([], tf.string),
            'p_reverb': tf.FixedLenFeature([], tf.string),
            'p_origin': tf.FixedLenFeature([], tf.string),
            'detail_label': tf.FixedLenFeature([], tf.string),            

        }
        parsed = tf.parse_single_example(record, keys_to_features)

        rev = tf.decode_raw(parsed['X_reverb'], tf.float32)
        ori = tf.decode_raw(parsed['Y_origin'], tf.float32)
        phase = tf.decode_raw(parsed['p_reverb'], tf.float32)
        phase2 = tf.decode_raw(parsed['p_origin'], tf.float32)
        label = tf.decode_raw(parsed['detail_label'], tf.float32)

        # rev.set_shape(self.canvas_size)
        # rev.set_shape((513*128*1))
        # ori.set_shape((513*128*1))
        # phase.set_shape((513*128*1))
        
        #high = (2./32767.) * tf.cast((high-16383.), tf.float32) + 1.

        return rev, ori, phase, phase2, label


    def parser_label(self, record):
        keys_to_features = {
            'detail_label': tf.FixedLenFeature([], tf.string),            

        }
        parsed = tf.parse_single_example(record, keys_to_features)

        label = tf.decode_raw(parsed['detail_label'], tf.float32)

        # rev.set_shape(self.canvas_size)
        # rev.set_shape((513*128*1))
        # ori.set_shape((513*128*1))
        # phase.set_shape((513*128*1))
        
        #high = (2./32767.) * tf.cast((high-16383.), tf.float32) + 1.

        return label

    def my_tf_round(self, x, decimals = 1):
        multiplier = tf.constant(10**decimals, dtype=x.dtype)
        return ((tf.round(x * multiplier) / multiplier) +3.14)/6.28

    def build_model_2(self, gpu_idx=0):
        if gpu_idx==0:

            # datasetl = tf.data.TFRecordDataset('../data/detail_label_clean.tfrecords')
            # datasetl = datasetl.map(self.parser_label)
            # # dataset = dataset.shuffle(buffer_size=5000)
            # datasetl = datasetl.repeat(self.epoch*10000)
            # datasetl = datasetl.batch(self.batch_size)
            # iterator_l = datasetl.make_one_shot_iterator()
            # label_B = iterator_l.get_next()
            
            dataset = tf.data.TFRecordDataset(self.tfrecords)
            dataset = dataset.map(self.parser)
            # dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.repeat(self.epoch)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            rev_train_, ori_train, rev_phase_train_, ori_phase_train_, _ = iterator.get_next()

            dataset2 = tf.data.TFRecordDataset(self.tfrecords)
            dataset2 = dataset2.map(self.parser)
            dataset2 = dataset2.shuffle(buffer_size=1000)
            dataset2 = dataset2.repeat(self.epoch)
            dataset2 = dataset2.batch(self.batch_size)
            iterator2 = dataset2.make_one_shot_iterator()
            rev_train, ori_train_, rev_phase_train, ori_phase_train, _ = iterator2.get_next()

            datasetv = tf.data.TFRecordDataset(self.tfrecords_val)
            datasetv = datasetv.map(self.parser)
            # datasetv = datasetv.shuffle(buffer_size=10000)
            datasetv = datasetv.repeat(self.epoch*50)
            datasetv = datasetv.batch(self.batch_size)
            iteratorv = datasetv.make_one_shot_iterator()
            rev_valid, ori_valid, rev_phase_valid, ori_phase_valid, _ = iteratorv.get_next()

            self.is_train = tf.placeholder(dtype=bool, shape=())
            self.is_valid = tf.placeholder(dtype=bool, shape=())
            self.is_mismatch = tf.placeholder(dtype=bool, shape=())
            revt, orit, rev_phase, ori_phase, self.keep_prob_var = tf.cond(self.is_valid, lambda: [rev_valid, ori_valid, rev_phase_valid, ori_phase_valid, self.keep_prob_var_valid], lambda: [rev_train, ori_train, rev_phase_train, ori_phase_train, self.keep_prob_var_train])
            revt, orit, rev_phase, ori_phase, self.keep_prob_var = tf.cond(self.is_mismatch, lambda: [rev_valid, orit, rev_phase_valid, ori_phase, self.keep_prob_var_valid], lambda: [revt, orit, rev_phase, ori_phase, self.keep_prob_var])
            revt = tf.reshape(revt, (self.batch_size, 513, 128, 1))
            orit = tf.reshape(orit, (self.batch_size, 513, 128, 1))
            rev_phase = tf.reshape(rev_phase, (self.batch_size, 513, 128, 1))
            ori_phase = tf.reshape(ori_phase, (self.batch_size, 513, 128, 1))


            revt = revt[:,:512,:,:]
            orit = orit[:,:512,:,:]
            rev_phase = rev_phase[:,:512,:,:]
            ori_phase = ori_phase[:,:512,:,:]

            # self.label_A, self.label_B = label_A, label_B

            self.rev_phase_ = []
            self.ori_phase_ = []

            self.GG_A = []
            self.GG_B = []
            self.GG_phase = []
            self.zz_A = []
            self.zz_B = []
            self.gt_high = []
            self.gt_low = []
            self.rev_phases = []
            self.ori_phases = []

            # revt = tf.convert_to_tensor(revt, dtype=tf.float32)
            # orit = tf.convert_to_tensor(orit, dtype=tf.float32)
            self.revt, self.orit, self.rev_phase, self.ori_phase = revt, orit, rev_phase, ori_phase

            self.rev_phase_.append(rev_phase)
            self.ori_phase_.append(ori_phase)
            self.gt_high.append(orit)
            self.gt_low.append(revt)
            self.rev_phases.append(rev_phase)
            self.ori_phases.append(ori_phase)

            self.g_losses = []
            self.g_losses_AB = []           
            self.g_losses_BA = []
            self.g_l1_losses_ABA = []
            self.g_l1_losses_BAB = []
            self.g_adv_losses = []

            self.d_A_losses = []
            self.d_B_losses = []
            self.d_losses = []

            dummy_input = tf.concat([orit, revt], axis=-1)
            # dummy_input = tf.add(orit, revt)
            dummy = discriminator(self, dummy_input, reuse=False)

            self.fake_AB, _ = self.generator(self, revt, reuse=False, spk=None, name='g_AB')
            self.fake_ABA, _ = self.generator(self, self.fake_AB, reuse=False ,spk=None, name='g_BA')
            self.fake_BA, _ = self.generator(self, orit, reuse=True, spk=None, name='g_BA')
            self.fake_BAB, _ = self.generator(self, self.fake_BA, reuse=True, spk=None, name='g_AB')

            self.fake_disc_A = self.discriminator(self, self.fake_BA, reuse=False, name='d_A')
            self.fake_disc_B = self.discriminator(self, self.fake_AB, reuse=False, name='d_B')

            self.real_disc_A = self.discriminator(self, revt, reuse=True, name='d_A')
            self.real_disc_B = self.discriminator(self, orit, reuse=True, name='d_B')
            '''
            self.g_adv_loss_AB = tf.reduce_mean(tf.abs(tf.square(tf.subtract(self.fake_disc_B, tf.ones_like(self.fake_disc_B))))) \
            self.g_adv_loss_BA = tf.reduce_mean(tf.abs(tf.square(tf.subtract(self.fake_disc_A, tf.ones_like(self.fake_disc_A))))) \
            '''
            # W-gan 
            self.g_adv_loss_AB = -tf.reduce_mean(self.fake_disc_B)
            self.g_adv_loss_BA = -tf.reduce_mean(self.fake_disc_A)


            self.g_l1_loss_AB = tf.reduce_mean(tf.abs(tf.subtract(orit, self.fake_AB)))
            self.g_l1_loss_BA = tf.reduce_mean(tf.abs(tf.subtract(revt, self.fake_BA)))
                        
            self.g_l1_loss_ABA = tf.reduce_mean(tf.abs(tf.subtract(revt, self.fake_ABA)))
            self.g_l1_loss_BAB = tf.reduce_mean(tf.abs(tf.subtract(orit, self.fake_BAB)))
            self.g_adv_loss = self.gan_lambda * (self.g_adv_loss_AB + self.g_adv_loss_BA)/2
            self.g_loss = self.gan_lambda * (self.g_adv_loss_AB + self.g_adv_loss_BA) \
                            + self.l2_lambda * self.g_l1_loss_BAB \
                            + self.l2_lambda * self.g_l1_loss_ABA \
                            # + self.l2_lambda * self.g_l1_loss_AB \
                            # + self.l2_lambda * self.g_l1_loss_BA

            self.fake_disc_A_sample = self.discriminator(self, self.fake_BA, reuse=True, name='d_A')
            self.fake_disc_B_sample = self.discriminator(self, self.fake_AB, reuse=True, name='d_B')

            '''
            self.d_loss_b_real = tf.reduce_mean(tf.abs(tf.square(tf.subtract(self.real_disc_B, tf.ones_like(self.real_disc_B)))))
            self.d_loss_a_real = tf.reduce_mean(tf.abs(tf.square(tf.subtract(self.real_disc_A, tf.ones_like(self.real_disc_A)))))
            self.d_loss_b_fake = tf.reduce_mean(tf.abs(tf.square(tf.subtract(self.fake_disc_B_sample, tf.zeros_like(self.fake_disc_B_sample)))))
            self.d_loss_a_fake = tf.reduce_mean(tf.abs(tf.square(tf.subtract(self.fake_disc_A_sample, tf.zeros_like(self.fake_disc_A_sample)))))
            '''

            # W-gan 
            self.d_loss_b_real = - tf.reduce_mean(self.real_disc_B)
            self.d_loss_a_real = - tf.reduce_mean(self.real_disc_A)
            self.d_loss_b_fake = tf.reduce_mean(self.fake_disc_B_sample)
            self.d_loss_a_fake = tf.reduce_mean(self.fake_disc_A_sample)            
            

            self.d_loss_b = (self.d_loss_b_real + self.d_loss_b_fake)/2
            self.d_loss_a = (self.d_loss_a_real + self.d_loss_a_fake)/2
            self.d_loss = self.d_loss_a + self.d_loss_b

            G_A, z_A = self.generator(self, revt, reuse=True, spk=None, name='g_AB')
            G_B, z_B = self.generator(self, orit, reuse=True, spk=None, name='g_BA')

            self.G_A, self.z_A = G_A, z_A
            self.G_B, self.z_B = G_B, z_B

            self.GG_A.append(self.G_A)
            self.zz_A.append(self.z_A)
            self.GG_B.append(self.G_B)
            self.zz_B.append(self.z_B)

            # gradient penalty
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat_A = revt*epsilon + (1-epsilon)*self.fake_BA
            d_hat_A = self.discriminator(self, x_hat_A, reuse=True, name='d_A')
            gradients = tf.gradients(d_hat_A, x_hat_A)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gradient_penalty_A = 3*tf.reduce_mean((slopes-1.0)**2)
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat_B = orit*epsilon + (1-epsilon)*self.fake_AB
            d_hat_B = self.discriminator(self, x_hat_B, reuse=True, name='d_B')
            gradients = tf.gradients(d_hat_B, x_hat_B)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gradient_penalty_B = 3*tf.reduce_mean((slopes-1.0)**2)
            self.d_loss = self.d_loss + gradient_penalty_A + gradient_penalty_B
            self.d_loss = self.d_loss*self.gan_lambda

            self.g_losses_AB.append(self.g_l1_loss_AB)
            self.g_losses_BA.append(self.g_l1_loss_BA)
            self.g_l1_losses_ABA.append(self.g_l1_loss_ABA)
            self.g_l1_losses_BAB.append(self.g_l1_loss_BAB)   
            self.g_adv_losses.append(self.g_adv_loss)         
            self.g_losses.append(self.g_loss)

            self.d_A_losses.append(self.d_loss_a)
            self.d_B_losses.append(self.d_loss_b)
            self.d_losses.append(self.d_loss)


            # self.d_real_loss_summary = scalar_summary("d_real_loss", d_real_loss)
            # self.d_fake_loss_summary = scalar_summary("d_fake_loss", d_fake_loss)
            # self.g_loss_summary = scalar_summary("g_loss", g_loss)
            # self.g_l2_loss_summary = scalar_summary("g_l2_loss", g_l2_loss)
            # self.g_loss_adv_summary = scalar_summary("g_adv_loss", g_adv_loss)
            # self.d_loss_summary = scalar_summary("d_loss", d_loss)

            self.get_vars()
    
    # def vbn(self, tensor, name):
    #     if self.disable_vbn:
    #         class Dummy(object):
    #             # Do nothing here, no bnorm
    #             def __init__(self, tensor, ignored):
    #                 self.reference_output=tensor
    #             def __call__(self, x):
    #                 return x
    #         VBN_cls = Dummy
    #     else:
    #         VBN_cls = VBN
    #     if not hasattr(self, name):
    #         vbn = VBN_cls(tensor, name)
    #         setattr(self, name, vbn)
    #         return vbn.reference_output
    #     vbn = getattr(self, name)
    #     return vbn(tensor)

    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars_dict = {}
        self.g_vars_dict = {}
        for var in t_vars:
            if var.name.startswith('d_'):
                self.d_vars_dict[var.name] = var
            if var.name.startswith('g_'):
                self.g_vars_dict[var.name] = var
        self.d_vars = list(self.d_vars_dict.values())
        self.g_vars = list(self.g_vars_dict.values())

        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in t_vars:
            assert x in self.g_vars or x in self.d_vars, x.name
        self.all_vars = t_vars
        if self.d_clip_weights:
            print('Clipping D weights')
            self.d_clip = [v.assign(tf.clip_by_value(v, -0.5, 0.5)) for v in self.d_vars]
        else:
            print('Not clipping D weights')

    def de_emphasis(self, y, coeff=0.95):
        xx = np.zeros(y.shape, dtype=np.float32)
        for nn in range(y.shape[0]):
            xx[nn][0] = y[nn][0]
            for n in range(1, y.shape[1], 1):
                xx[nn][n] = coeff * xx[nn][n-1] + y[nn][n]
        return xx


    def train(self, sess, config):
        """ Training the GAN """
        print ('initializing...opt')
        d_opt = self.d_opt
        g_opt = self.g_opt

        try:
            init = tf.global_variables_initializer()
            sess.run(init)
        except AttributeError:
            init = tf.intializer_all_varialble()
            sess.run(init)

        print ('initializing...var')
        # g_summaries = [self.d_fake_summary,
        #                 self.d_fake_loss_summary,
        #                 self.g_loss_summary,
        #                 self.g_l2_loss_summary,
        #                 self.g_loss_adv_summary,
        #                 self.generated_wav_summary]
        # d_summaries = [self.d_loss_summary, self.d_real_summary, self.d_real_loss_summary, self.high_wav_summary]

        # if hasattr(self, 'alpha_summ'):
        #     g_summaries += self.alpha_summ
        # self.g_sum = tf.summary.merge(g_summaries)
        # self.d_sum = tf.summary.merge(d_summaries)

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path, 'train'), self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sample_low, sample_high, sample_z = self.sess.run([self.gt_low[0], self.gt_high[0], self.zz_A[0]], feed_dict={self.is_valid:False, self.is_train:True, self.is_mismatch:False})
        v_sample_low, v_sample_high, v_sample_z = self.sess.run([self.gt_low[0], self.gt_high[0], self.zz_A[0]],
                                                          feed_dict={self.is_valid: True, self.is_train: False, self.is_mismatch:False})

        print ('sample low shape: ', sample_low.shape)
        print ('sample high shape: ', sample_high.shape)
        print ('sample z shape: ', sample_z.shape)

        save_path = config.save_path
        counter = 0
        # count of num of samples
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.tfrecords):
            num_examples += 1
        print ("total num of patches in tfrecords", self.tfrecords,":  ", num_examples)

        # last samples 
        # batch num
        num_batches = num_examples / self.batch_size
        print ('batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print ('load success')
        else:
            print ('load failed')
        batch_idx = 0
        current_epoch = 0
        batch_timings = []
        g_losses = []
        d_A_losses = []
        d_B_losses = []
        g_adv_losses = []
        g_l1_losses_BAB = []
        g_l1_losses_AB = []
        g_l1_losses_ABA = []
        g_l1_losses_BA = []


        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                if counter % config.save_freq == 0:

                    for d_iter in range(self.disc_updates):
                        _d_opt, d_A_loss, d_B_loss = self.sess.run([d_opt, self.d_A_losses[0], self.d_B_losses[0]], feed_dict={self.is_valid:False, self.is_train: True, self.is_mismatch:True})
                        _d_opt, d_A_loss, d_B_loss = self.sess.run([d_opt, self.d_A_losses[0], self.d_B_losses[0]], feed_dict={self.is_valid:False, self.is_train: True, self.is_mismatch:False})

                        #_d_sum, d_fake_loss, d_real_loss = self.sess.run(
                        #   [self.d_sum, self.d_fake_losses[0], self.d_real_losses[0]], feed_dict={self.is_valid: False})

                        if self.d_clip_weights:
                            self.sess.run(self.d_clip, feed_dict={self.is_valid:False,self.is_train: True})

                    #_g_opt, _g_sum, g_adv_loss, g_l2_loss = self.sess.run([g_opt, self.g_sum, self.g_adv_losses[0], self.g_l2_losses[0]], feed_dict={self.is_valid:False})
                    _g_opt, g_adv_loss, g_AB_loss, g_BA_loss, g_ABA_loss, g_BAB_loss = self.sess.run([g_opt, self.g_adv_losses[0], self.g_losses_AB[0], self.g_losses_BA[0],self.g_l1_losses_ABA[0], self.g_l1_losses_BAB[0]], feed_dict={self.is_valid:False,self.is_train: True, self.is_mismatch:True})                    
                    _g_opt, g_adv_loss, g_AB_loss, g_BA_loss, g_ABA_loss, g_BAB_loss = self.sess.run([g_opt, self.g_adv_losses[0], self.g_losses_AB[0], self.g_losses_BA[0],self.g_l1_losses_ABA[0], self.g_l1_losses_BAB[0]], feed_dict={self.is_valid:False,self.is_train: True, self.is_mismatch:False})
                    # _phase_opt, phase_loss = self.sess.run([phase_opt, self.phase_losses[0]], feed_dict={self.is_valid:False,self.is_train: True})

                else:
                    for d_iter in range(self.disc_updates):
                        _d_opt, d_A_loss, d_B_loss = self.sess.run([d_opt, self.d_A_losses[0], self.d_B_losses[0]], feed_dict={self.is_valid:False,self.is_train: True, self.is_mismatch:True})                        
                        _d_opt, d_A_loss, d_B_loss = self.sess.run([d_opt, self.d_A_losses[0], self.d_B_losses[0]], feed_dict={self.is_valid:False,self.is_train: True, self.is_mismatch:False})
                        #d_fake_loss, d_real_loss = self.sess.run(
                        #    [self.d_fake_losses[0], self.d_real_losses[0]], feed_dict={self.is_valid: False})
                        if self.d_clip_weights:
                            self.sess.run(self.d_clip, feed_dict={self.is_valid:False,self.is_train: True})
                    _g_opt, g_adv_loss, g_AB_loss, g_BA_loss, g_ABA_loss, g_BAB_loss = self.sess.run([g_opt, self.g_adv_losses[0], self.g_losses_AB[0], self.g_losses_BA[0],self.g_l1_losses_ABA[0], self.g_l1_losses_BAB[0]], feed_dict={self.is_valid:False,self.is_train: True, self.is_mismatch:True})                            
                    _g_opt, g_adv_loss, g_AB_loss, g_BA_loss, g_ABA_loss, g_BAB_loss = self.sess.run([g_opt, self.g_adv_losses[0], self.g_losses_AB[0], self.g_losses_BA[0],self.g_l1_losses_ABA[0], self.g_l1_losses_BAB[0]], feed_dict={self.is_valid:False,self.is_train: True, self.is_mismatch:False})
                    # _phase_opt, phase_loss = self.sess.run([phase_opt, self.phase_losses[0]], feed_dict={self.is_valid:False,self.is_train: True})

                end = timeit.default_timer()
                batch_timings.append(end - start)
                d_A_losses.append(d_A_loss)
                d_B_losses.append(d_B_loss)
                g_adv_losses.append(g_adv_loss)
                g_l1_losses_BAB.append(g_BAB_loss) # clean - reverb - clean
                g_l1_losses_AB.append(g_AB_loss) # reverb - clean
                g_l1_losses_ABA.append(g_ABA_loss) # reverb - clean  - reverb
                g_l1_losses_BA.append(g_BA_loss) # clean - reverb


                print('{}/{} (epoch {}), d_A_loss = {:.5f}, '
                      'd_B_loss = {:.5f}, '#d_nfk_loss = {:.5f}, '
                      'g_adv_loss = {:.5f}, g_AB_loss = {:.5f}, g_BAB_loss = {:.5f}, '
                      'g_BA_loss = {:.5f}, g_ABA_loss = {:.5f}, '                      
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    current_epoch,
                                                    d_A_loss,
                                                    d_B_loss,
                                                    g_adv_loss,
                                                    g_AB_loss,
                                                    g_BAB_loss,
                                                    g_BA_loss,
                                                    g_ABA_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += 1
                counter += 1
                
                if (counter) % 2000 == 0 and (counter) > 0:
                    self.save(config.save_path, counter)

                if (counter % config.save_freq == 0) or (counter==1):
                    # self.writer.add_summary(_g_sum, counter)
                    # self.writer.add_summary(_d_sum, counter)
                    #feed_dict = {self.gt_high[0]:v_sample_high, self.gt_low[0]:v_sample_low, self.zz[0]:v_sample_z, self.is_valid:True}

                    s_A, s_B, s_reverb, s_gt, r_phase, f_phase = self.sess.run([self.GG_A[0][0,:,:,:],self.GG_B[0][0,:,:,:], self.gt_low[0][0,:,:,:], self.gt_high[0][0,:,:,:], self.ori_phase_[0][0,:,:,:], self.rev_phase_[0][0,:,:,:]], feed_dict={self.is_valid:True,self.is_train: False, self.is_mismatch:False})

                    if not os.path.exists(save_path+'/wav'):
                        os.makedirs(save_path + '/wav')
                    if not os.path.exists(save_path + '/txt'):
                        os.makedirs(save_path + '/txt')
                    if not os.path.exists(save_path + '/spec'):
                        os.makedirs(save_path + '/spec')

                    print (str(counter)+'th finished')

                    x_AB = s_A
                    x_BA = s_B
                    x_reverb = s_reverb
                    x_gt = s_gt

                    Sre = self.get_spectrum(x_reverb).reshape(512,128)
                    Sgt = self.get_spectrum(x_gt).reshape(512,128)
                    SAB = self.get_spectrum(x_AB).reshape(512,128)
                    SBA = self.get_spectrum(x_BA).reshape(512,128)
                    S = np.concatenate((Sre, Sgt, SAB, SBA), axis=1)
                    fig = Figure(figsize=S.shape[::-1], dpi=1, frameon=False)
                    canvas = FigureCanvas(fig)
                    fig.figimage(S, cmap='jet')
                    fig.savefig(save_path + '/spec/' + 'valid_batch_index' + str(counter) + '-th_pr.png')

                    x_pr = librosa.istft(self.inv_magphase(s_A, f_phase))
                    librosa.output.write_wav(save_path + '/wav/'+str(counter)+'_AB(dereverb).wav', x_pr, 16000)                    
                    x_pr = librosa.istft(self.inv_magphase(s_B, r_phase))
                    librosa.output.write_wav(save_path + '/wav/'+str(counter)+'_BA(reverb).wav', x_pr, 16000)                    
                    x_lr = librosa.istft(self.inv_magphase(s_reverb, f_phase))
                    librosa.output.write_wav(save_path + '/wav/'+str(counter)+'_reverb.wav', x_lr, 16000)
                    x_hr = librosa.istft(self.inv_magphase(s_gt, r_phase))
                    librosa.output.write_wav(save_path + '/wav/'+str(counter)+'_orig.wav', x_hr, 16000)

                    s_AB, s_BA, s_reverb, s_gt = self.sess.run([self.GG_A[0][0,:,:,:],self.GG_B[0][0,:,:,:],self.gt_low[0][0,:,:,:], self.gt_high[0][0,:,:,:]], feed_dict={self.is_valid:False, self.is_train: True, self.is_mismatch:False})


                    x_AB = s_AB
                    x_BA = s_BA
                    x_reverb = s_reverb
                    x_gt = s_gt

                    Sre = self.get_spectrum(x_reverb).reshape(512,128)
                    Sgt = self.get_spectrum(x_gt).reshape(512,128)
                    SAB = self.get_spectrum(x_AB).reshape(512,128)
                    SBA = self.get_spectrum(x_BA).reshape(512,128)

                    S = np.concatenate((Sre, Sgt, SAB, SBA), axis=1)
                    fig = Figure(figsize=S.shape[::-1], dpi=1, frameon=False)
                    canvas = FigureCanvas(fig)
                    fig.figimage(S, cmap='jet')
                    fig.savefig(save_path + '/spec/' + 'train_batch_index' + str(counter) + '-th_pr.png')

                    
                        #np.savetxt(os.path.join(save_path, '/txt/d_real_losses.txt'), d_real_losses)
                        #np.savetxt(os.path.join(save_path, '/txt/d_fake_losses.txt'), d_fake_losses)
                        #np.savetxt(os.path.join(save_path, '/txt/g_adv_losses.txt'), g_adv_losses)
                        #np.savetxt(os.path.join(save_path, '/txt/g_l2_losses.txt'), g_l2_losses)

                if batch_idx >= num_batches:
                    current_epoch += 1
                    #reset batch idx
                    batch_idx = 0

                if current_epoch >= config.epoch:
                    print (str(self.epoch),': epoch limit')
                    print ('saving last model at iteration',str(counter))
                    self.save(config.save_path, counter)
                    # self.writer.add_summary(_g_sum, counter)
                    # self.writer.add_summary(_d_sum, counter)
                    break
        
        except tf.errors.InternalError:
            print ('InternalError')
            pass

        except tf.errors.OutOfRangeError:
            print('done training')
            pass
        finally:
            coord.request_stop()
        coord.join(threads)

    # def infer(self, x):
    #     ''' inference the high resolution signal from low resolution signal'''
    #     h_res = None
    #     canvas_w = self.sess.run(self.GG[0], feed_dict={self.is_valid:True})[0]
    #     canvas_w = canvas_w.reshape((self.canvas_size))
    #     h_res = self.de_emphasis(canvas_w, self.pre_emphasis)

    #     return h_res

