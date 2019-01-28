import numpy as np
import tensorflow as tf
import os
import time
import sys
import numpy.random as random

from scipy import interpolate
from scipy.signal import decimate, spectrogram
from scipy.signal import butter, lfilter

import sys
import librosa

from generator import *
from discriminator import *
import timeit
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Model(object):
    def __init__(self, name='DEREVERB_GAN'):
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


class DEREVERB_GAN(Model):
    def __init__(self, sess, args, infer=False, name='DEREVERB_GAN'):
        super(DEREVERB_GAN, self).__init__(name)
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.z_dim = args.z_dim
        self.z_depth = args.z_depth
        self.tfrecords = args.tfrecords
        self.tfrecords_val = args.tfrecords_val
        self.d_clip_weights = False
        # apply VBN or regular BN?
        self.disable_vbn = False
        self.save_path = args.save_path
        # num of updates to be applied to D before G
        # this is k in original GAN paper (https://arxiv.org/abs/1406.2661)
        self.disc_updates = 1
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
        d_opt = tf.train.AdamOptimizer(config.d_lr, 0.9, 0.999)
        g_opt = tf.train.AdamOptimizer(config.g_lr, 0.9, 0.999)

        self.build_model_2()
        g_grads = g_opt.compute_gradients(self.g_losses[-1], var_list=self.g_vars)
        all_g_grads.append(g_grads)
        #tf.get_variable_scope().reuse_variables()
        avg_g_grads = average_gradients(all_g_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)


        d_grads = d_opt.compute_gradients(self.d_losses[-1], var_list=self.d_vars)
        all_d_grads.append(d_grads)
        avg_d_grads = average_gradients(all_d_grads)
        self.d_opt = d_opt.apply_gradients(avg_d_grads)


    def emphasis(self, x, coeff=0.95):
        x0 = tf.reshape(x[0], [1,])
        diff = x[1:] - coeff * x[:-1]
        concat = tf.concat([x0, diff], axis=0)
        return concat

      
       
    def get_spectrum(self, x, n_fft=1024):
        S = librosa.stft(x, n_fft)
        p = np.angle(S)
        S = np.log(np.abs(S)+0.0001)
        return S

    def parser(self, record):
        keys_to_features = {
            'X_reverb': tf.FixedLenFeature([], tf.string),
            'Y_origin': tf.FixedLenFeature([], tf.string),
#            'shape': tf.FixedLenFeature([], tf.string),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        rev = tf.decode_raw(parsed['X_reverb'], tf.float32)
        ori = tf.decode_raw(parsed['Y_origin'], tf.float32)
#        shape = tf.decode_raw(parsed['shape'], tf.int32)
        rev.set_shape(self.canvas_size)

        #reverb = (2./32767.) * tf.cast((reverb-16383.), tf.float32) + 1.
        ori.set_shape(self.canvas_size)

        #nonreverb = (2./32767.) * tf.cast((nonreverb-16383.), tf.float32) + 1.

        return rev, ori

    def build_model_2(self, gpu_idx=0):
        if gpu_idx==0:
            dataset = tf.data.TFRecordDataset(self.tfrecords)
            dataset = dataset.map(self.parser)
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat(self.epoch)
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            rev_train, ori_train = iterator.get_next()

            datasetv = tf.data.TFRecordDataset(self.tfrecords_val)
            datasetv = datasetv.map(self.parser)
            # datasetv = datasetv.shuffle(buffer_size=10000)
            datasetv = datasetv.repeat(self.epoch*200000)
            datasetv = datasetv.batch(self.batch_size)
            iteratorv = datasetv.make_one_shot_iterator()
            rev_valid, ori_valid = iteratorv.get_next()

            self.is_train = tf.placeholder(dtype=bool, shape=())
            self.is_valid = tf.placeholder(dtype=bool, shape=())
            revt, orit, self.keep_prob_var = tf.cond(self.is_valid, lambda: [rev_valid, ori_valid, self.keep_prob_var_valid], lambda: [rev_train, ori_train, self.keep_prob_var_train])

            self.GG = []
            self.zz = []
            self.gt_nonreverb = []
            self.gt_reverb = []

            # revt = tf.convert_to_tensor(revt, dtype=tf.float32)
            # orit = tf.convert_to_tensor(orit, dtype=tf.float32)
            self.revt, self.orit = revt, orit

            self.gt_nonreverb.append(orit)
            self.gt_reverb.append(revt)

            revt = tf.expand_dims(revt, -1)
            orit = tf.expand_dims(orit, -1)

            dummy_input = tf.concat([orit, revt], axis=2)
            dummy = discriminator(self, dummy_input, reuse=False)

            G, z = self.generator(self, revt, spk=None)

            self.G, self.z = G, z
            self.GG.append(G)
            self.zz.append(z)

            # add new dim to merge with other pairs
            d_real_joint = tf.concat([orit,revt], axis=2)
            d_fake_joint = tf.concat([G, revt], axis=2)
            # build real disciminator

            d_real_logits = discriminator(self, d_real_joint, reuse=True)
            # build faking G discriminator
            d_fake_logits = discriminator(self, d_fake_joint, reuse=True)
            # make discriminator variable summaries
            self.d_real_summary = histogram_summary('d_real', d_real_logits)
            self.d_fake_summary = histogram_summary('d_fake', d_fake_logits)

            self.nonreverb_audio_summary = audio_summary('nonreverb_audio', orit)
            self.nonreverb_wav_summary = histogram_summary('nonreverb_wav', orit)
            self.reverb_audio_summary = audio_summary('reverb_audio', revt)
            self.reverb_wav_summary = histogram_summary('reverb_wav', revt)
            self.generated_audio_summary = audio_summary('generated_audio', G)
            self.generated_wav_summary = histogram_summary('generated_wav', G)

            self.g_losses = []
            self.g_l2_losses = []
            self.g_adv_losses = []
            self.d_real_losses = []
            self.d_fake_losses = []
            self.d_losses = []

            ''' gan loss 
            d_real_loss = tf.reduce_mean(tf.squared_difference(d_real_logits, 1.))
            d_fake_loss = tf.reduce_mean(tf.squared_difference(d_fake_logits, 0.))
            g_adv_loss = tf.reduce_mean(tf.squared_difference(d_fake_logits, 1.))
            '''

            # wgan
            d_real_loss = -tf.reduce_mean(d_real_logits)
            d_fake_loss = tf.reduce_mean(d_fake_logits)
            g_adv_loss = -tf.reduce_mean(d_fake_logits)

            d_loss = self.gan_lambda * (d_real_loss + d_fake_loss)

            # add l2 loss to G
            # g_l2_loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((G - orit) ** 2 + 1e-6, axis=[1, 2])), axis=0)

            g_l2_loss = tf.reduce_mean(tf.abs(tf.subtract(G, orit)))

            '''
            perceptual_loss_G = tf.contrib.signal.stft(G, frame_length=2048, frame_step=8192, fft_length=2048,
             pad_end=False)
            
            perceptual_loss_GT = tf.contrib.signal.stft(orit, frame_length=2048, frame_step=8192, fft_length=2048,
                pad_end=False)

            p_l1_loss = tf.reduce_mean(tf.abs(tf.subtract(perceptual_loss_G, perceptual_loss_GT)))
            '''

            g_loss = self.gan_lambda * g_adv_loss + self.l2_lambda * g_l2_loss
            
            # gradient penalty
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = d_real_joint*epsilon + (1-epsilon)*d_fake_joint
            d_hat = discriminator(self, x_hat, reuse=True)
            gradients = tf.gradients(d_hat, x_hat)[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gradient_penalty = 3*tf.reduce_mean((slopes-1.0)**2)
            d_loss = d_loss + gradient_penalty

            if self.d_weight == 0:
                d_loss = d_loss*self.gan_lambda


            self.g_l2_losses.append(g_l2_loss)
            self.g_adv_losses.append(g_adv_loss)
            self.g_losses.append(g_loss)
            self.d_real_losses.append(d_real_loss)
            self.d_fake_losses.append(d_fake_loss)
            self.d_losses.append(d_loss)

            self.d_real_loss_summary = scalar_summary("d_real_loss", d_real_loss)
            self.d_fake_loss_summary = scalar_summary("d_fake_loss", d_fake_loss)
            self.g_loss_summary = scalar_summary("g_loss", g_loss)
            self.g_l2_loss_summary = scalar_summary("g_l2_loss", g_l2_loss)
            self.g_loss_adv_summary = scalar_summary("g_adv_loss", g_adv_loss)
            self.d_loss_summary = scalar_summary("d_loss", d_loss)

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
        # for x in t_vars:
        #     assert x in self.g_vars or x in self.d_vars, x.name
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
        g_summaries = [self.d_fake_summary,
                        self.d_fake_loss_summary,
                        self.g_loss_summary,
                        self.g_l2_loss_summary,
                        self.g_loss_adv_summary,
                        self.generated_wav_summary,
                        self.generated_audio_summary]
        d_summaries = [self.d_loss_summary, self.d_real_summary, self.d_real_loss_summary, self.nonreverb_audio_summary, self.nonreverb_wav_summary]

        if hasattr(self, 'alpha_summ'):
            g_summaries += self.alpha_summ
        self.g_sum = tf.summary.merge(g_summaries)
        self.d_sum = tf.summary.merge(d_summaries)

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.summary.FileWriter(os.path.join(config.save_path, 'train'), self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sample_reverb, sample_nonreverb, sample_z = self.sess.run([self.gt_reverb[0], self.gt_nonreverb[0], self.zz[0]], feed_dict={self.is_valid:False})
        v_sample_reverb, v_sample_nonreverb, v_sample_z = self.sess.run([self.gt_reverb[0], self.gt_nonreverb[0], self.zz[0]],
                                                          feed_dict={self.is_valid: True, self.is_train: False})

        print ('sample reverb shape: ', sample_reverb.shape)
        print ('sample nonreverb shape: ', sample_nonreverb.shape)
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
        d_fake_losses = []
        d_real_losses = []
        g_adv_losses = []
        g_l2_losses = []

        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                if counter % config.save_freq == 0:

                    for d_iter in range(self.disc_updates):
                        _d_opt, _d_sum, d_fake_loss, d_real_loss = self.sess.run([d_opt, self.d_sum, self.d_fake_losses[0], self.d_real_losses[0]], feed_dict={self.is_valid:False, self.is_train: True})
                        #_d_sum, d_fake_loss, d_real_loss = self.sess.run(
                        #   [self.d_sum, self.d_fake_losses[0], self.d_real_losses[0]], feed_dict={self.is_valid: False})

                        if self.d_clip_weights:
                            self.sess.run(self.d_clip, feed_dict={self.is_valid:False,self.is_train: True})

                    #_g_opt, _g_sum, g_adv_loss, g_l2_loss = self.sess.run([g_opt, self.g_sum, self.g_adv_losses[0], self.g_l2_losses[0]], feed_dict={self.is_valid:False})
                    _g_opt, _g_sum, g_adv_loss, g_l2_loss = self.sess.run([g_opt, self.g_sum, self.g_adv_losses[0], self.g_l2_losses[0]], feed_dict={self.is_valid:False,self.is_train: True})

                else:
                    for d_iter in range(self.disc_updates):
                        _d_opt, d_fake_loss, d_real_loss = self.sess.run([d_opt, self.d_fake_losses[0], self.d_real_losses[0]], feed_dict={self.is_valid:False,self.is_train: True})
                        #d_fake_loss, d_real_loss = self.sess.run(
                        #    [self.d_fake_losses[0], self.d_real_losses[0]], feed_dict={self.is_valid: False})
                        if self.d_clip_weights:
                            self.sess.run(self.d_clip, feed_dict={self.is_valid:False,self.is_train: True})
                    #_g_opt, g_adv_loss, g_l2_loss = self.sess.run([g_opt, self.g_adv_losses[0], self.g_l2_losses[0]], feed_dict={self.is_valid:False})
                    _g_opt, g_adv_loss, g_l2_loss = self.sess.run([g_opt, self.g_adv_losses[0], self.g_l2_losses[0]], feed_dict={self.is_valid:False,self.is_train: True})

                end = timeit.default_timer()
                batch_timings.append(end - start)
                d_fake_losses.append(d_fake_loss)
                d_real_losses.append(d_real_loss)
                g_adv_losses.append(g_adv_loss)
                g_l2_losses.append(g_l2_loss)
                print('{}/{} (epoch {}), d_rl_loss = {:.5f}, '
                      'd_fk_loss = {:.5f}, '#d_nfk_loss = {:.5f}, '
                      'g_adv_loss = {:.5f}, g_l1_loss = {:.5f},'
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    current_epoch,
                                                    d_real_loss,
                                                    d_fake_loss,
                                                    g_adv_loss,
                                                    g_l2_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += 1
                counter += 1

                if (counter) % 2000 == 0 and (counter) > 0:
                    self.save(config.save_path, counter)
                if (counter) % config.save_freq == 0:
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    #feed_dict = {self.gt_nonreverb[0]:v_sample_nonreverb, self.gt_reverb[0]:v_sample_reverb, self.zz[0]:v_sample_z, self.is_valid:True}

                    canvas_w, s_reverb, s_nonreverb = self.sess.run([self.GG[0], self.gt_reverb[0], self.gt_nonreverb[0]], feed_dict={self.is_valid:True,self.is_train: False})

                    if not os.path.exists(save_path+'/wav'):
                        os.makedirs(save_path + '/wav')
                    if not os.path.exists(save_path + '/txt'):
                        os.makedirs(save_path + '/txt')
                    if not os.path.exists(save_path + '/spec'):
                        os.makedirs(save_path + '/spec')

                    print ('max :', np.max(canvas_w[0]), 'min :', np.min(canvas_w[0]))

                    if self.pre_emphasis>0:
                        canvas_w = self.de_emphasis(canvas_w, self.pre_emphasis)
                        s_reverb = self.de_emphasis(s_reverb, self.pre_emphasis)
                        s_nonreverb = self.de_emphasis(s_nonreverb, self.pre_emphasis)


                    x_pr = canvas_w.flatten()
                    x_pr = x_pr[:int(len(x_pr)/8)]
                    x_lr = s_reverb.flatten()[:len(x_pr)]
                    x_hr = s_nonreverb.flatten()[:len(x_pr)]

                    Sl = self.get_spectrum(x_lr, n_fft=2048)
                    Sh = self.get_spectrum(x_hr, n_fft=2048)
                    Sp = self.get_spectrum(x_pr, n_fft=2048)

                    S = np.concatenate((Sl.reshape(Sh.shape[0], Sh.shape[1]), Sh, Sp), axis=1)
                    fig = Figure(figsize=S.shape[::-1], dpi=1, frameon=False)
                    canvas = FigureCanvas(fig)
                    fig.figimage(S, cmap='jet')
                    fig.savefig(save_path + '/spec/' + 'valid_batch_index' + str(counter) + '-th_pr.png')

                    librosa.output.write_wav(save_path + '/wav/'+str(counter)+'_dereverb.wav', x_pr, 16000)

                    librosa.output.write_wav(save_path + '/wav/'+str(counter)+'_reverb.wav', x_lr, 16000)

                    librosa.output.write_wav(save_path + '/wav/'+str(counter)+'_orig.wav', x_hr, 16000)

                    canvas_w, s_reverb, s_nonreverb = self.sess.run([self.GG[0],self.gt_reverb[0], self.gt_nonreverb[0]], feed_dict={self.is_valid:False, self.is_train: True})


                    print ('max :', np.max(canvas_w[0]), 'min :', np.min(canvas_w[0]))



                    x_pr = canvas_w.flatten()
                    x_pr = x_pr[:int(len(x_pr)/8)]
                    x_lr = s_reverb.flatten()[:len(x_pr)]
                    x_hr = s_nonreverb.flatten()[:len(x_pr)]

                    Sl = self.get_spectrum(x_lr, n_fft=2048)
                    Sh = self.get_spectrum(x_hr, n_fft=2048)
                    Sp = self.get_spectrum(x_pr, n_fft=2048)

                    S = np.concatenate((Sl.reshape(Sh.shape[0], Sh.shape[1]), Sh, Sp), axis=1)
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
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    break

        except tf.errors.OutOfRangeError:
            print('done training')
            pass
        finally:
            coord.request_stop()
        coord.join(threads)

    # def infer(self, x):
    #     ''' inference the nonreverb resolution signal from reverb resolution signal'''
    #     h_res = None
    #     canvas_w = self.sess.run(self.GG[0], feed_dict={self.is_valid:True})[0]
    #     canvas_w = canvas_w.reshape((self.canvas_size))
    #     h_res = self.de_emphasis(canvas_w, self.pre_emphasis)

    #     return h_res

