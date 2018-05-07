import numpy as np
import tensorflow as tf

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from model_cycle_gan_stft import ReverbGAN
import os
import time
import sys
import numpy.random as random


from scipy import interpolate
from scipy.signal import decimate, spectrogram
from scipy.signal import butter, lfilter

import sys
import librosa


# os.environ["CUDA_VISIBLE_DEVICES"]="1"

flags = tf.app.flags
flags.DEFINE_integer("seed",111, "Random seed (Def: 111).")
flags.DEFINE_integer("epoch", 300, "Epochs to train (Def: 150).")
flags.DEFINE_integer("batch_size", 32, "Batch size (Def: 150).")
flags.DEFINE_integer("save_freq", 100, "Batch save freq (Def: 50).")
flags.DEFINE_integer("canvas_size", 2**14, "Canvas size (Def: 2^13).")
flags.DEFINE_integer("denoise_epoch", 5, "Epoch where noise in disc is "
                                          "removed (Def: 5).")
flags.DEFINE_integer("l2_remove_epoch", 150, "Epoch where L2 in G is "
                                           "removed (Def: 150).")
flags.DEFINE_integer("disc_updates", 2, "discriminator update")
flags.DEFINE_boolean("bias_deconv", False,
                     "Flag to specify if we bias deconvs (Def: False)")
flags.DEFINE_boolean("bias_downconv", False,
                     "flag to specify if we bias downconvs (def: false)")
flags.DEFINE_boolean("bias_D_conv", False,
                     "flag to specify if we bias D_convs (def: false)")
# TODO: noise decay is under check
flags.DEFINE_float("denoise_lbound", 0.001, "Min noise std to be still alive (Def: 0.001)")
flags.DEFINE_float("noise_decay", 0.7, "Decay rate of noise std (Def: 0.7)")
flags.DEFINE_float("d_label_smooth", 0.25, "Smooth factor in D (Def: 0.25)")
flags.DEFINE_float("init_noise_std", 0.5, "Init noise std (Def: 0.5)")
flags.DEFINE_float("skips", 0., "Skip connection or not (Def: 0)")
flags.DEFINE_float("l1_weight", 1., "Init L1 lambda (Def: 100)")
flags.DEFINE_float("d_weight", 1., "Init discriminator lambda (Def: 2)")
flags.DEFINE_integer("z_dim", 256, "Dimension of input noise to G (Def: 256).")
flags.DEFINE_integer("z_depth", 256, "Depth of input noise to G (Def: 256).")
flags.DEFINE_string("save_path", "output", "Path to save out model "
                                                   "files. (Def: ReverbGAN_results"
                                                   ").")
flags.DEFINE_string("g_activation", "leaky", "Type of nonlinearity in G: leaky or prelu. (Def: leaky).")
flags.DEFINE_string("model", "gan", "Type of model to train: gan or ae. (Def: gan).")
flags.DEFINE_string("deconv_type", "deconv", "Type of deconv method: deconv or "
                                             "nn_deconv (Def: deconv).")
flags.DEFINE_string("g_type", "com", "Type of G to use: com or dec. (Def: com).")
flags.DEFINE_float("g_lr", 0.00005, "G learning_rate (Def: 0.0002)")
flags.DEFINE_float("d_lr", 0.00005, "D learning_rate (Def: 0.0002)")
flags.DEFINE_float("beta_1", 0.9, "Adam beta 1 (Def: 0.5)")
flags.DEFINE_float("pre_emphasis", 0, "Pre-emph factor (Def: 0.95)")
flags.DEFINE_string("tfrecords", "../data/new_train_vctk_spec_56_50.tfrecords", "TFRecords"
                                                          " (Def: data/"
                                                          "ReverbGAN.tfrecords.")
flags.DEFINE_string("tfrecords_val", "../data/new_train_seoul_spec_56_25.tfrecords", "TFRecords_val"
                                                          " (Def: data/"
                                                          "ReverbGAN.tfrecords.")
flags.DEFINE_string("save_clean_path", "test_clean_results", "Path to save clean utts")
flags.DEFINE_string("test", None, "name of test wav (it won't train)")
flags.DEFINE_string("weights", None, "Weights file")
FLAGS = flags.FLAGS

def main(_):
  print ('parsed arguments: ', FLAGS.__flags)

  # make save path
  if not os.path.exists(FLAGS.save_path):
    os.makedirs(FLAGS.save_path)
  np.random.seed(FLAGS.seed)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  with tf.Session(config=config) as sess:
    model = ReverbGAN(sess, FLAGS)

    if FLAGS.test is None:
      model.train(sess, FLAGS)
    # else:
    #   if FLAGS.weights is None:
    #     raise ValueError('weights must be specified')
    #   print ('loading model weights')
    #   model.load(FLAGS.save_path, FLAGS.weights)



if __name__ == '__main__':
  tf.app.run()

