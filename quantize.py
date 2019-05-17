from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import helper
import pandas as pd
import json
import os
import fold_batchnorm


import argparse
import os.path
import sys
import numpy as np

import tensorflow as tf
import input_data
import quant_models as models

try:
    xrange
except NameError:
    xrange = range


def run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                        window_size_ms, window_stride_ms, dct_coefficient_count,
                        model_architecture, model_size_info, act_max, data_url, data_dir, silence_percentage,
                        unknown_percentage, checkpoint, batch_size, include_silence=True,
                        lower_frequency_limit=20, upper_frequency_limit=4000, filterbank_channel_count=40,
                        is_bg_volume_constant=False, feature_extraction='mfcc'):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
      wanted_words: Comma-separated list of the words we're trying to recognize.
      sample_rate: How many samples per second are in the input audio files.
      clip_duration_ms: How many samples to analyze for the audio pattern.
      window_size_ms: Time slice duration to estimate frequencies from.
      window_stride_ms: How far apart time slices should be.
      dct_coefficient_count: Number of frequency bands to analyze.
      model_architecture: Name of the kind of model to generate.
      model_size_info: Model dimensions : different lengths for different models
    """
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    words_list = input_data.prepare_words_list(wanted_words.split(','), include_silence)
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count, lower_frequency_limit, upper_frequency_limit, filterbank_channel_count)

    audio_processor = input_data.AudioProcessor(
        data_url, data_dir, silence_percentage,
        unknown_percentage,
        wanted_words.split(','), 0,
        100, model_settings)

    label_count = model_settings['label_count']
    fingerprint_size = model_settings['fingerprint_size']

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits = models.create_model(
        fingerprint_input,
        model_settings,
        model_architecture,
        model_size_info,
        act_max,
        is_training=False)

    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    models.load_variables_from_checkpoint(sess, checkpoint)

    for v in tf.trainable_variables():
        var_name = str(v.name)
        var_values = sess.run(v)
        min_value = var_values.min()
        max_value = var_values.max()
        int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
        dec_bits = 7 - int_bits
        # convert to [-128,128) or int8
        var_values = np.round(var_values * 2 ** dec_bits)
        var_values = var_values / (2 ** dec_bits)
        # update the weights in tensorflow graph for quantizing the activations
        var_values = sess.run(tf.assign(v, var_values))

        # test set
    set_size = audio_processor.set_size('testing')
    total_accuracy = 0
    total_conf_matrix = None
    for i in xrange(0, set_size, batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess, is_bg_volume_constant,
            feature_extraction)
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
            })
        batch_size = min(batch_size, set_size - i)
        total_accuracy += (test_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix

    tf.reset_default_graph()
    sess.close()
    return total_accuracy


def get_best_act_max(wanted_words, sample_rate, clip_duration_ms,
                   window_size_ms, window_stride_ms, dct_coefficient_count,
                   model_architecture, model_size_info, act_max, data_url, data_dir, silence_percentage,
                   unknown_percentage, checkpoint, batch_size, include_silence, lower_frequency_limit,
                   upper_frequency_limit, filterbank_channel_count, is_bg_volume_constant, feature_extraction):

    val_acc = run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                       window_size_ms, window_stride_ms, dct_coefficient_count,
                       model_architecture, model_size_info, act_max, data_url, data_dir, silence_percentage,
                       unknown_percentage, checkpoint, batch_size, include_silence, lower_frequency_limit,
                       upper_frequency_limit, filterbank_channel_count, is_bg_volume_constant, feature_extraction)

    print("Without quantization. Acc: {}".format(val_acc))
    print()

    for act_layer in range(len(act_max)):
        best_acc = 0
        for i in range(7):
            maximum = 2**i
            act_max[act_layer] = maximum
            val_acc = run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                               window_size_ms, window_stride_ms, dct_coefficient_count,
                               model_architecture, model_size_info, act_max, data_url, data_dir, silence_percentage,
                               unknown_percentage, checkpoint, batch_size, include_silence, lower_frequency_limit,
                               upper_frequency_limit, filterbank_channel_count, is_bg_volume_constant,
                               feature_extraction)

            print("Layer: {} | Max: {} | Acc: {}".format(act_layer, maximum, val_acc))
            print(act_max)
            print()
            if best_acc >= val_acc:
                act_max[act_layer] = int(maximum/2) # assign previous value, as acc starts to decline otherwise
                break
            else:
                best_acc = val_acc
    return act_max

def main():
    parameters = {'background_frequency': 1, 'background_volume': 1, 'batch_size': 100, 'check_nans': False,
                  'clip_duration_ms': 2000,
                  'data_dir': '/projects/tanelp/thesis/data/interim/training', 'data_url': '',
                  'dct_coefficient_count': 5, 'eval_step_interval': 100, 'how_many_training_steps': '3000,3000,3000',
                  'learning_rate': '0.0005,0.0001,0.00002', 'model_architecture': 'ds_cnn', 'sample_rate': 16000,
                  'save_step_interval': 100, 'search_space': {'feature_extraction': 'gfcc',
                                                              'is_bg_volume_constant': True, 'lower_frequency': 1000,
                                                              'model_size_info': {'layers': (
                                                              {'num_channels': 16.0, 'sx': 15.0, 'sy': 2.0},
                                                              {'num_channels': 16.0}, {'num_channels': 48.0},
                                                              {'num_channels': 32.0}, {'num_channels': 16.0}),
                                                                                  'num_layers': 5},
                                                              'num_fbank_filters': 6,
                                                              'upper_frequency': 4000, 'window_size_ms': 16,
                                                              'window_stride_coeff': 0.5},
                  'silence_percentage': 0, 'start_checkpoint': '', 'testing_percentage': 0, 'time_shift_ms': 100,
                  'unknown_percentage': 100,
                  'valid_dir': '/projects/tanelp/thesis/data/interim/hw_valid',
                  'validation_percentage': 50, 'wanted_words': 'parus_major',
                  'work_dir': 'work/HPO-PROPER/DS_CNN2/', 'train_dir': 'work/HPO-PROPER/DS_CNN2/training'}
    pe = helper.ParameterExtractor(parameters)


    feature_extraction = pe.get_param('feature_extraction')

    if feature_extraction == 'gfcc':
        # TODO: load these values if extraction method = 'gfcc'
        gfcc_maximums = [1288.51391602, 173.28684998, 115.05116272, 113.59124756, 79.25372314]
        gfcc_minimums = [414.26879883, -192.18196106, -154.86968994, -109.34169006, -101.20419312]
        helper.write_gfcc_tables("work/data.h", parameters, gfcc_minimums, gfcc_maximums)

    wanted_words = pe.get_param('wanted_words')
    clip_duration_ms = int(pe.get_param('clip_duration_ms'))
    window_size_ms = int(pe.get_param('window_size_ms'))
    window_stride_ms = int(pe.get_param('window_stride_coeff') * window_size_ms)
    dct_coefficient_count = int(pe.get_param('dct_coefficient_count'))
    sample_rate = int(pe.get_param('sample_rate'))
    model_architecture = pe.get_param('model_architecture')
    model_size_info = pe.get_model_size_info()
    num_layers = model_size_info[0]
    silence_percentage = pe.get_param('silence_percentage')
    include_silence = silence_percentage != 0

    checkpoint_base_dir = os.path.join(pe.get_param('train_dir'), 'best')
    max_acc = max([x.split('.')[0].split('_')[-1] for x in os.listdir(checkpoint_base_dir) if 'ckpt' in x])
    checkpoint = os.path.join(checkpoint_base_dir, [x for x in os.listdir(checkpoint_base_dir) if
                                                    max_acc in x and '.index' in x and '_bnfused' not in x][0].replace(
        '.index', ''))

    fold_batchnorm.fold_batch_norm(wanted_words, sample_rate, clip_duration_ms,
                                   window_size_ms, window_stride_ms, dct_coefficient_count, model_architecture,
                                   model_size_info, checkpoint, include_silence)

    act_max = [0] * (num_layers * 2 + 2)
    bnfused_checkpoint = checkpoint + '_bnfused'


    background_volume = 1
    background_frequency = 1
    eval_step_interval = 100
    save_step_interval = 100
    start_checkpoint = ''
    check_nans = False
    data_dir = '/projects/tanelp/thesis/data/interim/training'
    test_dir = '/projects/tanelp/thesis/data/interim/hw_test'
    unknown_percentage = 100
    time_shift_ms = 100

    data_url = pe.get_param('data_url')
    sample_rate = pe.get_param('sample_rate')
    batch_size = pe.get_param('batch_size')
    model_architecture = pe.get_param('model_architecture')
    testing_percentage = pe.get_param('testing_percentage')
    validation_percentage = pe.get_param('validation_percentage')
    unknown_percentage = pe.get_param('unknown_percentage')
    valid_dir = pe.get_param('valid_dir')
    lower_frequency_limit = pe.get_param('lower_frequency')
    upper_frequency_limit = pe.get_param('upper_frequency')
    filterbank_channel_count = pe.get_param('num_fbank_filters')
    is_bg_volume_constant = pe.get_param('is_bg_volume_constant')

    best_act_max = get_best_act_max(wanted_words, sample_rate, clip_duration_ms,
                       window_size_ms, window_stride_ms, dct_coefficient_count,
                       model_architecture, model_size_info, act_max, data_url, valid_dir, silence_percentage,
                       unknown_percentage, bnfused_checkpoint,
                       batch_size, include_silence, lower_frequency_limit, upper_frequency_limit,
                       filterbank_channel_count, is_bg_volume_constant, feature_extraction)
    print("best_act_max", best_act_max)


if __name__ == '__main__':
    main()