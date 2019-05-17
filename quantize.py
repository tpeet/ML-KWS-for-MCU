from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import helper
import pandas as pd
import json
import os
import fold_batchnorm
import math

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






def run_full_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                             window_size_ms, window_stride_ms, dct_coefficient_count,
                             model_architecture, model_size_info, act_max, data_url, data_dir,
                             silence_percentage, unknown_percentage, validation_percentage, testing_percentage,
                             checkpoint, batch_size, lower_frequency_limit,
                             upper_frequency_limit, filterbank_channel_count, is_bg_volume_constant,
                             feature_extraction):
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

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    words_list = input_data.prepare_words_list(wanted_words.split(','), silence_percentage!=0)
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count, lower_frequency_limit, upper_frequency_limit, filterbank_channel_count)

    audio_processor = input_data.AudioProcessor(
        data_url, data_dir, silence_percentage,
        unknown_percentage,
        wanted_words.split(','), validation_percentage,
        testing_percentage, model_settings)

    label_count = model_settings['label_count']
    fingerprint_size = model_settings['fingerprint_size']

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, fingerprints_4d = models.create_model(
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

    num_layers = model_size_info[0]
    helper.write_ds_cnn_cpp_file('ds_cnn.cpp', num_layers)

    ds_cnn_h_fname = "ds_cnn.h"
    weights_h_fname = "ds_cnn_weights.h"

    f = open(ds_cnn_h_fname, 'wb')
    f.close()

    with open(ds_cnn_h_fname, 'a') as f:
        helper.write_ds_cnn_h_beginning(f, wanted_words, sample_rate, clip_duration_ms,
                             window_size_ms, window_stride_ms, dct_coefficient_count,
                             model_size_info, act_max)

    #   # Quantize weights to 8-bits using (min,max) and write to file
    f = open(weights_h_fname, 'wb')
    f.close()

    total_layers = len(act_max)
    layer_no = 1
    weights_dec_bits = 0
    for v in tf.trainable_variables():
        var_name = str(v.name)
        var_values = sess.run(v)
        min_value = var_values.min()
        max_value = var_values.max()
        int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
        dec_bits = 7 - int_bits
        # convert to [-128,128) or int8
        var_values = np.round(var_values * 2 ** dec_bits)
        var_name = var_name.replace('/', '_')
        var_name = var_name.replace(':', '_')

        if (len(var_values.shape) > 2):  # convolution layer weights
            transposed_wts = np.transpose(var_values, (3, 0, 1, 2))
        else:  # fully connected layer weights or biases of any layer
            transposed_wts = np.transpose(var_values)

        # convert back original range but quantized to 8-bits or 256 levels
        var_values = var_values / (2 ** dec_bits)
        # update the weights in tensorflow graph for quantizing the activations
        var_values = sess.run(tf.assign(v, var_values))
        print(var_name + ' number of wts/bias: ' + str(var_values.shape) + \
              ' dec bits: ' + str(dec_bits) + \
              ' max: (' + str(var_values.max()) + ',' + str(max_value) + ')' + \
              ' min: (' + str(var_values.min()) + ',' + str(min_value) + ')')

        conv_layer_no = layer_no // 2 + 1

        wt_or_bias = 'BIAS'
        if 'weights' in var_name:
            wt_or_bias = 'WT'

        with open(weights_h_fname, 'a') as f:
            if conv_layer_no == 1:
                f.write('#define CONV1_{} {{'.format(wt_or_bias))
            elif conv_layer_no <= num_layers:
                if layer_no % 2 == 0:
                    f.write('#define CONV{}_DS_{} {{'.format(conv_layer_no, wt_or_bias))
                else:
                    f.write('#define CONV{}_PW_{} {{'.format(conv_layer_no, wt_or_bias))
            else:
                f.write('#define FINAL_FC_{} {{'.format(wt_or_bias))

            transposed_wts.tofile(f, sep=", ", format="%d")
            f.write('}\n')

        if 'weights' in var_name:
            weights_dec_bits = dec_bits

        if 'biases' in var_name:
            if layer_no == total_layers - 2:  # if averege pool layer, go to the next one
                layer_no += 1
            input_dec_bits = 7 - np.log2(act_max[layer_no - 1])
            output_dec_bits = 7 - np.log2(act_max[layer_no])
            weights_x_input_dec_bits = input_dec_bits + weights_dec_bits
            bias_lshift = int(weights_x_input_dec_bits - dec_bits)
            output_rshift = int(weights_x_input_dec_bits - output_dec_bits)
            print("Layer no: {} | Bias Lshift: {} | Output Rshift: {}\n".format(layer_no, bias_lshift, output_rshift))
            with open('ds_cnn.h', 'a') as f:
                if conv_layer_no == 1:
                    f.write("#define CONV1_BIAS_LSHIFT {}\n".format(bias_lshift))
                    f.write("#define CONV1_OUT_RSHIFT {}\n".format(output_rshift))
                elif conv_layer_no <= num_layers:
                    if layer_no % 2 == 0:
                        f.write("#define CONV{}_DS_BIAS_LSHIFT {}\n".format(conv_layer_no, bias_lshift))
                        f.write("#define CONV{}_DS_OUT_RSHIFT {}\n".format(conv_layer_no, output_rshift))

                    else:
                        f.write("#define CONV{}_PW_BIAS_LSHIFT {}\n".format(conv_layer_no, bias_lshift))
                        f.write("#define CONV{}_PW_OUT_RSHIFT {}\n".format(conv_layer_no, output_rshift))
                else:
                    f.write("#define FINAL_FC_BIAS_LSHIFT {}\n".format(bias_lshift))
                    f.write("#define FINAL_FC_OUT_RSHIFT {}\n".format(output_rshift))

            layer_no += 1
    input_dec_bits = 7 - np.log2(act_max[len(act_max) - 3])
    output_dec_bits = 7 - np.log2(act_max[len(act_max) - 2])
    with open(ds_cnn_h_fname, 'a') as f:
        f.write("#define AVG_POOL_OUT_LSHIFT {}\n\n".format(int(output_dec_bits - input_dec_bits)))
        helper.write_ds_cnn_h_end(f, wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count,
                           model_architecture, model_size_info, act_max)


    # Evaluate result after quantization on testing set
    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None

    for i in xrange(0, set_size, batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess, is_bg_volume_constant,
            feature_extraction)
        test_accuracy, conf_matrix, predictions, true_labels = sess.run(
            [evaluation_step, confusion_matrix, predicted_indices, expected_indices],
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

    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                       set_size))
    sess.close()


def main():
    parameters = {'background_frequency': 1, 'background_volume': 1, 'batch_size': 100, 'check_nans': False,
                  'clip_duration_ms': 2000,
                  'data_dir': '/projects/tanelp/thesis/data/interim/training', 'data_url': '',
                  'dct_coefficient_count': 5, 'eval_step_interval': 100, 'how_many_training_steps': '5000,5000,5000',
                  'learning_rate': '0.0005,0.0001,0.00002', 'model_architecture': 'ds_cnn', 'sample_rate': 16000,
                  'save_step_interval': 100, 'search_space': {'feature_extraction': 'gfcc',
                                                              'is_bg_volume_constant': True, 'lower_frequency': 1000,
                                                              'model_size_info': {'layers': (
                                                                  {'num_channels': 32.0, 'sx': 5.0, 'sy': 8.0},
                                                                  {'num_channels': 32.0}, {'num_channels': 32.0},
                                                                  {'num_channels': 32.0}),
                                                                  'num_layers': 4},
                                                              'num_fbank_filters': 10,
                                                              'upper_frequency': 8000, 'window_size_ms': 64,
                                                              'window_stride_coeff': 0.5},
                  'silence_percentage': 0, 'start_checkpoint': '', 'testing_percentage': 0, 'time_shift_ms': 100,
                  'unknown_percentage': 100,
                  'valid_dir': '/projects/tanelp/thesis/data/interim/hw_valid',
                  'validation_percentage': 100, 'wanted_words': 'parus_major',
                  'work_dir': 'work/FINAL-VERSION', 'train_dir': 'work/FINAL-VERSION/training'}
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

    run_full_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                               window_size_ms, window_stride_ms, dct_coefficient_count,
                               model_architecture, model_size_info, best_act_max, data_url, valid_dir, test_dir, silence_percentage,
                               unknown_percentage, validation_percentage, testing_percentage, bnfused_checkpoint,
                               batch_size)
if __name__ == '__main__':
    main()
