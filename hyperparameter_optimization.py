from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path
import sys

import numpy as np

import joblib
import pandas as pd
import train

from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, fmin, Trials
from timeit import default_timer as timer
import csv
import helper

TRIALS_FILE = 'work/trials.csv'
TRIALS_DIR = 'work/trials/'

global ITERATION
ITERATION = 0


space = {
  'search_space': {
    'window_size_ms': hp.choice('window_size_ms', [16, 32, 64, 128]),
    'window_stride_coeff': hp.quniform('window_stride_coeff', 0.25, 1, 0.25),
    'clip_duration_ms': 2000,
    'model_size_info': hp.choice('model_size_info',
                                 [
                                   {
                                     'num_layers': 3,
                                     'layers': [
                                       {
                                         'num_channels': hp.quniform('num_channels-1', 16, 48, 16),
                                         'sx': hp.quniform('sx-1', 5, 20, 5),
                                         'sy': hp.quniform('sy-1', 2, 8, 2),
                                       },
                                       {'num_channels': hp.quniform('num_channels-2', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-3', 16, 48, 16)},
                                     ]
                                   },
                                   {
                                     'num_layers': 4,
                                     'layers': [
                                       {
                                         'num_channels': hp.quniform('num_channels-4', 16, 48, 16),
                                         'sx': hp.quniform('sx-2', 5, 20, 5),
                                         'sy': hp.quniform('sy-2', 2, 8, 2),
                                       },
                                       {'num_channels': hp.quniform('num_channels-5', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-6', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-7', 16, 48, 16)},
                                     ]
                                   },
                                   {
                                     'num_layers': 5,
                                     'layers': [
                                       {
                                         'num_channels': hp.quniform('num_channels-8', 16, 48, 16),
                                         'sx': hp.quniform('sx-3', 5, 20, 5),
                                         'sy': hp.quniform('sy-3', 2, 8, 2),
                                       },
                                       {'num_channels': hp.quniform('num_channels-9', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-10', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-11', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-12', 16, 48, 16)},
                                     ]
                                   },
                                   {
                                     'num_layers': 6,
                                     'layers': [
                                       {
                                         'num_channels': hp.quniform('num_channels-13', 16, 48, 16),
                                         'sx': hp.quniform('sx-4', 5, 20, 5),
                                         'sy': hp.quniform('sy-4', 2, 8, 2),
                                       },
                                       {'num_channels': hp.quniform('num_channels-14', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-15', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-16', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-17', 16, 48, 16)},
                                       {'num_channels': hp.quniform('num_channels-18', 16, 48, 16)},
                                     ]
                                   },

                                 ]),

  },
  'data_url': '',
  'background_volume': 1,
  'background_frequency': 1,
  'sample_rate': 16000,
  'eval_step_interval': 100,
  'batch_size': 100,
  'save_step_interval': 100,
  'model_architecture': 'ds_cnn',
  'start_checkpoint': '',
  'check_nans': False,
  'wanted_words': 'parus_major',
  'data_dir': '/projects/tanelp/thesis/data/interim/training',
  'valid_dir': '/projects/tanelp/thesis/data/interim/hw_test',
  'testing_percentage': 0,
  'validation_percentage': 50,
  'unknown_percentage': 100,
  'silence_percentage': 0,
  'time_shift_ms': 100,
  'work_dir': 'work/HPO-HW_AUDIO-7/DS_CNN{}/',
  'lower_frequency': 1000,
  'upper_frequency': 8000,
  'num_fbank_filters': 10,
  'is_bg_volume_constant': True,
  'feature_extraction': 'gfcc',
  'how_many_training_steps': '1,1,1',
  'learning_rate': '0.0005,0.0001,0.00002',
  'dct_coefficient_count': 5,
}


def objective(parameters):
  """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

  pe = helper.ParameterExtractor(parameters)
  summaries_dir = os.path.join(pe.get_param('work_dir').format(ITERATION), 'retrain_logs')
  train_dir = os.path.join(pe.get_param('work_dir').format(ITERATION), 'training')
  start = timer()

  model_size_info = pe.get_model_size_info()

  window_size_ms = int(pe.get_param('window_size_ms'))
  window_stride_ms = int(pe.get_param('window_stride_coeff') * window_size_ms)

  print()
  print(parameters)
  sys.stdout.flush()

  # Hyperopt tends to use same hyperparameters several times
  df = pd.read_csv(TRIALS_FILE)
  if str(parameters) in df['parameters'].tolist():
    print("There is already trial with these experiments!")
    loss = df[df.parameters == str(parameters)]['loss'].tolist()[0]
    best_val_acc = df[df.parameters == str(parameters)]['best_val_acc'].tolist()[0]
    num_params = df[df.parameters == str(parameters)]['num_params'].tolist()[0]
    return {'loss': loss, 'parameters': parameters, 'iteration': ITERATION,
            'acc': best_val_acc, "num_params": num_params,
            'train_time': 0, 'status': STATUS_OK}

  best_val_acc, num_params = \
    train.train(pe.get_param('wanted_words'), int(pe.get_param('sample_rate')), int(pe.get_param('clip_duration_ms')),
                window_size_ms, window_stride_ms, int(pe.get_param('time_shift_ms')),
                int(pe.get_param('dct_coefficient_count')), pe.get_param('data_url'),
                pe.get_param('data_dir'), pe.get_param('valid_dir'), pe.get_param('silence_percentage'),
                pe.get_param('unknown_percentage'), pe.get_param('validation_percentage'),
                pe.get_param('testing_percentage'),pe.get_param('how_many_training_steps'),
                pe.get_param('learning_rate'), pe.get_param('model_architecture'), model_size_info,
                pe.get_param('check_nans'), summaries_dir, train_dir, pe.get_param('start_checkpoint'),
                int(pe.get_param('batch_size')), pe.get_param('background_frequency'),
                pe.get_param('background_volume'), int(pe.get_param('eval_step_interval')),
                int(pe.get_param('lower_frequency')), int(pe.get_param('upper_frequency')),
                int(pe.get_param('num_fbank_filters')), 0, pe.get_param('is_bg_volume_constant'),
                pe.get_param('feature_extraction'), True)

  loss = 1 - best_val_acc
  run_time = timer() - start

  with open(TRIALS_FILE, 'a') as f:
    writer = csv.writer(f)
    parameters['train_dir'] = train_dir
    parameters['work_dir'] = parameters['work_dir'].format(ITERATION)
    writer.writerow([ITERATION, loss, best_val_acc, run_time, num_params, parameters])

  return {'loss': loss, 'parameters': parameters, 'iteration': ITERATION,
          'acc': best_val_acc, "num_params": num_params,
          'train_time': run_time, 'status': STATUS_OK}




def run_trials(parameters):
  if not os.path.exists(TRIALS_DIR):
    os.makedirs(TRIALS_DIR)

  if not os.path.exists(TRIALS_FILE):
    # Write the headers to the file
    with open(TRIALS_FILE, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['ITERATION', 'loss', 'best_val_acc', 'run_time', 'num_params', 'parameters'])

  global ITERATION
  trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
  max_trials = 2  # initial max_trials. put something small to not have to wait

  iterations = [int(x.split('-')[1].split('.')[0]) for x in os.listdir(TRIALS_DIR) if x.endswith(".hyperopt")]
  if len(iterations) == 0:
    trials = Trials()
    print("Created new Trials object")
  else:
    ITERATION = max(iterations)
    trials_fname = os.path.join(TRIALS_DIR, "trial-{}.hyperopt".format(ITERATION))
    try:  # try to load an already saved trials object, and increase the max
      trials = joblib.load(trials_fname)
      max_trials = len(trials.trials) + trials_step
      print("\nSuccessfully loaded! Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials,
                                                                                        trials_step))
    except:  # if failed to load, try again with previous file
      ITERATION -= 1
      trials = joblib.load(trials_fname)
      max_trials = len(trials.trials) + trials_step
      print("\nFailed to load! Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials,
                                                                                   trials_step))

  # Keep track of evals
  best = fmin(fn=objective, space=parameters, algo=tpe.suggest, max_evals=max_trials, trials=trials,
              rstate=np.random.RandomState(35))

  try:
    # save the trials object
    joblib.dump(trials, os.path.join(TRIALS_DIR, "trial-{}.hyperopt".format(ITERATION + 1)))
  except:
    print("Failed to save trials file")


# loop indefinitely and stop whenever you like
while True:
  run_trials(space)
