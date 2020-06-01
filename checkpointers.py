import time
import warnings
import io
from time import time
import logging
import numpy as np
from datetime import datetime
import pytz
from pytz import timezone
import os
import keras


class ModelCheckpoint(keras.callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, drive_folder_path=None, model_save_name=None, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, task_pharse=1):
        super(ModelCheckpoint, self).__init__()
        self.drive_folder_path = drive_folder_path
        self.model_save_name = model_save_name

        self.monitor = monitor
        self.verbose = verbose
        self.drive_folder_path = drive_folder_path
        self.model_save_name = model_save_name
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.task_pharse = task_pharse

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf


    def on_train_begin(self, logs=None):
        # tz_VN = pytz.timezone('Asia/Ho_Chi_Minh') 
        # cur_time = datetime.now(tz_VN)
        # time_folder = str(cur_time.month) + '_' + str(cur_time.day) + ':' + str(cur_time.hour) + '_' + str(cur_time.minute)

        # task_phare_folder = 'task_' + str(self.task_pharse)
        
        # self.folder_path = os.path.join(self.drive_folder_path, task_phare_folder) + '/' + time_folder
        # try:
        #     os.mkdir(self.folder_path)
        # except Exception as e:
        #     print(e)

        #delete folder
        try:
            os.system('rm {}/*'.format(self.drive_folder_path))
        except:
            print("can't remove folder")
    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        logs = logs or {}
        self.epochs_since_last_save += 1

        filepath = os.path.join(self.drive_folder_path, self.model_save_name).format(epoch=epoch + 1, **logs)

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            logging.info('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            logging.info('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    logging.info('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
     

