# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 9:03
# @Author  : zhaogang

import sys
sys.path.insert(0,'/home/qa/zg/EasyTransfer-master')
import tensorflow as tf
from easytransfer import base_model, FLAGS
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import TFRecordReader
from easytransfer.losses import softmax_cross_entropy,focal_loss_softmax
from sklearn.metrics import classification_report
import numpy as np

class MultiTaskTFRecordReader(TFRecordReader):
    def __init__(self, input_glob, batch_size, is_training=False,
                 **kwargs):

        super(MultiTaskTFRecordReader, self).__init__(input_glob, batch_size, is_training, **kwargs)
        self.task_fps = []
        with tf.gfile.Open(input_glob, 'r') as f:
            for line in f:
                line = line.strip()
                self.task_fps.append(line)

    def get_input_fn(self):
        def input_fn():
            num_datasets = len(self.task_fps)
            datasets = []
            for input_glob in self.task_fps:
                dataset = tf.data.TFRecordDataset(input_glob)
                dataset = self._get_data_pipeline(dataset, self._decode_tfrecord)
                datasets.append(dataset)

            choice_dataset = tf.data.Dataset.range(num_datasets).repeat()
            return tf.data.experimental.choose_from_datasets(datasets, choice_dataset).batch(num_datasets)

        return input_fn

############ multiloss
from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K
# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var)
            #loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
            #loss += K.sum(precision * softmax_cross_entropy(y_true, y_pred.shape[1], y_pred) + log_var, -1)
            loss += K.sum(precision * focal_loss_softmax(y_true, y_pred.shape[1], y_pred) + log_var, -1)
        return K.mean(loss)


    def call(self, inputs):
        ys_pred = [inputs[0],inputs[1],inputs[2]]
        ys_true = [inputs[3],inputs[4],inputs[5]]

        loss=self.multi_loss(ys_true, ys_pred)

        #self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        #return K.concatenate(inputs, -1)
        return loss
############

class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)
        self.inputs_logits=[]
        self.inputs_labels = []

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)

        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        global_step = tf.train.get_or_create_global_step()

        tnews_dense = layers.Dense(15,
                     kernel_initializer=layers.get_initializer(0.02),
                     name='tnews_dense')

        ocemotion_dense = layers.Dense(7,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='ocemotion_dense')

        ocnli_dense = layers.Dense(3,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='ocnli_dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)

        outputs_tnews = model([input_ids[0], input_mask[0], segment_ids[0]], mode=mode)
        pooled_output_tnews = outputs_tnews[1]
        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output_tnews = tf.nn.dropout(pooled_output_tnews, keep_prob=0.2)
        logits_tnews = tnews_dense(pooled_output_tnews)

        outputs_ocemotion = model([input_ids[1], input_mask[1], segment_ids[1]], mode=mode)
        pooled_output_ocemotion = outputs_ocemotion[1]
        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output_ocemotion = tf.nn.dropout(pooled_output_ocemotion, keep_prob=0.2)
        logits_ocemotion = ocemotion_dense(pooled_output_ocemotion)

        outputs_ocnli = model([input_ids[2], input_mask[2], segment_ids[2]], mode=mode)
        pooled_output_ocnli = outputs_ocnli[1]
        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output_ocnli = tf.nn.dropout(pooled_output_ocnli, keep_prob=0.5)
        logits_ocnli = ocnli_dense(pooled_output_ocnli)


        return [logits_tnews,logits_ocemotion,logits_ocnli], [label_ids[0],label_ids[1],label_ids[2]]

    # def build_loss(self, logits, labels):
    #     global_step = tf.train.get_or_create_global_step()
    #     return tf.case([(tf.equal(tf.mod(global_step, 3), 0), lambda : softmax_cross_entropy(labels, 15, logits)),
    #                   (tf.equal(tf.mod(global_step, 3), 1), lambda : softmax_cross_entropy(labels, 7, logits)),
    #                   (tf.equal(tf.mod(global_step, 3), 2), lambda : softmax_cross_entropy(labels, 3, logits)),
    #                   ], exclusive=True)

    def build_loss(self,logits, labels):
        return CustomMultiLossLayer(nb_outputs=3)([logits[0],logits[1],logits[2], labels[0],labels[1],labels[2]])


    def build_predictions(self, output):
        tnews_logits = output['tnews_logits']
        ocemotion_logits = output['ocemotion_logits']
        ocnli_logits = output['ocnli_logits']

        tnews_predictions = tf.argmax(tnews_logits, axis=-1, output_type=tf.int32)
        ocemotion_predictions = tf.argmax(ocemotion_logits, axis=-1, output_type=tf.int32)
        ocnli_predictions = tf.argmax(ocnli_logits, axis=-1, output_type=tf.int32)

        ret_dict = {
            "tnews_predictions": tnews_predictions,
            "ocemotion_predictions": ocemotion_predictions,
            "ocnli_predictions": ocnli_predictions,
            "label_ids": output['label_ids']
        }
        return ret_dict

def main(_):
    FLAGS.mode = "train"
    FLAGS.config = "./config/multitask_finetune_multiloss_roberta.json"
    app = Application()
    train_reader = MultiTaskTFRecordReader(input_glob=app.train_input_fp,
                                           is_training=True,
                                           input_schema=app.input_schema,
                                           batch_size=app.train_batch_size)
    app.run_train(reader=train_reader)


if __name__ == "__main__":
    tf.app.run()
