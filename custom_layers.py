import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras import layers as kl


class YoloDecoder(kl.Layer):
    """
    Implementation of the YOLO head. In some framewors it is call decoder
    """
    NAME = 'Standard'

    def __init__(self,
                 head_type=None,
                 image_shape=None,
                 strides=None,
                 num_classes=None,
                 anchors=None,
                 xy_scale=None,
                 max_bb_sizes_per_scale=None,
                 anchors_per_scale=3,
                 cap_value=-1, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(YoloDecoder, self).__init__(**kwargs)

        self.head_type = head_type
        self.image_shape = image_shape
        self.strides = strides
        self.num_classes = num_classes
        self.anchors = anchors
        self.xy_scale = xy_scale
        self.max_bb_sizes_per_scale = max_bb_sizes_per_scale
        self.anchors_per_scale = anchors_per_scale or 1  # None means anchorless, but this is used for reshaping so 1
        self.cap_value = cap_value

    def _smooth_clamp(self, x):
        """
        Return a smooth cap based on a modified sigmoid
        :param x (tensor): Value to smooth
        :return: The smoothed values
        """
        if self.cap_value <= 0:
            return x

        mi = -self.cap_value
        mx = self.cap_value

        t = (x - mi) / (mx - mi)

        value = mi + (mx - mi) / (1 + 55 ** (-t + 0.5))
        value = tf.where(x < 0, x, value)

        return value

    def call(self, inputs, **kwargs):
        """
        Implementation modified from:
        :param inputs: Input of the layers
        :param kwargs: Internal parameters
        :return:
        """
        # inputs = ops.convert_to_tensor(inputs)

        bbox_tensors = []
        prob_tensors = []
        div = lambda x, factor: [x_i // factor for x_i in x]

        if self.anchors is None:
            self.anchors = [None] * len(inputs)

        if self.head_type == 'train':
            for i, fm in enumerate(inputs):
                output_tensors = self.decode_train(fm, div(self.image_shape, self.strides[i]), i)

                bbox_tensors.append(fm)
                bbox_tensors.append(output_tensors)

            return bbox_tensors
        else:
            for i, fm in enumerate(inputs):
                output_tensors = self.decode(fm, div(self.image_shape, self.strides[i]), i)

                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])

            pred_bbox = tf.concat(bbox_tensors, axis=1)
            pred_prob = tf.concat(prob_tensors, axis=1)

        return (pred_bbox, pred_prob)

    def decode(self, inputs, output_size, current_scale):
        """
        Decode the output of the model depending on the type
        :param inputs:
        :return:
        """
        if self.head_type == 'train':
            output = self.decode_train(inputs, output_size, current_scale)
            output = (inputs, output)
        else:
            if self.head_type == 'trt':
                output = self.decode_trt(inputs, output_size, current_scale)
            elif self.head_type == 'tflite':
                output = self.decode_tflite(inputs, output_size, current_scale)
            else:
                output = self.decode_tf(inputs, output_size, current_scale)

        return output

    def decode_base(self, conv_output, output_size, current_scale):

        strides = self.strides[current_scale]
        anchors = self.anchors[current_scale]
        xy_scale = self.xy_scale[current_scale]

        conv_output = tf.reshape(
            conv_output,
            (tf.shape(conv_output)[0], output_size[0], output_size[1], self.anchors_per_scale, 5 + self.num_classes)
        )

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.num_classes),
                                                                              axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size[1]), tf.range(output_size[0]))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, self.anchors_per_scale, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        if self.head_type == 'trt':
            pred_xy = (tf.reshape(tf.sigmoid(conv_raw_dxdy),
                                  (-1, 2)) * xy_scale - 0.5 * (xy_scale - 1) + tf.reshape(xy_grid, (-1, 2))) * strides
        else:
            pred_xy = ((tf.sigmoid(conv_raw_dxdy) * xy_scale) - 0.5 * (xy_scale - 1) + xy_grid) * strides

        # ISAAC's Modification --- I don't like conv_raw_dwdh to be unbounded. So, I have bounded to -4 to 4. That is
        # plenty of room for the exponential to work
        pred_wh = self._smooth_clamp(conv_raw_dwdh)   # conv_raw_dwdh
        # This is for debugging purpose
        if tf.math.reduce_any(pred_wh > 40):
            a=1

        pred_wh = (tf.exp(pred_wh) * anchors)
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return pred_xywh, pred_conf, pred_prob

    def decode_train(self, conv_output, output_size, current_scale):
        pred_xywh, pred_conf, pred_prob = self.decode_base(conv_output, output_size, current_scale)
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    #@tf.function
    def decode_tf(self, conv_output, output_size, current_scale):
        batch_size = tf.shape(conv_output)[0]
        pred_xywh, pred_conf, pred_prob = self.decode_base(conv_output, output_size, current_scale)

        pred_prob = pred_conf * pred_prob
        pred_prob = tf.reshape(pred_prob, (batch_size, -1, self.num_classes))
        pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

        return pred_xywh, pred_prob

    def decode_trt(self, conv_output, output_size, current_scale):

        batch_size = tf.shape(conv_output)[0]
        pred_xywh, pred_conf, pred_prob = self.decode_base(conv_output, output_size, current_scale)

        pred_prob = pred_conf * pred_prob

        pred_prob = tf.reshape(pred_prob, (batch_size, -1, self.num_classes))
        pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
        return pred_xywh, pred_prob
        # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    #@tf.function
    def decode_tflite(self, conv_output, output_size, current_scale):

        strides = self.strides[current_scale]
        anchors = self.anchors[current_scale]
        xy_scale = self.xy_scale[current_scale]
        num_classes = self.num_classes

        conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0, \
        conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1, \
        conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(conv_output,
                                                                      (2, 2, 1 + num_classes, 2, 2, 1 + num_classes,
                                                                       2, 2, 1 + num_classes), axis=-1)

        conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
        for idx, score in enumerate(conv_raw_score):
            score = tf.sigmoid(score)
            score = score[:, :, :, 0:1] * score[:, :, :, 1:]
            conv_raw_score[idx] = tf.reshape(score, (1, -1, num_classes))
        pred_prob = tf.concat(conv_raw_score, axis=1)

        conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
        for idx, dwdh in enumerate(conv_raw_dwdh):
            dwdh = tf.exp(dwdh) * anchors[idx]  # before it was achors[i][idx]
            conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
        pred_wh = tf.concat(conv_raw_dwdh, axis=1)

        xy_grid = tf.meshgrid(tf.range(output_size[1]), tf.range(output_size[0]))
        xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
        xy_grid = tf.expand_dims(xy_grid, axis=0)
        xy_grid = tf.cast(xy_grid, tf.float32)

        conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
        for idx, dxdy in enumerate(conv_raw_dxdy):
            dxdy = ((tf.sigmoid(dxdy) * xy_scale) - 0.5 * (xy_scale - 1) + xy_grid) * strides
            conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
        pred_xy = tf.concat(conv_raw_dxdy, axis=1)
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        return pred_xywh, pred_prob

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_type": self.head_type,
            "image_shape": self.image_shape,
            "strides": self.strides,
            "num_classes": self.num_classes,
            "anchors": self.anchors,
            "xy_scale": self.xy_scale,
            "max_bb_sizes_per_scale": self.max_bb_sizes_per_scale,
            "anchors_per_scale": self.anchors_per_scale,
            "cap_value": self.cap_value,
        })
        return config


class YoloKerasWrapper(tf.keras.models.Model):
    """
    Simple implementation of the keras model with a different training and validation step
    """
    def __init__(self, strides=None, *args, use_ms_dayolo=False, **kwargs):
        self.strides = strides
        self.use_ms_dayolo = use_ms_dayolo
        self.current_epoch = -1
        super().__init__(*args, **kwargs)

    def compile(self, optimizers, losses, metrics=None, loss_weights=None, **kwargs):
        self.loss_weights = loss_weights or [1.0, 1.0]
        if self.use_ms_dayolo:
            super().compile(optimizers, losses[0], metrics, loss_weights=None, **kwargs)
            self.loss = losses
            self.loss_dayolo = losses[1]
            self.main_loss = losses[0]
        else:
            self.main_loss = losses
            super().compile(optimizers, losses, metrics, loss_weights=None, **kwargs)

    def train_step(self, data: list, *args, **kwargs):
        """
        Train step for the training process
        :param batch: Image and target
        :param epoch: The current epoch
        :param current_iteration: The current iteration
        :return: Losses and metrics as dictionary each of them (key name of the loss or metric and value their values)
        """
        image_data = data[0]
        target = data[1]
        with tf.GradientTape() as tape:
            pred_result = self(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            step = 2
            if self.use_ms_dayolo:
                separation_pos = 2 * len(pred_result)//3
                ms_dayolo_output = pred_result[separation_pos:]
                pred_result = pred_result[:separation_pos]
                step = 3

            # optimizing process
            aux_loss = 0.
            for i in range(len(target)//step):
                pos = i * step
                loss_items = self.main_loss(target[pos:pos + step], pred_result[2*i:2*(i + 1)], strides=self.strides[i])
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

                if self.use_ms_dayolo:
                    aux_loss += self.loss_dayolo(target[pos:pos + step], ms_dayolo_output[i])

            total_loss = self.loss_weights[0] * (giou_loss + conf_loss + prob_loss) + self.loss_weights[1] * aux_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        try:
            if hasattr(self.optimizer, 'optimizer'):
                self.optimizer.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            else:
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        except ValueError as e:
            # This is probably saying that the gradients are not training, not an issue with da yolo
            grad = 0

        losses = {'loss': total_loss, 'giou_loss': giou_loss, 'conf_loss': conf_loss, 'prob_loss': prob_loss}
        if self.use_ms_dayolo:
            losses['domain_loss'] = aux_loss

        # TODO: Add some metrics to control the performance of the model. Kept the metrics variable for better reminder
        metrics = {}

        return losses

    def test_step(self, batch, *args, training=False, **kwargs):
        image_data = batch[0]
        target = batch[1]

        pred_result = self(image_data, training=training)
        giou_loss = conf_loss = prob_loss = 0

        step = 2
        if self.use_ms_dayolo:
            separation_pos = 2 * len(pred_result) // 3
            pred_result = pred_result[:separation_pos]
            step = 3

        # optimizing process
        for i in range(len(target)//step):
            pos = i * step
            loss_items = self.main_loss(target[pos:pos + step], pred_result[2*i:2*(i+1)], strides=self.strides[i])
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

        losses = {'loss': total_loss, 'giou_loss': giou_loss, 'conf_loss': conf_loss, 'prob_loss': prob_loss}
        # TODO: Add some metrics to control the performance of the model. Kept the metrics variable for better reminder
        metrics = {}

        return losses

    def train_to_test_output(self, output, num_classes):
        """
        Convert the output of a model set to train to a model set to test
        :param output:
        :return:
        """
        total_pred_xywh, total_pred_prob = [], []
        for i in range(1, len(output), 2):
            output_i = output[i]
            batch_size = tf.shape(output_i)[0]

            pred_xywh, pred_conf, pred_prob = tf.split(output_i, (4, 1, -1), axis=-1)

            pred_prob = pred_conf * pred_prob
            pred_prob = tf.reshape(pred_prob, (batch_size, -1, num_classes))
            pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

            total_pred_xywh.append(pred_xywh)
            total_pred_prob.append(pred_prob)

        pred_bbox = tf.concat(total_pred_xywh, axis=1)
        pred_prob = tf.concat(total_pred_prob, axis=1)

        return (pred_bbox, pred_prob)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_ms_dayolo": self.use_ms_dayolo,
            "strides": self.strides
        })
        return config
