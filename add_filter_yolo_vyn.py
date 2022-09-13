import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from custom_layers import YoloDecoder, YoloKerasWrapper

from decoders import decode, filter_boxes


flags.DEFINE_string('model', './checkpoints/yolov4.h5', 'define the keras model path as a .h5 file')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_float('score_thres', 0.5, 'define score threshold')
flags.DEFINE_integer('input_size', 0, 'define input size of export model')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('save_file', './checkpoints/yolov4_complete.h5', 'define where to save the model')


def save_tf_with_decoder_layer():
    strides = [8, 16, 32]
    model = tf.keras.models.load_model(FLAGS.model, custom_objects={'YoloDecoder': YoloDecoder(strides=strides),
                                                                  'YoloKerasWrapper': YoloKerasWrapper()})
    model.strides = strides
    input_size = model.input.shape[1]
    pred = model.output
    if FLAGS.framework != 'tflite':
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=FLAGS.score_thres,
                                        input_shape=tf.constant([input_size, input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(model.input, pred)
    model.summary()
    model.save(FLAGS.save_file)

def save_tf():
    model = tf.keras.models.load_model(FLAGS.model, custom_objects={'YoloDecoder': YoloDecoder(),
                                                                    'YoloKerasWrapper': YoloKerasWrapper()})
    input_size = FLAGS.input_size if FLAGS.input_size else model.input.shape[1]
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE  = model.layers[-1].strides, model.layers[-1].anchors, model.layers[-1].num_classes, model.layers[-1].xy_scale
    feature_maps = model.layers[-1].input

    bbox_tensors = []
    prob_tensors = []
    if FLAGS.tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            elif i == 1:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if FLAGS.framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres,
                                        input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(model.input, pred)
    model.summary()
    model.save(FLAGS.save_file)


# model.save('yolo_tiny_1stJune.h5')

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass