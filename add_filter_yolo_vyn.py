import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from custom_layers import YoloDecoder, YoloKerasWrapper

from decoders import decode, filter_boxes


flags.DEFINE_string('model', './checkpoints/yolov4.h5', 'define the keras model path as a .h5 file')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_float('score_thres', 0.5, 'define score threshold')
flags.DEFINE_integer('input_size', 0, 'define input size of export model')
flags.DEFINE_boolean('add_preprocessing', False, 'Add preprocessing step to the image before going into the model.')
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


def save_tf(input_size, model_path, framework, add_preprocessing, score_thres, is_tiny, save_file):
    model = tf.keras.models.load_model(model_path, custom_objects={'YoloDecoder': YoloDecoder(),
                                                                    'YoloKerasWrapper': YoloKerasWrapper()}, compile=False)
    input_size = input_size if input_size else model.input.shape[1]
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = model.layers[-1].strides, model.layers[-1].anchors, model.layers[-1].num_classes, model.layers[-1].xy_scale

    if add_preprocessing:
        aux_model = tf.keras.Model(model.input, model.layers[-1].input)
        inp = tf.keras.layers.Input([None, None, 3])
        x = tf.keras.layers.Lambda(tf.image.resize_with_pad, arguments={'target_height': input_size, 'target_width': input_size})(inp) / 255.
        feature_maps = aux_model(x)
    else:
        inp = model.input
        feature_maps = model.layers[-1].input

    bbox_tensors = []
    prob_tensors = []
    if is_tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        framework)
            elif i == 1:
                output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        framework)
            else:
                output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=score_thres,
                                        input_shape=tf.constant([input_size, input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(inp, pred)
    model.summary()
    model.save(save_file)


# model.save('yolo_tiny_1stJune.h5')

def main(_argv):
  save_tf(FLAGS.input_size, FLAGS.model, FLAGS.framework, FLAGS.add_preprocessing, FLAGS.score_thres, FLAGS.tiny, FLAGS.save_file)


if __name__ == '__main__':
    save_tf(None, '/home/isaac/Downloads/Site_OD.h5', 'tf', False, 0.5, False, '/home/isaac/Downloads/Site_OD_Output.h5')
    try:
        app.run(main)
    except SystemExit:
        pass
