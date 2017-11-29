from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string("host", "gpu03", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 8500, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "default", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", -1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS


def main():
    host = FLAGS.host
    port = FLAGS.port
    model_name = FLAGS.model_name
    model_version = FLAGS.model_version
    request_timeout = FLAGS.request_timeout

    # Generate inference data
    egg = numpy.asarray([1.1, 2.2, 3.3])
    egg_tensor_proto = tf.contrib.util.make_tensor_proto(egg, dtype=tf.float32)

    bacon = numpy.asarray([7.7, 8.8, 9.9])
    bacon_tensor_proto = tf.contrib.util.make_tensor_proto(bacon, dtype=tf.float32)

    # Create gRPC client and request
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'magic_model'

    if model_version > 0:
        request.model_spec.version.value = model_version
    request.inputs['egg'].CopyFrom(egg_tensor_proto)
    request.inputs['bacon'].CopyFrom(bacon_tensor_proto)

    # Send request
    result = stub.Predict(request, request_timeout)

    # tensorflow_serving.apis.predict_pb2.PredictResponse
    print type(result)

    """
    outputs {
      key: "spam"
      value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
        }
        float_val: 11.0
        float_val: 15.4000005722
        float_val: 19.7999992371
      }
    }
    """
    # print result

    # result.outputs['spam'].float_val is a list
    print result.outputs['spam'].float_val


if __name__ == '__main__':
    main()
