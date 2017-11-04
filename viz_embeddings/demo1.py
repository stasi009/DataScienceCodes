import os
import math
import numpy as np
import tensorflow as tf
import download
from tensorflow.contrib.tensorboard.plugins import projector

tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """Path to model.""")

# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "classify_image_graph_def.pb"

# Directory to store the downloaded data.
data_dir = "inception/"

IMAGE_DIR = 'img'
LOG_DIR = 'logs'


def maybe_download():
    """
    Download the Inception model from the internet if it does not already
    exist in the data_dir. The file is about 85 MB.
    """

    print("Downloading Inception v3 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


def load_graph():
    # Create a new TensorFlow computational graph.
    graph = tf.Graph()

    # Set the new graph as the default.
    with graph.as_default():
        # Open the graph-def file for binary reading.
        path = os.path.join(data_dir, path_graph_def)
        with tf.gfile.FastGFile(path, 'rb') as file:
            # The graph-def is a saved copy of a TensorFlow graph.
            # First we need to create an empty graph-def.
            graph_def = tf.GraphDef()

            # Then we load the proto-buf file into the graph-def.
            graph_def.ParseFromString(file.read())

            # Finally we import the graph-def to the default TensorFlow graph.
            tf.import_graph_def(graph_def, name='')
    return graph


def main(argv=None):
    maybe_download()
    graph = load_graph()

    basedir = os.path.dirname(__file__)

    # ensure log directory exists
    logs_path = os.path.join(basedir, LOG_DIR)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    with tf.Session(graph=graph) as sess:

        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        jpeg_data = tf.placeholder(tf.string)
        thumbnail = tf.cast(tf.image.resize_images(
            tf.image.decode_jpeg(jpeg_data, channels=3), [100, 100]), tf.uint8)

        outputs = []
        images = []

        # Create metadata
        metadata_path = os.path.join(basedir, LOG_DIR, 'metadata.tsv')
        metadata = open(metadata_path, 'w')
        metadata.write("Name\tLabels\n")

        for folder_name in os.listdir(IMAGE_DIR):
            for file_name in os.listdir(IMAGE_DIR + '/' + folder_name):
                if not file_name.endswith('.jpg'):
                    continue
                print('process %s...' % file_name)

                with open(os.path.join(basedir, IMAGE_DIR + '/' + folder_name, file_name), 'rb') as f:
                    data = f.read()
                    results = sess.run([pool3, thumbnail], {
                        'DecodeJpeg/contents:0': data, jpeg_data: data})
                    outputs.append(results[0])
                    images.append(results[1])
                    metadata.write('{}\t{}\n'.format(file_name, folder_name))
        metadata.close()

        embedding_var = tf.Variable(tf.stack(
            [tf.squeeze(x) for x in outputs], axis=0), trainable=False, name='embed')

        # prepare projector config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        summary_writer = tf.summary.FileWriter(os.path.join(basedir, LOG_DIR))

        # link metadata
        embedding.metadata_path = metadata_path

        # write to sprite image file
        image_path = os.path.join(basedir, LOG_DIR, 'sprite.jpg')
        size = int(math.sqrt(len(images))) + 1
        while len(images) < size * size:
            images.append(np.zeros((100, 100, 3), dtype=np.uint8))
        rows = []
        for i in range(size):
            rows.append(tf.concat(images[i * size:(i + 1) * size], 1))
        jpeg = tf.image.encode_jpeg(tf.concat(rows, 0))
        with open(image_path, 'wb') as f:
            f.write(sess.run(jpeg))

        embedding.sprite.image_path = image_path
        embedding.sprite.single_image_dim.extend([100, 100])

        # save embedding_var
        projector.visualize_embeddings(summary_writer, config)
        sess.run(tf.variables_initializer([embedding_var]))

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(basedir, LOG_DIR, 'model.ckpt'))


if __name__ == '__main__':
    tf.app.run()
