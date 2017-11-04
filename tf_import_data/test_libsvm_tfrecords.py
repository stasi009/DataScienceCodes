import tensorflow as tf
from tqdm import tqdm


def write_libsvm_to_tfrecord(infile, outfile):
    with tf.python_io.TFRecordWriter(outfile) as writer:

        for idx, line in tqdm(enumerate(open(infile, "rt"), start=1)):
            data = line.split(" ")
            label = int(data[0])
            ids = []
            values = []
            for fea in data[1:]:
                id, value = fea.split(":")
                ids.append(int(id))
                values.append(float(value))

            # Write each example one by one
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "ids": tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
                "values": tf.train.Feature(float_list=tf.train.FloatList(value=values))
            }))

            writer.write(example.SerializeToString())

        print "{} records saved".format(idx)


def read_libsvm_from_tfrecord(infile):
    batch_size = 32
    num_threads = 4
    min_after_dequeue = 100
    num_epochs = 1

    filename_queue = tf.train.string_input_producer([infile], num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    batch_serialized_examples = tf.train.shuffle_batch(
        [serialized_example],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue + 3 * batch_size,
        min_after_dequeue=min_after_dequeue)

    features = tf.parse_example(
        batch_serialized_examples,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'index': tf.VarLenFeature(tf.int64),
            'value': tf.VarLenFeature(tf.float32),
        })

    label = features['label']
    index = features['index']
    value = features['value']

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print "queue started"

        try:
            step = 0
            while not coord.should_stop():
                label_, index_, value_ = sess.run([label, index, value])
                if step == 10:
                    print('--label show dense tensor result')
                    print(label_)
                    print(len(label_))
                    print('--show SpareTesnor index_ and value_')
                    print(index_)
                    print(value_)
                    print('--index[0] as indices')
                    # --indices
                    print(index_[0])
                    # --values
                    print('--index[1] as values')
                    print(index_[1])
                    # --shape
                    print('--index[2] as sparse tensor shape')
                    print(index_[2])
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

            # Wait for threads to finish.
        coord.join(threads)


if __name__ == "__main__":
    # write_libsvm_to_tfrecord('temp.libsvm', 'temp.tfrecord')
    read_libsvm_from_tfrecord('temp.tfrecord')
