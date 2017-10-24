# -*- coding: utf-8 -*-

from datetime import datetime
import os
import time

import tensorflow as tf
import mnist_inference

# 定义训练神经网络时需要用到的配置。这些配置与5.5节中定义的配置类似。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 4

# 定义日志和模型输出的路径。
MODEL_SAVE_PATH = "/path/to/logs_and_models/"
MODEL_NAME = "model.ckpt"

# 定义数据存储的路径。因为需要为不同的GPU提供不同的训练数据，所以通过placerholder
# 的方式就需要手动准备多份数据。为了方便训练数据的获取过程，可以采用第7章中介绍的输
# 入队列的方式从TFRecord中读取数据。于是在这里提供的数据文件路径为将MNIST训练数据
# 转化为TFRecords格式之后的路径。如何将MNIST数据转化为TFRecord格式在第7章中有
# 详细介绍，这里不再赘述。
DATA_PATH = "/path/to/data.tfrecords"

# 定义输入队列得到训练数据，具体细节可以参考第7章。
def get_input():
    filename_queue = tf.train.string_input_producer([DATA_PATH])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 定义数据解析格式。
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    # 解析图片和标签信息。
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [784])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # 定义输入队列并返回。
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    return tf.train.shuffle_batch(
        [retyped_image, label],
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

# 定义损失函数。对于给定的训练数据、正则化损失计算规则和命名空间，计算在这个命名空间
# 下的总损失。之所以需要给定命名空间是因为不同的GPU上计算得出的正则化损失都会加入名为
# loss的集合，如果不通过命名空间就会将不同GPU上的正则化损失都加进来。
def get_loss(x, y_, regularizer, scope):
    # 沿用5.5节中定义的函数来计算神经网络的前向传播结果。
    y = mnist_inference.inference(x, regularizer)
    # 计算交叉熵损失。
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    # 计算当前GPU上计算得到的正则化损失。
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    # 计算最终的总损失。
    loss = cross_entropy + regularization_loss
    return loss

    # 计算每一个变量梯度的平均值。
    def average_gradients(tower_grads):
      average_grads = []
      # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
      for grad_and_vars in zip(*tower_grads):
          # 计算所有GPU上的梯度平均值。
          grads = []
          for g, _ in grad_and_vars:
              expanded_g = tf.expand_dims(g, 0)
              grads.append(expanded_g)
          grad = tf.concat(0, grads)
          grad = tf.reduce_mean(grad, 0)

          v = grad_and_vars[0][1]
          grad_and_var = (grad, v)
          # 将变量和它的平均梯度对应起来。
          average_grads.append(grad_and_var)
    # 返回所有变量的平均梯度，这将被用于变量更新。
    return average_grads

# 主训练过程。
def main(argv=None):
    # 将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上。
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 获取训练batch。
        x, y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

        # 定义训练轮数和指数衰减的学习率。
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0),
             trainable=False)
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE,
            LEARNING_ RATE_DECAY)

        # 定义优化方法。
        opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        # 将神经网络的优化过程跑在不同的GPU上。
        for i in range(N_GPU):
            # 将优化过程指定在一个GPU上。
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    cur_loss = get_loss(x, y_, regularizer, scope)
                    # 在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以
                    # 让不同的GPU更新同一组参数。注意tf.name_scope函数并不会影响
                    # tf.get_ variable的命名空间。
                    tf.get_variable_scope().reuse_variables()

                    # 使用当前GPU计算所有变量的梯度。
                    grads = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)

        # 计算变量的平均梯度，并输出到TensorBoard日志中。
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(
                    'gradients_on_average/%s' % var.op.name, grad)

        # 使用平均梯度更新参数。
        apply_gradient_op = opt.apply_gradients(
            grads, global_step=global_ step)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # 计算变量的滑动平均值。
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # 每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()

        # 训练过程。
        with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)) as sess:
            # 初始化所有变量并启动队列。
            init.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.train.SummaryWriter(
                MODEL_SAVE_PATH, sess.graph)

            for step in range(TRAINING_STEPS):
                # 执行神经网络训练操作，并记录训练操作的运行时间。
                start_time = time.time()
                _, loss_value = sess.run([train_op, cur_loss])
                duration = time.time() - start_time

                # 每隔一段时间展示当前的训练进度，并统计训练速度。
                if step != 0 and step % 10 == 0:
                    # 计算使用过的训练数据个数。因为在每一次运行训练操作时，每一个GPU
                    # 都会使用一个batch的训练数据，所以总共用到的训练数据个数为
                    # batch大小×GPU个数。
                    num_examples_per_step = BATCH_SIZE * N_GPU

                    # num_examples_per_step为本次迭代使用到的训练数据个数，
                    # duration为运行当前训练过程使用的时间，于是平均每秒可以处理的训
                    # 练数据个数为num_examples_per_step / duration。
                    examples_per_sec = num_examples_per_step / duration

                    # duration为运行当前训练过程使用的时间，因为在每一个训练过程中，
                    # 每一个GPU都会使用一个batch的训练数据，所以在单个batch上的训
                    # 练所需要时间为duration / GPU个数。
                    sec_per_batch = duration / N_GPU

                    # 输出训练信息。
                    format_str = ('step %d, loss = %.2f (%.1f examples/ '
                                    ' sec; %.3f sec/batch)')
                    print(format_str % (step, loss_value,
                                            examples_per_sec, sec_per_batch))

                    # 通过TensorBoard可视化训练过程。
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)

                # 每隔一段时间保存当前的模型。
                if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                    checkpoint_path = os.path.join(
                        MODEL_SAVE_PATH, MODEL_ NAME)
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
tf.app.run()

'''
在AWS的g2.8xlarge实例上运行上面这段程序可以得到类似下面的结果：
step 10, loss = 71.90 (15292.3 examples/sec; 0.007 sec/batch)
step 20, loss = 37.97 (18758.3 examples/sec; 0.005 sec/batch)
step 30, loss = 9.54 (16313.3 examples/sec; 0.006 sec/batch)
step 40, loss = 11.84 (14199.0 examples/sec; 0.007 sec/batch)
...
step 980, loss = 0.66 (15034.7 examples/sec; 0.007 sec/batch)
step 990, loss = 1.56 (16134.1 examples/sec; 0.006 sec/batch)
'''