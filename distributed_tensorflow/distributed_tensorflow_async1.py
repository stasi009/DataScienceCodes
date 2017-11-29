# -*- coding: utf-8 -*-

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 和5.5节中类似的配置神经网络的设置。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
# 模型保存的路径。
MODEL_SAVE_PATH = "/path/to/model"
# MNIST数据路径。
DATA_PATH = "/path/to/data"

# 通过flags指定运行的参数。在10.4.1小节中对于不同的任务（task）给出了不同的程序，
# 但这不是一种可扩展的方式。在这一小节中将使用运行程序时给出的参数来配置在不同
# 任务中运行的程序。
FLAGS = tf.app.flags.FLAGS

# 指定当前运行的是参数服务器还是计算服务器。参数服务器只负责TensorFlow中变量的维护
# 和管理，计算服务器负责每一轮迭代时运行反向传播过程。
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')

# 指定集群中的参数服务器地址。
tf.app.flags.DEFINE_string(
    'ps_hosts', ' tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. '
    'e.g. "tf-ps0:2222,tf-ps1:1111" ')

# 指定集群中的计算服务器地址。
tf.app.flags.DEFINE_string(
    'worker_hosts', ' tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. '
    'e.g. "tf-worker0:2222,tf-worker1:1111" ')

# 指定当前程序的任务ID。TensorFlow会自动根据参数服务器/计算服务器列表中的端口号
# 来启动服务。注意参数服务器和计算服务器的编号都是从0开始的。
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# 定义TensorFlow的计算图，并返回每一轮迭代时需要运行的操作。这个过程和5.5节中的主
# 函数基本一致，但为了使处理分布式计算的部分更加突出，本小节将此过程整理为一个函数。
def build_model(x, y_, is_chief):
   regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
   # 通过和5.5节给出的mnist_inference.py代码计算神经网络前向传播的结果。
   y = mnist_inference.inference(x, regularizer)
   global_step = tf.Variable(0, trainable=False)

   # 计算损失函数并定义反向传播过程。
   cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        y, tf.argmax(y_, 1))
   cross_entropy_mean = tf.reduce_mean(cross_entropy)
   loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
   learning_rate = tf.train.exponential_decay(
       LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE,
        LEARNING_RATE_ DECAY)

   # 定义每一轮迭代需要运行的操作。
   train_op = tf.train.GradientDescentOptimizer(learning_rate)\
                   .minimize(loss, global_step=global_step)
   return global_step, loss, train_op

# 训练分布式深度学习模型的主过程。
def main(argv=None):
   # 解析flags并通过tf.train.ClusterSpec配置TensorFlow集群。
   ps_hosts = FLAGS.ps_hosts.split(',')
   worker_hosts = FLAGS.worker_hosts.split(',')
   cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
   # 通过ClusterSpec以及当前任务创建Server。
   server = tf.train.Server(
       cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

   # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。server.join()
   # 会一直停在这条语句上。
   if FLAGS.job_name == 'ps':
       server.join()

   # 定义计算服务器需要运行的操作。在所有的计算服务器中有一个是主计算服务器，它除了负责
   # 计算反向传播的结果，它还负责输出日志和保存模型。
   is_chief = (FLAGS.task_id == 0)
   mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

   # 通过tf.train.replica_device_setter函数来指定执行每一个运算的设备。
   # tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，而
   # 计算分配到当前的计算服务器上。图10-9展示了通过TensorBoard可视化得到的第一个计
   # 算服务器上运算分配的结果。
   with tf.device(tf.train.replica_device_setter(
           worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster)):
       x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE],
            name='x-input')
       y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE],
            name='y-input')
        # 定义训练模型需要运行的操作。
       global_step, loss, train_op = build_model(x, y_)

        # 定义用于保存模型的saver。
       saver = tf.train.Saver()
        # 定义日志输出操作。
       summary_op = tf.merge_all_summaries()
        # 定义变量初始化操作。
       init_op = tf.initialize_all_variables()
        # 通过tf.train.Supervisor管理训练深度学习模型的通用功能。
        # tf.train. Supervisor能统一管理队列操作、模型保存、日志输出以及会话的生成。
       sv = tf.train.Supervisor(
           is_chief=is_chief,           # 定义当前计算服务器是否为主计算服务器，只有
                                            # 主计算服务器会保存模型以及输出日志。
         logdir=MODEL_SAVE_PATH,    # 指定保存模型和输出日志的地址。
         init_op=init_op,               # 指定初始化操作。
         summary_op=summary_op,     # 指定日志生成操作。
         saver = saver,                 # 指定用于保存模型的saver。
         global_step=global_step,   # 指定当前迭代的轮数，这个会用于生成保存模
                                            # 型文件的文件名。
         save_model_secs=60,            # 指定保存模型的时间间隔。
         save_summaries_secs=60)    # 指定日志输出的时间间隔。

       sess_config = tf.ConfigProto(allow_soft_placement=True,
                                           log_device_placement=False)
        # 通过tf.train.Supervisor生成会话。
       sess = sv.prepare_or_wait_for_session(
             server.target, config=sess_config)

       step = 0
       start_time = time.time()
        # 执行迭代过程。在迭代过程中tf.train.Supervisor会帮助输出日志并保存模型，
        # 所以不需要直接调用这些过程。
       while not sv.should_stop():
           xs, ys = mnist.train.next_batch(BATCH_SIZE)
           _, loss_value, global_step_value = sess.run(
                 [train_op, loss, global_step], feed_dict={x: xs, y_: ys})
           if global_step_value >= TRAINING_STEPS: break

             # 每隔一段时间输出训练信息。
           if step > 0 and step % 100 == 0:
               duration = time.time() - start_time
                  # 不同的计算服务器都会更新全局的训练轮数，所以这里使用
                  # global_step_value可以直接得到在训练中使用过的batch的总数。
               sec_per_batch = duration / global_step_value
               format_str = ("After %d training steps (%d global steps), "
                                  "loss on training batch is %g.  " 
                                  "(%.3f sec/batch)")
               print(format_str % (step, global_step_value,
                                          loss_value, sec_per_batch))
           step += 1
   sv.stop()

if __name__ == "__main__":
   tf.app.run()