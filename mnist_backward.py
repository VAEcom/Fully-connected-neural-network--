import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

STEPS=50000
MODEL_SAVE_PATH='./model/'#模型保存路径
MODEL_NAME='mnist_model'#模型保存路径名
BATCH_SIZE=200
LEARNING_RATE_BASE=0.1
LEARNING_RATE_DECAY=0.99
REGULARIZER=0.0001
MOVING_AVERAGE_DECAY=0.99

def backward(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_=tf.placeholder(tf.float32,[None,mnist_forward.OUYPUT_NODE])
    y=mnist_foward.forward(x,REGULARIZER)
    global_step=tf.Variable(0,trainable=False)

    #定义包含正则化的损失函数
    ce=tf.nn.Sparse_softmax_cross_entropy_with_logits(logits=y,labels=\
    tf.argmax(y_,1)
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses')

    #指数衰减学习率
    learning_rate=tf.train.exponential_decay(\
    LEARING_RATE_BASE,
    global_step,
    mnist.train.num_examples/BATCH_SIZE,
    LEARINNG_RATE_DECAY,
    staircase=True)

    #训练方法
    train_step=tf.train.GradientDescentOptimizer(learinng_rate).minimize(loss,global_step=global_step)

    #滑动平均率
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())#可自动把待训练的参数汇总成列表并>求滑动平均
    with tf.control_dependencies（[train_step,ema_op]):#把训练过程和计算滑动平均
绑定在一起运行
        train_op=tf.no_op(name='train')#把他们合成一个训练节点

    saver=tf.train.Saver()#实例化saver对象

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            #从训练集中随机抽取BATCH_SIZE组数据和标签，分别赋给xs,ys
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],\
            feed_dict={x:xs,y_=ys})
            if i%1000==0:
                print"After %d training step(s),loss on training batch\
                is %g"%(step,loss_value)
                #保存模型至该路径，并在文件尾加上当前轮数
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),\
                global_step=global_step)

def main():
    #使用该函数自动加载数据集，告知已读热码的形式读取
    mnist=input_data.read_data_sets("./data/",one_hot=TRUE)
    backward(mnist)

#判断python运行的文件是否是主文件,如果是主文件，则执行main()函数
if __name__='__main()':
    main()
