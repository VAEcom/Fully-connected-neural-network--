#coding:utf-8
import time#为了延时，导入time模块
import tensorflow as tf
import tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SEVS=5#完成程序循环间隔是5s

def test(mnist):
    with tf.Graph().as_default() as g:#复现计算图
    x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_=tf.placeholder(tf.float32,[None,mnist_forward.OUTOUT_NODE])
    y=mnist_forward.forward(x,None)

    #实例化带滑动平均率的saver
    ema=tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
    ema_restore=ema.variables_to_restore()
    saver=tf.train.Saver(ema_restore)

    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    while True:
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step=(ckpt.model_checkpoint_path.split('/')[-1].\
                split('-')[-1]
                accuracy_score=sess.run(accuracy,feed_dict={x:mnist.test.images\                ,y_:mnist.test.labels})
                print"After %s training steps,test accuarcy=%g"%(global_step,\                  accuarcy_score)
            else:
                print "NO checkpoint file found"
                return
        time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    test(mnist)

if __name__ ='__main()__':
    main()
