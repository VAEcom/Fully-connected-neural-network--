      import tensorflow as tf
INPUT_NODE =748 #表神经网络输入节点是748（28×28）个
OUTPUT_NODE =10#表输入10个数
LAYER1_NODE=500#隐藏层节点个数为500

def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer !=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return 0

def forward(x,regularizer):
    w1=get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b1=get_bias([LAYER1_NODE])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)

    w2=grt_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b2=get_bias([OUTPUT_NODE])
    y=tf.matmul(y1,w2)+b2
    return y
          
