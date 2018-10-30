import tensorflow as tf

start = tf.Variable(0,dtype=tf.int64)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1,201):
        t = tf.constant(i,dtype=tf.int64)
        new_value = tf.add(start,t)
        update = tf.assign(start,new_value)
        sess.run(update)
        print(t.eval(),':',sess.run(start))