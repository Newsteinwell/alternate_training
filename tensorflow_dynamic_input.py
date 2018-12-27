import numpy as np
import tensorflow as tf
import pickle
class TestDynamicInput():
    def __init__(self, batch_size=10, softID_dim=1, epochs=50, start_alternate_train=30):
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.epochs = epochs
        self.softID_dim = softID_dim
        self.start_alternate_train = start_alternate_train
        self.x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        self.y = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        self.softID = tf.placeholder(shape=(None, self.softID_dim), dtype=tf.float32)
        self.soft_train = tf.placeholder(shape=(), dtype=tf.bool)
        self.out, self.softID_trained = self.build_model()
        self.loss = self.get_loss(self.y, self.out)
        self.op_all = tf.train.AdamOptimizer(learning_rate=0.05).minimize(self.loss)
        train_vars = tf.trainable_variables()
        nonsoft_vars = [var for var in train_vars if 'soft' not in var.name]
        self.op_nonsoft = tf.train.AdamOptimizer(learning_rate=0.05).minimize(self.loss, var_list=nonsoft_vars)
        soft_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softID')
        self.op_soft = tf.train.AdamOptimizer(learning_rate=0.05).minimize(self.loss, var_list=soft_vars)

    def gen_data(self):
        all_len = 10000
        train_len = np.int(all_len*0.8)
        x_original = np.random.random_sample(size=(all_len, 3))
        w_original = np.array([[1], [2], [3]])
        y = np.matmul(x_original, w_original)

        x_ = x_original[:,:-1]

        x_original_train = x_original[:train_len]
        x_train = x_[:train_len]
        y_train = y[:train_len]
        x_original_test = x_original[train_len:]
        x_test = x_[:train_len:]
        y_test = y[:train_len:]
        return (x_original_train, x_train, y_train, x_original_test, x_test, y_test)

    def build_model(self):
        x = self.x
        x_dim = x.get_shape().as_list()[1]
        with tf.variable_scope('softID') as sp:
            softID_train = tf.Variable(tf.zeros((1, self.softID_dim)), name='softid')
        softID_train_ = tf.tile(softID_train, [tf.shape(x)[0], 1])
        input_ = tf.cond(self.soft_train, lambda: tf.concat([x, softID_train_], axis=1),
                         lambda: tf.concat([x, self.softID], axis=1))
        input_dim = x_dim + self.softID_dim

        with tf.variable_scope('first_layer') as sp1:
            layers1 = tf.layers.dense(input_, units=input_dim*2,
                                      name='first_layer')
        with tf.variable_scope('output') as spo:
            output = tf.layers.dense(input_, units=input_dim*2,
                                      name='output_layer')
        return (output, softID_train_)

    def get_loss(self, y, y_):
        loss = tf.sqrt(tf.reduce_mean(tf.square(y-y_)))
        return (loss)

    def train(self, X, y, softID, epoch):
        if  epoch <= self.start_alternate_train or epoch % 2 == 0:
            _, loss, softID_trained = self.sess.run([self.op_nonsoft, self.loss, self.softID_trained], feed_dict={self.x: X, self.y:y,
                                                    self.softID:softID, self.soft_train:False})

        elif epoch % 2 == 1:
            _, loss, softID_trained = self.sess.run([self.op_soft, self.loss, self.softID_trained], feed_dict={self.x: X, self.y:y,
                                                    self.softID:softID, self.soft_train:True})
        return (loss, softID_trained)

    def fit(self, X, y):
        self.sess.run(tf.global_variables_initializer())
        loss_result = {}
        sample_num = X.shape[0]
        idx = np.arange(sample_num)
        for epoch in range(self.epochs):
            np.random.shuffle(idx)
            for i in range(sample_num // self.batch_size):
                idx_batch = idx[i*self.batch_size : (i+1)*self.batch_size]
                X_batch = X[idx_batch]
                y_batch = y[idx_batch]
                soft_batch = np.zeros((X_batch.shape[0],1))
                loss, softID_trained = self.train(X=X_batch, y=y_batch, softID=soft_batch, epoch=epoch)

                if i%50 == 1:  print ('epoch: {}, step: {}, loss: {}'.format(epoch, i, loss))
                if i%50 == 1:  print ('softID_trained: {}'.format(softID_trained))
            loss_result[epoch] = loss

        self.save_obj(loss_result, 'loss_result')
    def predict(self, X):
        softID = np.zeros((X.shape[0],1))
        y_pred = self.sess.run([self.out], feed_dict={self.x: X,
                                             self.softID:softID, self.soft_train:True})
        return (y_pred)

    def save_obj(self, obj, name):
        with open('./result/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


test_ = TestDynamicInput()
x_or_train, x_train, y_train, x_or_test, x_test, y_test = test_.gen_data()
test_.fit(X=x_train, y=y_train)
y_pred = test_.predict(X=x_train)

loss = test_.get_loss(y=y_test, y_=y_pred)
print (loss)

