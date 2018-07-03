

def get_optimizer(loss):
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    # 为了避免梯度爆炸的问题，我们求出梯度的二范数。
    # 然后判断该二范数是否大于1.25，若大于，则变成
    # gradients * (1.25 / global_norm)作为当前的gradients
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    # 将刚刚求得的梯度组装成相应的梯度下降法
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    return optimizer, learning_rate
 
def logprob(predictions, labels):
    # 计算交叉熵
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

loadData = LoadData()
train_text = loadData.train_text
valid_text = loadData.valid_text
 
train_batcher = BatchGenerator(text=train_text, batch_size=config.batch_size, num_unrollings=config.num_unrollings)
vaild_batcher = BatchGenerator(text=valid_text, batch_size=1, num_unrollings=1)
 
# 定义训练数据由num_unrollings个占位符组成
train_data = list()
for _ in range(config.num_unrollings + 1):
    train_data.append(
        tf.placeholder(tf.float32, shape=[config.batch_size, config.vocabulary_size]))
 
train_input = train_data[:config.num_unrollings]
train_label= train_data[1:]
 
# define the lstm train model
model = LSTM_Cell(
    train_data=train_input,
    train_label=train_label)
# get the loss and the prediction
logits, loss, train_prediction = model.loss_func()
optimizer, learning_rate = get_optimizer(loss)
