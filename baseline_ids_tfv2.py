import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import ops
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from sklearn.metrics import confusion_matrix



# set RNG seed
RNG_SEED = 0
tf.keras.utils.set_random_seed(RNG_SEED)
np.random.seed(RNG_SEED)



# load dataset
FEATURE_SCALE = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])

data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)

data_x = data.iloc[:, :-1]
data_x = data_x.to_numpy() / FEATURE_SCALE
data_y = data.iloc[:, -1]
data_y = data_y.to_numpy()

train_x = data_x[:int(0.7*len(data_x))]
train_y = data_y[:int(0.7*len(data_y))]
val_x = data_x[int(0.7*len(data_x)):int(0.7*len(data_x))+1024]
val_y = data_y[int(0.7*len(data_y)):int(0.7*len(data_y))+1024]
test_x = data_x[int(0.7*len(data_x))+1024:]
test_y = data_y[int(0.7*len(data_y))+1024:]

print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'validation')
print(test_x.shape, test_y.shape, 'test')



# define model
hidden_activation = 'relu'
l2_lam = 0.001 # must be defined here because of keras implementation

model_x = layers.Input(shape=(10,), name='baseline_model_input')
model_y = layers.Dense(16, activation=hidden_activation, kernel_regularizer=regularizers.l2(l2_lam), name='baseline_model_hidden1')(model_x)
model_y = layers.Dense(16, activation=hidden_activation, kernel_regularizer=regularizers.l2(l2_lam), name='baseline_model_hidden2')(model_y)
model_y = layers.Dense(16, activation=hidden_activation, kernel_regularizer=regularizers.l2(l2_lam), name='baseline_model_hidden3')(model_y)
model_y = layers.Dense(16, activation=hidden_activation, kernel_regularizer=regularizers.l2(l2_lam), name='baseline_model_hidden4')(model_y)
model_y = layers.Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(l2_lam), name='baseline_model_output')(model_y)

model = tf.keras.Model(model_x, model_y, name='baseline_ids')
model.summary()



# train model
lr = 0.001
epochs = 5
batch_size = 512

model.compile(optimizer=optimizers.AdamW(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_x, val_y))



# evaluate model
test_loss, test_accuracy = model.evaluate(test_x, test_y)
test_cfm = confusion_matrix(test_y, np.argmax(model.predict(test_x), axis=-1), labels=range(5))

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(test_cfm)
# Test loss: 0.07621482759714127
# Test accuracy: 0.9943768978118896
# [[4550742   10238    3014    3477     372]  normal
 # [      0  176506       0       0       0]  dos
 # [  11776       0  133867       8     729]  fuzzy
 # [      0       0       0  179474       0]  gear
 # [      0       0       0       0  196299]] rpm



# plot results
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation', c='r')
plt.scatter([epochs-1], [test_loss], marker='x', label='test', c='g')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
#plt.savefig(f'baseline_ids_tfv2_[acc{test_accuracy:4f}]'.replace('.','_')+'.png')



# save model
#model.save('baseline_ids_tfv2.keras')
model.save_weights('baseline_ids_tfv2.weights.h5')
