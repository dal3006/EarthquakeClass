"""
        - Implementing a deep convolutional neural network using TensorFlow
            and the Keras API
        - The multilayer CNN architecture
        -> applied to hand-written numbers: mnist

        Architecture:
        7 layers: 1 input, 1 Conv2D, 1 Pool, 1Conv2D, 1 Pool, 1 FC, 1 Dropout, 1 FC

"""
import matplotlib as mpl
#mpl.use( 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io, os
#---------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# learning curve
from sklearn.model_selection import learning_curve

import tensorflow as tf
import time


tf.random.set_seed( 1)
t1 = time.time()
#=================================================
#                parameters
#=================================================
#BUFFER_SIZE    = 10000
BATCH_SIZE     = 50
NUM_EPOCHS     = 10
valid_size     = .1
#----------------------------------------
nFilt          = 8 #increase by factor of two for each CVl
filt_shape     = 3 #size of conv. kernel
p_drop         = 0.1 # dropout rate
nHidden        = 10 #number of units in hiddenlayer
nLabelOut      = 3 #output of last FC layer


#------------------------files and dirs-----------------
model_dir ='models/spec_cnn'
train_dir ='models/spec_cnn/train_results'
# save results during training: loss, accuracy, validation etc
file_train = 'CNN_spec_train'
#=================================================
#            load and split data
#=================================================
data_dir  = f"{os.environ['HOME']}/PycharmProjects/ML_CERI8703/data"
data_file = "labquake_spec.mat"
#===============================================================================
#                 load and pre-process data
#===============================================================================
dSpec = scipy.io.loadmat( f"{data_dir}/{data_file}", squeeze_me = True, struct_as_record = False)
print( dSpec.keys())
X     = dSpec['mSpec'][:,0:-1]
y     = dSpec['mSpec'][:,-1]
# shuffle original dataset
a_ran_int = np.random.randint(0, len(y)-1, len(y))
X     = X[a_ran_int]
y     = y[a_ran_int]

_nx,_ny = dSpec['nx'], dSpec['ny']
print( _nx, _ny)
print( greeg)
Ntot  = X.shape[0]

# def train_input_fn(x_train, y_train, batch_size = None):
#     # reshape each column to 2D and convert to tf dataset
#     dataset = tf.data.Dataset.from_tensor_slices( ({'input-features': x_train},
#                                                    y_train.T,
#                                                    train_input_fn( X_train, y_train, batch_size=BATCH_SIZE)))
#     dataset = tf.reshape(dataset, [_nx, _ny])
#     return dataset.shuffle(100).repeat().batch(batch_size=batch_size)
#
# def eval_input_fn(x_test, y_test, batch_size = None):
#     # reshape each column to 2D and convert to tf dataset
#     dataset = tf.data.Dataset.from_tensor_slices( ({'input-features': x_test.reshape(_nx,_ny)},
#                                                    y_test.reshape(-1,1)))
#     return dataset.batch(batch_size=batch_size)

#------------------standardize data-----------------
for i in range( Ntot):
    X[i] = (X[i] - X[i].mean())/X[i].std()

print( 'total number of instances: ', Ntot, len( y), len( X[:,0]), 'image size: ', _nx, _ny)

# -----------------train, test split----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)
print( f"training size: {len(X_train[:,0])}, testing size: {len(X_test[:,0])}")

#=================================================
#            Configuring CNN layers in Keras
#=================================================
# 9 layers: 1 input, 1 Conv2D, 1 Pool, 1Conv2D, 1 Pool, 1Conv2D, 1 Pool, 2 MLPs

model = tf.keras.Sequential()

model.add( tf.keras.layers.Reshape( (_nx, _ny, 1), input_shape=(_nx*_ny,)) )
#model.add( tf.keras.layers.InputLayer( input_shape=(nx,ny, 1), batch_size=BATCH_SIZE))

model.add(tf.keras.layers.Conv2D(    filters=nFilt,
                                     input_shape=(_nx,_ny, 1),
                                     kernel_size=(filt_shape, filt_shape),
                                    strides=(1, 1), padding='same',
                                    data_format='channels_last',
                                    name='conv_1', activation='relu'))

model.add(tf.keras.layers.MaxPool2D( pool_size=(2, 2), name='pool_1'))

model.add(tf.keras.layers.Conv2D(   filters=nFilt*2, kernel_size=(filt_shape, filt_shape),
                                    strides=(1, 1), padding='same',
                                    name='conv_2', activation='relu'))

model.add(tf.keras.layers.MaxPool2D( pool_size=(2, 2), name='pool_2'))

model.add(tf.keras.layers.Conv2D(   filters=nFilt*4, kernel_size=(filt_shape, filt_shape),
                                    strides=(1, 1), padding='same',
                                    name='conv_3', activation='relu'))

model.add(tf.keras.layers.MaxPool2D( pool_size=(2, 2), name='pool_3'))
# check layer sizes
model.compute_output_shape( input_shape=(BATCH_SIZE, _nx, _ny, 1))
#### fully connected layers
# needs to be flattened first (2D only for Conv2D and Pool2D)
model.add(tf.keras.layers.Flatten())

model.compute_output_shape(input_shape=(BATCH_SIZE, _nx, _ny, 1))

model.add(tf.keras.layers.Dense(    units=nHidden, name='fc_1',
                                    activation='relu'))

model.add(tf.keras.layers.Dropout(  rate=p_drop))

model.add(tf.keras.layers.Dense(    units=nLabelOut, name='fc_2',
                                    activation='softmax'))
# specify input layer shape here instead of in first Conv layer or keras.layers.InputLayer
model.build( input_shape=(BATCH_SIZE, _nx, _ny, 1))

model.compute_output_shape(input_shape=(BATCH_SIZE, _nx, _ny, 1))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) # same as `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`

#my_estimator = tf.keras.estimator.model_to_estimator( keras_model=model, model_dir='models/mnist_cnn')
#=================================================
#               train and evalute
#=================================================

dRes = model.fit(   X_train, y_train,
                    #train_input_fn( X_train.T, y_train, batch_size=BATCH_SIZE),
                    epochs = NUM_EPOCHS,
                     validation_split=valid_size,
                     shuffle = True).history

test_results = model.evaluate( X_test, y_test, batch_size=BATCH_SIZE
                                #eval_input_fn(X_test, y_test, batch_size=BATCH_SIZE)
                               )
print( '---------------sample test predictions:--------------------')
m_p_predict = model.predict( X_test)
a_numbers = np.arange(nLabelOut)
y_predict = np.array([ a_numbers[np.argmax(m_p_predict[i])] for i in range(m_p_predict.shape[0])], dtype=np.int16)
print('\nTest Acc. {:.2f}%'.format(test_results[1]*100))
print( round( accuracy_score( y_test, y_predict), 3))

#=================================================
#             test plot
#=================================================
x_arr = np.arange(len(dRes['loss'])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, dRes['loss'], '-o', label='Train loss')
ax.plot(x_arr, dRes['val_loss'], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, dRes['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, dRes['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.savefig('6a_loss_valid.png', dpi=300)
print( f"run time {round(time.time()-t1 ,1)}")
#---------------correct predictions--------------------------
sel = y_predict == y_test
X_true = X_test[sel]
fig, m_ax = plt.subplots( 5, 5, sharex=True, figsize=(10,6))
for i in range( 5):
    for j in range( 5):
        i_tot = j+i*5
        # check if there are enough False predictions
        if sel.sum()-1 > i_tot:
            ax = m_ax[i,j]
            Sxx = X_true[i_tot].reshape( dSpec['ny'], dSpec['nx'])

            ax.set_title( f"y-p: {y_predict[sel][i_tot]}, y-t={y_test[sel][i_tot]}")
            plot1 = ax.pcolormesh(dSpec['a_t'], dSpec['a_f'], Sxx,
                                  shading = 'nearest',
                                  #shading='gouraud',
                                  cmap = plt.cm.RdYlGn_r,
                                  # norm = norm
                                  )
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
plt.savefig( '6b_spec_predict.png')

#---------------False predictions--------------------------
sel = y_predict != y_test
X_true = X_test[sel]
fig, m_ax = plt.subplots( 5, 5, sharex=True, figsize=(10,6))
for i in range( 5):
    for j in range( 5):
        i_tot = j+i*5
        # check if there are enough False predictions
        if sel.sum()-1 > i_tot:
            ax = m_ax[i,j]
            Sxx = X_true[i_tot].reshape( dSpec['ny'], dSpec['nx'])

            ax.set_title( f"y-p: {y_predict[sel][i_tot]}, y-t={y_test[sel][i_tot]}")
            plot1 = ax.pcolormesh(dSpec['a_t'], dSpec['a_f'], Sxx,
                                  shading = 'nearest',
                                  #shading='gouraud',
                                  cmap = plt.cm.RdYlGn_r,
                                  # norm = norm
                                  )
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
plt.savefig( '6b_spec_false_predict.png')
plt.show()
#=================================================
#         save model and training results
#=================================================
#model.save( model_dir)

#scipy.io.savemat( f"{train_dir}/{file_train}.mat", dRes, do_compression=True, format='5')
#utils.saveHistory( dRes, f"models/mnist_cnn/train_result/{file_train}.dic")
