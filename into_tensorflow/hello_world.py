try:
    import numpy as np
    import pandas as pd
    import keras
    import tensorflow as tf
except ImportError as e:
    print(e)

## defining the input data

xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0],dtype=float)
ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5],dtype=float)

## defining the model architecture

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(xs,ys,epochs=500)

## predicting the value
predicted_value = model.predict([7.0])[0]
print("A house with with 7 rooms will cost:",predicted_value)



