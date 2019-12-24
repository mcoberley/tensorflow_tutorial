mcoberley@Michaels-MBP ~ % ipython
Python 3.7.4 (default, Aug 13 2019, 15:17:50) 
Type 'copyright', 'credits' or 'license' for more information
IPython 7.8.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from __future__ import absolute_import, division, print_function, unicode_literals                                                  

In [2]: import tensorflow as tf                                                                                                             

In [3]: from tensorflow import keras                                                                                                        

In [4]: import numpy as np                                                                                                                  

In [5]: import matplotlib.pyplot as plt                                                                                                     

In [6]: print(tf.__version__)                                                                                                               
2.0.0

In [7]: fashion_mnist = keras.datasets.fashion_mnist                                                                                        

In [8]: (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()                                                                                                                
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 1us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 20s 1us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 1us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 3s 1us/step

In [9]: class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
   ...:                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']                                                                   

In [10]: train_images.shape                                                                                                                 
Out[10]: (60000, 28, 28)

In [11]: train_labels                                                                                                                       
Out[11]: array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)

In [12]: plt.figure()                                                                                                                       
Out[12]: <Figure size 1280x960 with 0 Axes>

In [13]: plt.imshow(train_images[0])                                                                                                        
Out[13]: <matplotlib.image.AxesImage at 0x63c3bae10>

In [14]: plt.colorbar()                                                                                                                     
Out[14]: <matplotlib.colorbar.Colorbar at 0x633966d10>

In [15]: plt.grid(False)                                                                                                                    

In [16]: plt.show()                                                                                                                         

In [17]: train_images = train_images / 255.0                                                                                                

In [18]: test_images = test_images / 255.0                                                                                                  

In [19]: plt.figure(figsize=(10,10)) 
    ...: for i in range(25): 
    ...:     plt.subplot(5,5,i+1) 
    ...:     plt.xticks([]) 
    ...:     plt.yticks([]) 
    ...:     plt.grid(False) 
    ...:     plt.imshow(train_images[i], cmap=plt.cm.binary) 
    ...:     plt.xlabel(class_names[train_labels[i]]) 
    ...: plt.show()                                                                                                                         

In [20]: model = keras.Sequential([ 
    ...:     keras.layers.Flatten(input_shape=(28, 28)), 
    ...:     keras.layers.Dense(128, activation='relu'), 
    ...:     keras.layers.Dense(10, activation='softmax') 
    ...:     ])                                                                                     
2019-12-23 18:37:10.656797: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-12-23 18:37:10.693541: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fbefa2ce680 executing computations on platform Host. Devices:
2019-12-23 18:37:10.693598: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version

In [21]: model.compile(optimizer='adam', 
    ...:     loss='sparse_categorical_crossentropy', 
    ...:     metrics=['accuracy'])                                                                  

In [22]: model.fit(train_images, train_labels, epochs=10)                                           
Train on 60000 samples
Epoch 1/10
60000/60000 [==============================] - 4s 64us/sample - loss: 0.4968 - accuracy: 0.8270
Epoch 2/10
60000/60000 [==============================] - 3s 42us/sample - loss: 0.3784 - accuracy: 0.8639
Epoch 3/10
60000/60000 [==============================] - 2s 37us/sample - loss: 0.3379 - accuracy: 0.8764
Epoch 4/10
60000/60000 [==============================] - 2s 38us/sample - loss: 0.3140 - accuracy: 0.8845
Epoch 5/10
60000/60000 [==============================] - 2s 37us/sample - loss: 0.2944 - accuracy: 0.8922
Epoch 6/10
60000/60000 [==============================] - 2s 37us/sample - loss: 0.2794 - accuracy: 0.8967
Epoch 7/10
60000/60000 [==============================] - 4s 65us/sample - loss: 0.2677 - accuracy: 0.9009
Epoch 8/10
60000/60000 [==============================] - 2s 41us/sample - loss: 0.2555 - accuracy: 0.9052
Epoch 9/10
60000/60000 [==============================] - 3s 42us/sample - loss: 0.2463 - accuracy: 0.9088
Epoch 10/10
60000/60000 [==============================] - 6s 94us/sample - loss: 0.2407 - accuracy: 0.9110
Out[22]: <tensorflow.python.keras.callbacks.History at 0x63dbf1150>

In [23]: test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)                  
10000/1 - 4s - loss: 0.2203 - accuracy: 0.8808

In [24]: print('\nTest accuracy:', test_acc)                                                        

Test accuracy: 0.8808

In [25]: predictions = model.predict(test_images)                                                   

In [26]: predictions[0]                                                                             
Out[26]: 
array([2.1139114e-07, 6.3271813e-08, 1.8707894e-08, 1.9136385e-06,
       1.0028897e-06, 5.5475910e-03, 3.5629384e-06, 3.6702756e-02,
       1.7106043e-08, 9.5774293e-01], dtype=float32)

In [27]: np.argmax(predictions[0])                                                                  
Out[27]: 9

In [28]: print(test_labels[np.argmax(predictions[0])])                                              
7

In [29]: def plot_image(i, predictions_array, true_label, img): 
    ...:   predictions_array, true_label, img = predictions_array, true_label[i], img[i] 
    ...:   plt.grid(False) 
    ...:   plt.xticks([]) 
    ...:   plt.yticks([]) 
    ...:  
    ...:   plt.imshow(img, cmap=plt.cm.binary) 
    ...:  
    ...:   predicted_label = np.argmax(predictions_array) 
    ...:   if predicted_label == true_label: 
    ...:     color = 'blue' 
    ...:   else: 
    ...:     color = 'red' 
    ...:  
    ...:   plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 
    ...:                                 100*np.max(predictions_array), 
    ...:                                 class_names[true_label]), 
    ...:                                 color=color) 
    ...:  
    ...: def plot_value_array(i, predictions_array, true_label): 
    ...:   predictions_array, true_label = predictions_array, true_label[i] 
    ...:   plt.grid(False) 
    ...:   plt.xticks(range(10)) 
    ...:   plt.yticks([]) 
    ...:   thisplot = plt.bar(range(10), predictions_array, color="#777777") 
    ...:   plt.ylim([0, 1]) 
    ...:   predicted_label = np.argmax(predictions_array) 
    ...:  
    ...:   thisplot[predicted_label].set_color('red') 
    ...:   thisplot[true_label].set_color('blue') 
    ...:                                                                                            

In [30]: i = 0                                                                                      

In [31]: plt.figure(figsize=(6,3))                                                                  
Out[31]: <Figure size 1200x600 with 0 Axes>

In [32]: plt.subplot(1,2,1)                                                                         
Out[32]: <matplotlib.axes._subplots.AxesSubplot at 0x6392d2e10>

In [33]: plot_image(i, predictions[i], test_labels, test_images)                                    

In [34]: plt.subplot(1,2,2)                                                                         
Out[34]: <matplotlib.axes._subplots.AxesSubplot at 0x6393d9450>

In [35]: plot_value_array(i, predictions[i], test_labels)                                           

In [36]: plt.show()                                                                                 

In [37]: i = 12 
    ...: plt.figure(figsize=(6,3)) 
    ...: plt.subplot(1,2,1) 
    ...: plot_image(i, predictions[i], test_labels, test_images) 
    ...: plt.subplot(1,2,2) 
    ...: plot_value_array(i, predictions[i],  test_labels) 
    ...: plt.show()                                                                                 

In [38]: # Plot the first X test images, their predicted labels, and the true labels. 
    ...: # Color correct predictions in blue and incorrect predictions in red. 
    ...: num_rows = 5 
    ...: num_cols = 3 
    ...: num_images = num_rows*num_cols 
    ...: plt.figure(figsize=(2*2*num_cols, 2*num_rows)) 
    ...: for i in range(num_images): 
    ...:   plt.subplot(num_rows, 2*num_cols, 2*i+1) 
    ...:   plot_image(i, predictions[i], test_labels, test_images) 
    ...:   plt.subplot(num_rows, 2*num_cols, 2*i+2) 
    ...:   plot_value_array(i, predictions[i], test_labels) 
    ...: plt.tight_layout() 
    ...: plt.show()                                                                                 

In [39]: img = test_images[1]                                                                       

In [40]: print(img.shape)                                                                           
(28, 28)

In [41]: img = (np.expand_dims(img, 0))                                                             

In [42]: print(img.shape)                                                                           
(1, 28, 28)

In [43]: predictions_single = model.predict(img)                                                    

In [44]: print(predictions_single)                                                                  
[[5.3045201e-06 1.0881648e-14 9.9993551e-01 6.0344195e-12 5.3667332e-05
  4.9328447e-10 5.4621410e-06 3.8146114e-19 5.8522248e-10 1.2310194e-18]]

In [45]: plot_value_array(1, predictions_single[0], test_labels)                                    

In [46]: _ = plt.xticks(range(10), class_names, rotation=45)                                        

In [47]: np.argmax(predictions_single[0])                                                           
Out[47]: 2

In [48]:                                                                                            
