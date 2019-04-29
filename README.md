
# Android Project

The Android project is available on GitHub at https://github.com/shubham0204/Skinly_for_Melanoma

# Google Colab Notebook

The Google Colab Notebook is included in the repository as `Melanoma_Classification.ipynb` and can be seen [here](https://colab.research.google.com/drive/1T1SYfzT_Oe974YtI0xqp5tET5dhVo77f).

# Android application

The Android application is available on the Google Play Store at https://play.google.com/store/apps/details?id=com.health.inceptionapps.skinly .

# Data

The images used for training the model were extracted from the internet and are available in NumPy array form ( .npy ) at https://github.com/shubham0204/Dataset_Archives/blob/master/dis_recog_imgs_processed.zip

# About The Files

1. `DataProcessor.py` : To augment and process the images stored in two directories in the master directory `images`.
2. `MainFile` : Brings the model and data together for training and evaluation.
3. `Model.py` : Defines the Keras model used for classification.
4. `TFLiteBufferConverter` : Converts the Keras model ( .h5 ) to a TensorFlow Lite model ( .tflite ).


