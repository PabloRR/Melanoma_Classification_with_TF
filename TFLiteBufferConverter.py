
import tensorflow as tf

tf.logging.set_verbosity( tf.logging.ERROR )

keras_model_path = input( "Enter Keras model file path > " )
tflite_file_path = input( "Enter TFLite model file path > " )

converter = tf.lite.TFLiteConverter.from_keras_model_file( keras_model_path )
tflite_buffer = converter.convert()
open( tflite_file_path , 'wb' ).write( tflite_buffer )

print( 'TFLite model created.')

interpreter = tf.lite.Interpreter(model_path=tflite_file_path )
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_input_details())

print(interpreter.get_output_details())
print(interpreter.get_output_details())
