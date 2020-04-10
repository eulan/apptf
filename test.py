import pickle
import tensorflow as tf
import numpy as np
import os
import re, string
import preprocessor as p

with open("modelsChat/vocabulario.pickle", 'rb') as handle:
  vocabulario = pickle.load(handle)
with open('modelsChat/numatext.pickle', 'rb') as handle:
  NumAtext = pickle.load(handle)
with open('modelsChat/textanum.pickle', 'rb') as handle:
  textANum = pickle.load(handle)

def builder_model(vocab_size, embedding_dim, rnn_units,batch_size):
  model = tf.keras.Sequential([
                               tf.keras.layers.Embedding(
                                   vocab_size,
                                   embedding_dim, 
                                   batch_input_shape = [batch_size,None]),
                               tf.keras.layers.GRU(
                                   rnn_units,
                                   return_sequences=True, 
                                   stateful=True,
                                   recurrent_initializer = 'glorot_uniform'),
                               tf.keras.layers.LSTM(
                                   rnn_units,
                                   return_sequences=True, 
                                   stateful=True,
                                   recurrent_initializer = 'glorot_uniform'),
                               tf.keras.layers.Dense(vocab_size)
  ])
  return model


class ModelGenerator:  

    

  def __init__(self,text):
    vocab_size = len(vocabulario)
    embedding_dim = 512
    rnn_units = 256

    checkpoint_dir = './modelsChat'
    checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt_{epoch}')
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

    tf.train.latest_checkpoint(checkpoint_dir)

    model = builder_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    self.text = text
    self.model = model

  def transformer(self):
    self.text= p.clean(self.text)
    self.text = re.sub(r'\W+',' ',self.text)
    self.text = self.text.lower()
    self.text = str(self.text)  

def generate_text(model, start_string,num_generate = 1000):
  input_eval = [textANum[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(NumAtext[predicted_id])
  return (start_string + ''.join(text_generated))
