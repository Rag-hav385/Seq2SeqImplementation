#!/usr/bin/env python
# coding: utf-8

# # <font color='red'>**Sequence to sequence implementation**</font>

# <h1>Load the data</h1>

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('wget https://www.dropbox.com/s/ddkmtqz01jc024u/glove.6B.100d.txt')


# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# import seaborn as sns
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense , Input , Softmax , RNN , LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# In[ ]:


#Practice

###########################################################################

#Practice 1 : Making a Custom layer
class myDenseLayer(tf.keras.layers.Layer):
    """
    This class will create custom layers
    """
    #Overriding __init__
    def __init__(self , output_shape):
        """
        overridding __init__ function of tf.keras.layers
        initialization of myDenseLayer
        argument : output_shape
        """
        super().__init__() #so that instance have all the other functionalities of tf.keras.layers
        self.num_output = output_shape
    
    #Overriding build
    def build(self , input_shape):
        """
        overriding build function of tf.keras.layers
        Keras will automatically build a layer with specified input shape
        argument : input_shape
        """
        print("Input-shape : " ,input_shape)
        print("="*50)
        #add_weight : argument : name , shape for the initialized layer
        self.kernel = self.add_weight("kernel" , shape = [int(input_shape[-1]) , self.num_output])
    
    #Overriding call
    def call(self , input):
        """
        overriding call function of tf.keras.layers
        Execution of the layers
        argument : input
        """
        print("="*50)
        print("Input-Shape",input.shape , "Kernel-Shape",self.kernel.shape)
        print("="*50)
        #simple Matrix multiplication
        return tf.matmul(input , self.kernel)


print("Practice - 1")
print("="*50)
print()
print("="*50)
data = tf.zeros([10,5])
print("="*50)
print("Data : ",data)
print("="*50)

x = Input(shape = (5,)) #<KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'input_3')>
output = myDenseLayer(10)(x)

#creating model
model = Model(inputs = x , outputs = output)

model.summary()

###########################################################################

###########################################################################

#Practice 2 : How Layers are recursively composible

class MLPBlock(tf.keras.layers.Layer):
    """
    This function will directly create a MLP Block from myDenseLayers recursively
    """
    def __init__(self):
        """
        Overriding  __init__ function and recursively creating three instances of myDenseLayer
        """
        super(MLPBlock , self).__init__()
        self.layer_1 = myDenseLayer(32)
        self.layer_2 = myDenseLayer(32)
        self.layer_3 = myDenseLayer(10)
    
    def call(self , input):
        """
        Will create a MLP Block and will give out the MLP Logic
        """
        print("Calculating.....")
        x = self.layer_1(input)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.relu(x)
        return self.layer_3(x)

print()
print("="*50)
print("Practice - 2")
print("="*50)

mlp = MLPBlock()
print("Data :",tf.ones(shape=(3,64)))
y = mlp(tf.ones(shape=(3,64)))

print("Output")
print(y)
print("="*50)
print("Weights" , len(mlp.weights))
print("Trainable Weights" , len(mlp.trainable_weights))
print("="*50)
###########################################################################

###########################################################################

#Practice 3 : Building a Custom model

class MyModel(Model):
    def __init__(self, num_inputs, num_outputs, rnn_units):
        super().__init__() # https://stackoverflow.com/a/27134600/4084039
        self.dense = myDenseLayer(num_outputs) 
        # we can't use the LSTM layer directly when we are building the custom model
        # we need to write like to get the functionality of the LSTM layer
        self.lstmcell = tf.keras.layers.LSTMCell(rnn_units)
        self.rnn = RNN(self.lstmcell)
        self.softmax = Softmax()
        
    def call(self, input):
        output = self.rnn(input)
        print("LSTM-Output" , output)
        output = self.dense(output)
        print("Dense-Output" , output)
        output = self.softmax(output)
        print("Softmax-output" , output)
        return output

import numpy as np

print()
print("="*50)
print("Practice - 3")
print("="*50)
data = np.zeros([10,5,5])
y = np.zeros([10,2])
print("="*50)
print("data :",data.shape)
print("y    :" , y.shape)
print("="*50)
model  = MyModel(num_inputs=5, num_outputs=2, rnn_units=32)

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,loss=loss_object)
model.fit(data,y, steps_per_epoch=1)

model.summary()

###########################################################################

###########################################################################

#Practice 4 : Building a Custom Encoder Decoder

#Check the reference all the code is given 
print()
print("="*50)
print("Practice - 4")
print("="*50)
print("="*50)
print("Custom Encoder-Decoder check the reference")
print("="*50)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, input_length, enc_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.enc_units= enc_units
        self.lstm_output = 0
        self.lstm_state_h=0
        self.lstm_state_c=0
        
    def build(self, input_shape):
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_encoder")
        self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True, name="Encoder_LSTM")
        
    def call(self, input_sentances, training=True):
        print("ENCODER ==> INPUT SQUENCES SHAPE :",input_sentances.shape)
        input_embedd                           = self.embedding(input_sentances)
        print("ENCODER ==> AFTER EMBEDDING THE INPUT SHAPE :",input_embedd.shape)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    def get_states(self):
        return self.lstm_state_h,self.lstm_state_c
    
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, input_length, dec_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.input_length = input_length

    def build(self, input_shape):
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_decoder")
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Encoder_LSTM")
        
    def call(self, target_sentances, state_h, state_c):
        print("DECODER ==> INPUT SQUENCES SHAPE :",target_sentances.shape)
        target_embedd           = self.embedding(target_sentances)
        print("WE ARE INITIALIZING DECODER WITH ENCODER STATES :",state_h.shape, state_c.shape)
        lstm_output, _,_        = self.lstm(target_embedd, initial_state=[state_h, state_c])
        return lstm_output
    

class MyModel(Model):
    def __init__(self, encoder_inputs_length,decoder_inputs_length, output_vocab_size):
        super().__init__() # https://stackoverflow.com/a/27134600/4084039
        self.encoder = Encoder(vocab_size=500, embedding_dim=50, input_length=encoder_inputs_length, enc_units=64)
        self.decoder = Decoder(vocab_size=500, embedding_dim=50, input_length=decoder_inputs_length, dec_units=64)
        self.dense   = Dense(output_vocab_size, activation='softmax')
        
        
    def call(self, data):
        input,output = data[0], data[1]
        print("="*20, "ENCODER", "="*20)
        encoder_output, encoder_h, encoder_c = self.encoder(input)
        print("-"*27)
        print("ENCODER ==> OUTPUT SHAPE",encoder_output.shape)
        print("ENCODER ==> HIDDEN STATE SHAPE",encoder_h.shape)
        print("ENCODER ==> CELL STATE SHAPE", encoder_c.shape)
        print("="*20, "DECODER", "="*20)
        decoder_output                       = self.decoder(output, encoder_h, encoder_c)
        output                               = self.dense(decoder_output)
        print("-"*27)
        print("FINAL OUTPUT SHAPE",output.shape)
        print("="*50)
        return output

model  = MyModel(encoder_inputs_length=30,decoder_inputs_length=20,output_vocab_size=500)

ENCODER_SEQ_LEN = 30
DECODER_SEQ_LEN = 20

input = np.random.randint(0, 499, size=(2000, ENCODER_SEQ_LEN))
output = np.random.randint(0, 499, size=(2000, DECODER_SEQ_LEN))
target = tf.keras.utils.to_categorical(output, 500)


# loss_object = loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')

model.fit([input, output], output, steps_per_epoch=1)

"""
or you can try this

model.compile(optimizer=optimizer,loss='categorical_crossentropy')
model.fit([input, output], target, steps_per_epoch=1)

"""
model.summary()
print()
print("="*50)

###########################################################################

###########################################################################


# In[ ]:


np.expand_dims(input[0], 0).shape


# In[ ]:


print("=" * 30, "Inference", "=" * 30)
enc_output, enc_state_h, enc_state_c = model.layers[0](np.expand_dims(input[0], 0))
states_values = [enc_state_h, enc_state_c]
pred = []
cur_vec = np.zeros((1, 1))
print('-'*20,"started predition","-"*20)
print("at time step 0 the word is 0")
for i in range(DECODER_SEQ_LEN):
    cur_emb = model.layers[1].embedding(cur_vec)
    [infe_output, state_h, state_c] = model.layers[1].lstm(cur_emb, initial_state=states_values)
    infe_output=model.layers[2](infe_output)
    states_values = [state_h, state_c]
    # np.argmax(infe_output) will be a single value, which represents the the index of predicted word
    # but to pass this data into next time step embedding layer, we are reshaping it into (1,1) shape
    print("prediction :",np.argmax(infe_output))
    cur_vec = np.reshape(np.argmax(infe_output), (1, 1))
    print("at time step 0 the word is ", cur_vec)
    pred.append(cur_vec)


# <h1> Preprocess Data </h1>

# In[ ]:


with open('/content/drive/MyDrive/ita.txt', 'r', encoding="utf8") as f:
    eng=[]
    ita=[]
    for i in f.readlines():
        eng.append(i.split("\t")[0])
        ita.append(i.split("\t")[1])
data = pd.DataFrame(data=list(zip(eng, ita)), columns=['english','italian'])
print(data.shape)
data.head()


# In[ ]:


def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase

def preprocess(text):
    # convert all the text into lower letters
    # use this function to remove the contractions: https://gist.github.com/anandborad/d410a49a493b56dace4f814ab5325bbd
    # remove all the spacial characters: except space ' '
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    return text

def preprocess_ita(text):
    # convert all the text into lower letters
    # remove the words betweent brakets ()
    # remove these characters: {'$', ')', '?', '"', '’', '.',  '°', '!', ';', '/', "'", '€', '%', ':', ',', '('}
    # replace these spl characters with space: '\u200b', '\xa0', '-', '/'
    # we have found these characters after observing the data points, feel free to explore more and see if you can do find more
    # you are free to do more proprocessing
    # note that the model will learn better with better preprocessed data 
    
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[$)\?"’.°!;\'€%:,(/]', '', text)
    text = re.sub('\u200b', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('-', ' ', text)
    return text


data['english'] = data['english'].apply(preprocess)
data['italian'] = data['italian'].apply(preprocess_ita)
data.head()


# In[ ]:


ita_lengths = data['italian'].str.split().apply(len)
eng_lengths = data['english'].str.split().apply(len)


# In[ ]:


print("-"*25)
print("FOR ITALIAN")
print("-"*25)
print("-"*25)
print("0-100 with 10 steps..")
print("-"*25)
for i in range(0,101,10):
    print(i,"%",np.percentile(ita_lengths, i))
print("-"*25)
print("90-100 with 1 step")
print("-"*25)
for i in range(90,101):
    print(i,"%",np.percentile(ita_lengths, i))
print("-"*25)
print("for : [99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100]")
print("-"*25)
for i in [99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100]:
    print(i,"%",np.percentile(ita_lengths, i))


# In[ ]:


print("-"*25)
print("FOR English")
print("-"*25)
print("-"*25)
print("0-100 with 10 steps..")
print("-"*25)
for i in range(0,101,10):
    print(i,"%",np.percentile(eng_lengths, i))
print("-"*25)
print("90-100 with 1 step")
print("-"*25)
for i in range(90,101):
    print(i,"%",np.percentile(eng_lengths, i))
print("-"*25)
print("for : [99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100]")
print("-"*25)
for i in [99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100]:
    print(i,"%",np.percentile(eng_lengths, i))


# In[ ]:


data['italian_len'] = data['italian'].str.split().apply(len)
data = data[data['italian_len'] < 20]

data['english_len'] = data['english'].str.split().apply(len)
data = data[data['english_len'] < 20]

data['english_inp'] = '<start> ' + data['english'].astype(str)
data['english_out'] = data['english'].astype(str) + ' <end>'

data = data.drop(['english','italian_len','english_len'], axis=1)
# only for the first sentance add a toke <end> so that we will have <end> in tokenizer
data.head()


# In[ ]:


data.sample(10)


# <h1> Getting Train and test data </h1>

# In[ ]:


from sklearn.model_selection import train_test_split
train, validation = train_test_split(data, test_size=0.2)


# In[ ]:


print(train.shape, validation.shape)
# for one sentence we will be adding <end> token so that the tokanizer learns the word <end>
# with this we can use only one tokenizer for both encoder output and decoder output
train.iloc[0]['english_inp']= str(train.iloc[0]['english_inp'])+' <end>'
train.iloc[0]['english_out']= str(train.iloc[0]['english_out'])+' <end>'


# In[ ]:


train.head()


# In[ ]:


validation.head()


# In[ ]:


ita_lengths = train['italian'].str.split().apply(len)
eng_lengths = train['english_inp'].str.split().apply(len)
import seaborn as sns
sns.kdeplot(ita_lengths)
plt.show()
sns.kdeplot(eng_lengths)
plt.show()


# In[2]:


import pickle
location = "drive/MyDrive/files_for_task_1/"
train = pickle.load(open(location + "train.pickle" , "rb"))
train.head()


# In[3]:


validation = pickle.load(open(location + "validation.pickle" , "rb"))
validation.head()


# ### Creating Tokenizer on the train data and learning vocabulary

# > Note that we are fitting the tokenizer only on train data and check the filters for english, we need to remove symbols &lt; and &gt;

# In[ ]:


# '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tknizer_ita = Tokenizer()
tknizer_ita.fit_on_texts(train['italian'].values)
tknizer_eng = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n') # removed '<' and '>'
tknizer_eng.fit_on_texts(train['english_inp'].values)


# In[4]:


tknizer_ita = pickle.load(open(location + "tokenizer_italian.pickle" ,"rb"))
tknizer_eng = pickle.load(open(location + "tokenizer_english.pickle" ,"rb"))


# In[5]:


vocab_size_eng=len(tknizer_eng.word_index.keys())#12828
print(vocab_size_eng)
vocab_size_ita=len(tknizer_ita.word_index.keys())#26188
print(vocab_size_ita)


# In[6]:


tknizer_eng.word_index['<start>'], tknizer_eng.word_index['<end>']#(1,10120)


# ### Creating embeddings for english sentences

# In[ ]:


embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size_eng+1, 100))
for word, i in tknizer_eng.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[7]:


embedding_matrix = pickle.load(open(location + "embedding_matrix.pickle" , "rb"))


# In[ ]:


embedding_matrix.shape#(12829, 100)


# ## <font color='blue'>**Implement custom encoder decoder**</font>

# <font color='blue'>**Encoder**</font>

# In[8]:


#Taken from reference

class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):

        #Initialize Embedding layer
        #Intialize Encoder LSTM layer
        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_dim = embedding_size
        self.input_length = input_length
        self.enc_units= lstm_size
        self.lstm_output = 0
        self.lstm_state_h=0
        self.lstm_state_c=0
    
    def build(self, input_shape):
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_encoder")
        self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True, name="Encoder_LSTM")
        
    def call(self, input_sentances, training=True):
        '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- encoder_output, last time step's hidden and cell state
        '''
        input_embedd                           = self.embedding(input_sentances)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    

    def get_states(self):
        return self.lstm_state_h,self.lstm_state_c

      

    
    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      self.lstm_state_h = np.zeros([batch_size , self.enc_units])
      self.lstm_state_c = np.zeros([batch_size , self.enc_units]) 
      


# In[ ]:


#Written own for practice.

class Encoder(tf.keras.Model):
    """
    This is a custom based Encoder
    """
    def __init__(self, vocab_size , embedding_size , lstm_size , input_length):
        """
        Overriding __init_ function of the Model API
        """
        #Initializing Embedding layer and lstm layer.
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        self.lstm_output = 0
        self.lstm_state_h = 0
        self.lstm_state_c = 0

    def build(self , input_shape):
        """
        Initializing embedding and LSTM Layer
        """
        self.embedding = Embedding(input_dim=self.vocab_size , output_dim=self.embedding_size , input_length = self.input_length,
                                   mask_zero = True , name = "Embedding_Layer_encoder")
        self.lstm = LSTM(self.lstm_size , return_state=True, return_sequences=True, name="Encoder_LSTM")

    def call(self , input_sentances , training = True):
        """

        This will implement the logic of the embedding layer
        """

        input_embedd                           = self.embedding(input_sentances)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    
    def get_states(self):
        """
        return lstm_hidden state  , lstm_cell_sate
        """
        return self.lstm_state_h , self.lstm_state_c

    def initialize_states(self,batch_size):
        '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
        '''
        self.lstm_state_h = np.zeros([batch_size , self.lstm_size])
        self.lstm_state_c = np.zeros([batch_size , self.lstm_size]) 



# <font color='orange'>**Grader function - 1**</font>

# In[ ]:


def grader_check_encoder():
    '''
        vocab-size: Unique words of the input language,
        embedding_size: output embedding dimension for each word after embedding layer,
        lstm_size: Number of lstm units,
        input_length: Length of the input sentence,
        batch_size
    '''
    vocab_size=10
    embedding_size=20
    lstm_size=32
    input_length=10
    batch_size=16
    #Intialzing encoder 
    encoder=Encoder(vocab_size,embedding_size,lstm_size,input_length)
    input_sequence=tf.random.uniform(shape=[batch_size,input_length],maxval=vocab_size,minval=0,dtype=tf.int32)
    #Intializing encoder initial states
    initial_state=encoder.initialize_states(batch_size)
    
    encoder_output,state_h,state_c=encoder(input_sequence,initial_state)
    print(encoder_output.shape , (batch_size,input_length,lstm_size))
    print(state_h.shape , (batch_size,lstm_size))
    print( state_c.shape , (batch_size,lstm_size))
    assert(encoder_output.shape==(batch_size,input_length,lstm_size) and state_h.shape==(batch_size,lstm_size) and state_c.shape==(batch_size,lstm_size))
    return True
print(grader_check_encoder())


# In[ ]:


class Decoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''
    def __init__(self,out_vocab_size,embedding_size,lstm_size,input_length):
        """
        Overrriding __init__ function of the keras.model API
        
        """
        super().__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length

        #as the input shape is defined, so we ll make a embedding layer using embedding matrix,with trainable = False

        self.embedding = Embedding(input_dim=self.out_vocab_size , output_dim=self.embedding_size , input_length=self.input_length , 
                                   weights = [embedding_matrix] , mask_zero = True , name = "Embedding_layer_decoder" , trainable = False)
        self.lstm = LSTM(units = self.lstm_size , return_sequences=True , return_state=True , name = "LSTM_Decoder")

    def call(self , input_sequences , state_h , state_c):

        embedded_input = self.embedding(input_sequences)
        lstm_output ,lstm_state_h , lstm_state_c = self.lstm(embedded_input , initial_state = [state_h , state_c])
        return lstm_output ,lstm_state_h , lstm_state_c






# <font color='orange'>**Grader function - 2**</font>

# In[ ]:


def grader_decoder():
    '''
        out_vocab_size: Unique words of the target language,
        embedding_size: output embedding dimension for each word after embedding layer,
        dec_units: Number of lstm units in decoder,
        input_length: Length of the input sentence,
        batch_size
        
    
    '''
    out_vocab_size=13 
    embedding_dim=12 
    input_length=10
    dec_units=16 
    batch_size=32
    
    target_sentences=tf.random.uniform(shape=(batch_size,input_length),maxval=10,minval=0,dtype=tf.int32)
    encoder_output=tf.random.uniform(shape=[batch_size,input_length,dec_units])
    state_h=tf.random.uniform(shape=[batch_size,dec_units])
    state_c=tf.random.uniform(shape=[batch_size,dec_units])
    states=[state_h,state_c]
    decoder=Decoder(out_vocab_size, embedding_dim, dec_units,input_length )
    output,_,_=decoder(target_sentences, states[0] , states[1])
    assert(output.shape==(batch_size,input_length,dec_units))
    return True
print(grader_decoder())


# In[10]:


class Dataset:
    def __init__(self, data, tknizer_ita, tknizer_eng, max_len):
        self.encoder_inps = data['italian'].values
        self.decoder_inps = data['english_inp'].values
        self.decoder_outs = data['english_out'].values
        self.tknizer_eng = tknizer_eng
        self.tknizer_ita = tknizer_ita
        self.max_len = max_len

    def __getitem__(self, i):
        self.encoder_seq = self.tknizer_ita.texts_to_sequences([self.encoder_inps[i]]) # need to pass list of values
        self.decoder_inp_seq = self.tknizer_eng.texts_to_sequences([self.decoder_inps[i]])
        self.decoder_out_seq = self.tknizer_eng.texts_to_sequences([self.decoder_outs[i]])

        self.encoder_seq = pad_sequences(self.encoder_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, maxlen=self.max_len, dtype='int32', padding='post')
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq

    def __len__(self): # your model.fit_gen requires this function
        return len(self.encoder_inps)

    
class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))


    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)]
        # we are creating data like ([italian, english_inp], english_out) these are already converted into seq
        return tuple([[batch[0],batch[1]],batch[2]])

    def __len__(self):  # your model.fit_gen requires this function
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)


# In[11]:


train_dataset = Dataset(train, tknizer_ita, tknizer_eng, 20)
test_dataset  = Dataset(validation, tknizer_ita, tknizer_eng, 20)

train_dataloader = Dataloder(train_dataset, batch_size=1024)
test_dataloader = Dataloder(test_dataset, batch_size=1024)


print(train_dataloader[0][0][0].shape, train_dataloader[0][0][1].shape, train_dataloader[0][1].shape)


# In[ ]:


vocab_size_ita+1


# In[12]:


#Taken with reference to the encoder decoder taken from reference, didnt write this again,its easy to interpret now.
class Encoder_decoder(tf.keras.Model):
    
    def __init__(self, encoder_inputs_length,decoder_inputs_length, output_vocab_size):
        super().__init__() # https://stackoverflow.com/a/27134600/4084039
        self.encoder = Encoder(inp_vocab_size=vocab_size_ita+1, embedding_size=50, input_length=encoder_inputs_length, lstm_size=256)
        self.decoder = Decoder(out_vocab_size=vocab_size_eng+1, embedding_size=100, input_length=decoder_inputs_length,lstm_size=256)
        self.dense   = Dense(output_vocab_size, activation='softmax')
        
        
    def call(self, data):
        input,output = data[0], data[1]
        encoder_output, encoder_h, encoder_c = self.encoder(input)
        decoder_output ,_,_                  = self.decoder(output, encoder_h, encoder_c)
        output                               = self.dense(decoder_output)
        return output
        
        


# In[ ]:


#Create an object of encoder_decoder Model class, 
# Compile the model and fit the model
#encoder_inputs_length,decoder_inputs_length, output_vocab_size
model  = Encoder_decoder(encoder_inputs_length=20,decoder_inputs_length=20,output_vocab_size=vocab_size_eng)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')
train_steps=train.shape[0]//1024
valid_steps=validation.shape[0]//1024
model.fit_generator(train_dataloader, steps_per_epoch=train_steps, epochs=50, validation_data=test_dataloader, validation_steps=valid_steps)
model.summary()


# In[ ]:


#Saving Important variables for task 1
model.save_weights("my_model_weights.h5")
import pickle
with open("tokenizer_english.pickle" , "wb") as f:
  pickle.dump(tknizer_eng , f)
with open("tokenizer_italian.pickle" , "wb") as f:
  pickle.dump(tknizer_ita , f)
with open("embedding_matrix.pickle" , "wb") as f:
  pickle.dump(embedding_matrix , f)
with open("train.pickle" , "wb") as f:
  pickle.dump(train , f)

with open("validation.pickle" , "wb") as f:
  pickle.dump(validation , f)


# In[13]:


#Loading weights of the model

model  = Encoder_decoder(encoder_inputs_length=20,decoder_inputs_length=20,output_vocab_size=vocab_size_eng)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')
train_steps=train.shape[0]//1024
valid_steps=validation.shape[0]//1024
model.fit_generator(train_dataloader, steps_per_epoch=train_steps, epochs=1, validation_data=test_dataloader, validation_steps=valid_steps)

print("Loading Weights..")
model.load_weights(location + "my_model_weights.h5")
print("Done..")


# In[14]:


input_sentence = test_dataloader[0][0][0][1000]
input_sentence


# In[ ]:


english_dict = dict([(value, key) for key, value in tknizer_eng.word_index.items()]) 
english_dict


# In[16]:


def predict(input_sentence):

  '''
  A. Given input sentence, convert the sentence into integers using tokenizer used earlier
  B. Pass the input_sequence to encoder. we get encoder_outputs, last time step hidden and cell state
  C. Initialize index of <start> as input to decoder. and encoder final states as input_states to decoder
  D. till we reach max_length of decoder or till the model predicted word <end>:
         predicted_out,state_h,state_c=model.layers[1](dec_input,states)
         pass the predicted_out to the dense layer
         update the states=[state_h,state_c]
         And get the index of the word with maximum probability of the dense layer output, using the tokenizer(word index) get the word and then store it in a string.
         Update the input_to_decoder with current predictions
  F. Return the predicted sentence
  '''
  DECODER_SEQ_LEN = 20
  enc_output, enc_state_h, enc_state_c = model.layers[0](np.expand_dims(input_sentence , 0))
  states_values = [enc_state_h, enc_state_c]
  pred = []
  cur_vec = np.zeros((1, 1))

  for i in range(DECODER_SEQ_LEN):
      cur_emb = model.layers[1].embedding(cur_vec)
      [infe_output, state_h, state_c] = model.layers[1].lstm(cur_emb, initial_state=states_values)
      infe_output=model.layers[2](infe_output)
      states_values = [state_h, state_c]
    # np.argmax(infe_output) will be a single value, which represents the the index of predicted word
    # but to pass this data into next time step embedding layer, we are reshaping it into (1,1) shape

      cur_vec = np.reshape(np.argmax(infe_output), (1, 1))
      pred.append(np.argmax(infe_output))
  
  #Sentence formation
  result = []
  for num in pred:
    if english_dict[num] == "<end>":
      continue
    else:
      result.append(english_dict[num])
  
  result = " ".join(result)
  return result + " <end>"



# In[ ]:


italian_dict = dict([(value, key) for key, value in tknizer_ita.word_index.items()]) 
def get_italian(seq):
  result = ""
  for num in seq:
    if num == 0:
      continue
    result += italian_dict[num] + " "
  return result

get_italian(test_dataloader[0][0][0][0])


# In[17]:


predict(test_dataloader[0][0][0][0])


# In[18]:


y = pd.DataFrame(validation["english_out"])
y = y.reset_index()
y = y.drop("index" , axis = 1)
y = y[:1000]
y


# In[19]:


#Calculating bleu score for the same
import nltk.translate.bleu_score as bleu
from tqdm import tqdm
bleu_score = []
for i in tqdm(range(0,1000)):
  prediction = predict(test_dataloader[0][0][0][i])
  real = y["english_out"][i]
  prediction_list = prediction.split()
  real_list = real.split()
  bleu_score.append(bleu.sentence_bleu(real_list , prediction_list))


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,8))
plt.title("Bleu Score for 1000 italian-to-english sentences".upper())
sns.kdeplot(bleu_score)
plt.xlabel("BLEU-Scores")
plt.grid(True)
plt.show()

print("="*50)
print("Density of score < 0.6 is more i.e ")
print("simple encoder-decoder is not a good model for this")
print("="*50)


# ## Task -2: Including Attention mechanisum

# ### <font color='blue'>**Implement custom encoder decoder and attention layers**</font>

# <font color='blue'>**Encoder**</font>

# In[11]:


class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):

        #Initialize Embedding layer
        #Intialize Encoder LSTM layer
        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        self.ltsm_outputs = 0
        self.lstm_hidden_h = 0
        self.lstm_hidden_c = 0

        #Layers
        self.embedding = Embedding(input_dim=self.vocab_size , output_dim=self.embedding_size , input_length=self.input_length,
                                   mask_zero = True , name = "encoder_embedding_layer")
        self.lstm = LSTM(units=self.lstm_size , return_sequences=True , return_state=True , name = "LSTM_ENCODER")



    def call(self,input_sequence,states):
      '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- All encoder_outputs, last time steps hidden and cell state
      '''
      #print("Input-sentence:",input_sequence.shape)
      embedding_input = self.embedding(input_sequence)
      #print("embedding_input_encoder:",embedding_input.shape)
      self.ltsm_outputs , self.lstm_hidden_h , self.lstm_hidden_c = self.lstm(embedding_input)
      return self.ltsm_outputs , self.lstm_hidden_h , self.lstm_hidden_c

      # Input-sentence: (16, 10)
      # Embedding_input_encoder: (16, 10, 20)
    
    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      self.lstm_hidden_h = tf.zeros((batch_size , self.lstm_size))
      self.lstm_hidden_c = tf.zeros((batch_size , self.lstm_size))

      return self.lstm_hidden_h , self.lstm_hidden_c
      


# <font color='cyan'>**Grader function - 1**</font>

# In[12]:


def grader_check_encoder():
    
    '''
        vocab-size: Unique words of the input language,
        embedding_size: output embedding dimension for each word after embedding layer,
        lstm_size: Number of lstm units in encoder,
        input_length: Length of the input sentence,
        batch_size
    '''
    
    vocab_size=10
    embedding_size=20
    lstm_size=32
    input_length=10
    batch_size=16
    encoder=Encoder(vocab_size,embedding_size,lstm_size,input_length)
    input_sequence=tf.random.uniform(shape=[batch_size,input_length],maxval=vocab_size,minval=0,dtype=tf.int32)
    initial_state=encoder.initialize_states(batch_size)
    encoder_output,state_h,state_c=encoder(input_sequence,initial_state)
    
    assert(encoder_output.shape==(batch_size,input_length,lstm_size) and state_h.shape==(batch_size,lstm_size) and state_c.shape==(batch_size,lstm_size))
    return True
print(grader_check_encoder())


# <font color='blue'>**Attention**</font>

# In[13]:


class Attention(tf.keras.layers.Layer):
  '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
  '''
  def __init__(self,scoring_function, att_units):


    # Please go through the reference notebook and research paper to complete the scoring functions
    super().__init__()
    self.contex_vector = 0
    self.attention_weights = 0
    self.scoring_function = scoring_function
    self.att_units = att_units

    if self.scoring_function=='dot':
      # Intialize variables needed for Dot score function here
      #using tf dot layer
      self.dot_layer = tf.keras.layers.Dot(axes = (2,2))
      
      
    if self.scoring_function == 'general':
      # Intialize variables needed for General score function here
      self.dot_layer = tf.keras.layers.Dot(axes = (2,2))
      self.connector = tf.keras.layers.Dense(self.att_units)

      
    elif self.scoring_function == 'concat':
      # Intialize variables needed for Concat score function here
      self.w_1 = tf.keras.layers.Dense(self.att_units)
      self.w_2 = tf.keras.layers.Dense(self.att_units)
      self.v_1 = tf.keras.layers.Dense(1)
      pass
  
  
  def call(self,decoder_hidden_state,encoder_output):
    '''
      Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs.
      * Based on the scoring function we will find the score or similarity between decoder_hidden_state and encoder_output.
        Multiply the score function with your encoder_outputs to get the context vector.
        Function returns context vector and attention weights(softmax - scores)
    '''
    decoder_time_stamp = tf.expand_dims(decoder_hidden_state , 1)
    #Decoder hidden state : (16, 32)
    #Decoder time stamp   : (16, 1, 32)
    #encoder_output       : (16, 10, 32)
    
    if self.scoring_function == 'dot':
        # Implement Dot score function here

        #Shape : (16, 10, 1)
        dotted_first_encoder = tf.matmul(encoder_output,tf.expand_dims(decoder_hidden_state , axis = -1))

        #Going through softmax
        self.attention_weights = tf.nn.softmax(dotted_first_encoder , axis = 1)
        
        #getting context vector
        self.context_vector = self.attention_weights * encoder_output #shape : (16,10,32)
        self.context_vector = tf.reduce_sum(self.context_vector, axis=1) #shape : (16,32)
        
        return self.context_vector , self.attention_weights
    
    elif self.scoring_function == 'general':
        # Implement General score function here
        #print("I m in attention---------------------------------------------------".upper())
        #Shape : (16, 10, 1)
        connect_dotted_encoder = tf.matmul(self.connector(encoder_output),tf.expand_dims(decoder_hidden_state , axis = -1))
        
        #Going through Softmax
        self.attention_weights = tf.nn.softmax(connect_dotted_encoder , axis = 1)

        #getting context vector
        self.context_vector = self.attention_weights * encoder_output
        self.context_vector = tf.reduce_sum(self.context_vector , axis = 1)

        return self.context_vector , self.attention_weights

    elif self.scoring_function == 'concat':
        # Implement General score function here

        #print(self.w_1(decoder_time_stamp).shape , self.w_2(encoder_output).shape )
        summation = self.w_1(decoder_time_stamp) + self.w_2(encoder_output)
        #print("summation:",summation.shape)
        #Shape : (16, 10, 1)
        relative_score = self.v_1(tf.nn.tanh(summation))
        #print("relative-score:",relative_score.shape)
        
        #Going through softmax
        self.attention_weights = tf.nn.softmax(relative_score , axis = 1)

        #Getting Context vector
        self.context_vector = self.attention_weights * encoder_output
        self.context_vector = tf.reduce_sum(self.context_vector , axis = 1)

        #print("Context-Vector " ,self.context_vector.shape)
        return self.context_vector , self.attention_weights



        
    
    


# <font color='cyan'>**Grader function - 2**</font>

# In[14]:


def grader_check_attention(scoring_fun):
    
    ''' 
        att_units: Used in matrix multiplications for scoring functions,
        input_length: Length of the input sentence,
        batch_size
    '''
    
    input_length=10
    batch_size=16
    att_units=32
    
    state_h=tf.random.uniform(shape=[batch_size,att_units])
    encoder_output=tf.random.uniform(shape=[batch_size,input_length,att_units])
    attention=Attention(scoring_fun,att_units)
    context_vector,attention_weights=attention(state_h,encoder_output)
    assert(context_vector.shape==(batch_size,att_units) and attention_weights.shape==(batch_size,input_length,1))
    return True
print(grader_check_attention('dot'))
print(grader_check_attention('general'))
print(grader_check_attention('concat'))


# <font color='blue'>**OneStepDecoder**</font>

# In[15]:


class One_Step_Decoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):

      # Initialize decoder embedding layer, LSTM and any other objects needed
      super().__init__()
      #self.vocab_size = tar_vocab_size
      #self.embedding_dim = embedding_dim
      #self.input_length = input_length
      #self.dec_units = dec_units
      #self.score_fun = score_fun
      #self.att_units = att_units
      self.attention_weights = 0
      self.lstm_state_h = 0
      self.lstm_state_c = 0
      self.context_vector = 0

      #Layers
      self.embedding = Embedding(input_dim=tar_vocab_size , output_dim=embedding_dim , input_length=input_length,
                                 mask_zero = True , name = "Embedding_OneStepDecoder")
      self.lstm      = LSTM(dec_units , return_sequences=True , return_state=True , name = "Lstm_decoder")
      self.layer_dense = tf.keras.layers.Dense(tar_vocab_size , name = "Dense_layer_onestepDecoder")

      self.attention_instance = Attention(score_fun , att_units)

  def call(self,input_to_decoder, encoder_output, state_h,state_c):
    '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B
    '''
    #Shape : (32 , 1 , 12)
    #Getting embedding output
    embedding_output = self.embedding(input_to_decoder)
    #print("Embedding_output:",embedding_output.shape)

    #Getting Context_vector and attention_weights
    #attention_instance = Attention(self.score_fun , self.att_units)
    self.context_vector , self.attention_weights = self.attention_instance(state_h , encoder_output)

    #print(tf.expand_dims(self.context_vector , 1).shape , embedding_output.shape)
    #Concatenating embedding_output and context_vector
    lstm_input = tf.concat([tf.expand_dims(self.context_vector , 1) , embedding_output] ,2)
    
    #getting decoder_output and states
    decoder_outputs , self.lstm_state_h , self.lstm_state_c = self.lstm(lstm_input , initial_state = [state_h , state_c])

    #Getting Dense Layer Output
    Dense_output = self.layer_dense(decoder_outputs)
    Dense_output = tf.reshape(Dense_output , (-1 , Dense_output.shape[2]))
    
    #returning required variables
    return Dense_output ,self.lstm_state_h , self.lstm_state_c , self.attention_weights , self.context_vector



    




# <font color='cyan'>**Grader function - 3**</font>

# In[16]:


def grader_onestepdecoder(score_fun):
    
    '''
        tar_vocab_size: Unique words of the target language,
        embedding_dim: output embedding dimension for each word after embedding layer,
        dec_units: Number of lstm units in decoder,
        att_units: Used in matrix multiplications for scoring functions in attention class,
        input_length: Length of the target sentence,
        batch_size
        
    
    '''
    
    tar_vocab_size=13 
    embedding_dim=12 
    input_length=10
    dec_units=16 
    att_units=16
    batch_size=32
    onestepdecoder=One_Step_Decoder(tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units)
    input_to_decoder=tf.random.uniform(shape=(batch_size,1),maxval=10,minval=0,dtype=tf.int32)
    encoder_output=tf.random.uniform(shape=[batch_size,input_length,dec_units])
    state_h=tf.random.uniform(shape=[batch_size,dec_units])
    state_c=tf.random.uniform(shape=[batch_size,dec_units])
    output,state_h,state_c,attention_weights,context_vector=onestepdecoder(input_to_decoder,encoder_output,state_h,state_c)

    assert(output.shape==(batch_size,tar_vocab_size))
    assert(state_h.shape==(batch_size,dec_units))
    assert(state_c.shape==(batch_size,dec_units))
    assert(attention_weights.shape==(batch_size,input_length,1))
    assert(context_vector.shape==(batch_size,dec_units))
    return True
    
print(grader_onestepdecoder('dot'))
print(grader_onestepdecoder('general'))
print(grader_onestepdecoder('concat'))
    


# <font color='blue'>**Decoder**</font>

# In[17]:


class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
      super(Decoder , self).__init__()
      self.vocab_size = out_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.score_fun = score_fun
      self.att_units = att_units

      self.onestep_decoder = One_Step_Decoder(self.vocab_size ,self.embedding_dim , 
                                             self.input_length , self.dec_units , 
                                             self.score_fun , self.att_units)


        
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state):

        #Initialize an empty Tensor array, that will store the outputs at each and every time step
        #Create a tensor array as shown in the reference notebook
        
        #Iterate till the length of the decoder input
            # Call onestepdecoder for each token in decoder_input
            # Store the output in tensorarray
        # Return the tensor array
        all_outputs = tf.TensorArray(tf.float32 , size=tf.shape(input_to_decoder)[1] , name = "output_arrays_decoder")
        #print(input_to_decoder)
        for timestep in range(tf.shape(input_to_decoder)[1]):
          #print(input_to_decoder[:,timestep:timestep+1])
          output,decoder_hidden_state,decoder_cell_state,attention_weights,context_vector = self.onestep_decoder(input_to_decoder[:,timestep:timestep+1],encoder_output,
                                                                                         decoder_hidden_state,decoder_cell_state) 
          all_outputs = all_outputs.write(timestep , output)
        all_outputs = tf.transpose(all_outputs.stack() , [1,0,2])

        return all_outputs
    


# <font color='cyan'>**Grader function - 4**</font>

# In[18]:


def grader_decoder(score_fun):
    
    '''
        out_vocab_size: Unique words of the target language,
        embedding_dim: output embedding dimension for each word after embedding layer,
        dec_units: Number of lstm units in decoder,
        att_units: Used in matrix multiplications for scoring functions in attention class,
        input_length: Length of the target sentence,
        batch_size
        
    
    '''
    
    out_vocab_size=13 
    embedding_dim=12 
    input_length=11
    dec_units=16 
    att_units=16
    batch_size=32
    
    target_sentences=tf.random.uniform(shape=(batch_size,input_length),maxval=10,minval=0,dtype=tf.int32)
    encoder_output=tf.random.uniform(shape=[batch_size,input_length,dec_units])
    state_h=tf.random.uniform(shape=[batch_size,dec_units])
    state_c=tf.random.uniform(shape=[batch_size,dec_units])
    
    decoder=Decoder(out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units)
    output=decoder(target_sentences,encoder_output, state_h, state_c)
    assert(output.shape==(batch_size,input_length,out_vocab_size))
    return True
print(grader_decoder('dot'))
print(grader_decoder('general'))
print(grader_decoder('concat'))


# <font color='blue'>**Encoder Decoder model**</font>

# In[19]:


class encoder_decoder(tf.keras.Model):
  def __init__(self,embedding_size , lstm_state , input_length ,dec_units, score_fun , att_units , batch_size):
    #Intialize objects from encoder decoder
    super().__init__()
    self.encoder = Encoder(vocab_size_ita + 1 , embedding_size , lstm_state , input_length)
    self.decoder = Decoder(vocab_size_eng + 1 , embedding_size , input_length , dec_units , score_fun , att_units)
    self.batch_size = batch_size
  
  def call(self,data):
    #Intialize encoder states, Pass the encoder_sequence to the embedding layer
    # Decoder initial states are encoder final states, Initialize it accordingly
    # Pass the decoder sequence,encoder_output,decoder states to Decoder
    # return the decoder output

    input_ , target = data[0] , data[1]
    initial_states = self.encoder.initialize_states(self.batch_size)
    #print(initial_states)
    encoder_output , state_h , state_c = self.encoder(input_ , initial_states)
    
    output = self.decoder(target , encoder_output , state_h , state_c)
    return output




# <font color='blue'>**Custom loss function**</font>

# In[20]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def custom_lossfunction(real,pred):
  # Custom loss function that will not consider the loss for padded zeros.
  # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention#define_the_optimizer_and_the_loss_function
  
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


# In[23]:


#Tensorboard callback
import datetime
log_dir="logs_general\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True)



# <font color='blue'>**Training**</font>

# Implement dot function here.

# In[54]:


# Implement teacher forcing while training your model. You can do it two ways.
# Prepare your data, encoder_input,decoder_input and decoder_output
# if decoder input is 
# <start> Hi how are you
# decoder output should be
# Hi How are you <end>
# i.e when you have send <start>-- decoder predicted Hi, 'Hi' decoder predicted 'How' .. e.t.c

# or
 
# model.fit([train_ita,train_eng],train_eng[:,1:]..)
# Note: If you follow this approach some grader functions might return false and this is fine.

#embedding_size , lstm_state , input_length ,dec_units, score_fun , att_units , batch_size
model_dot  = encoder_decoder(embedding_size = 100,lstm_state = 268,input_length = 20,dec_units = 268,score_fun="dot",att_units = 268,batch_size = 268)
optimizer = tf.keras.optimizers.Adam()
model_dot.compile(optimizer=optimizer,loss= custom_lossfunction)
train_steps=train.shape[0]//1024
valid_steps=validation.shape[0]//1024
model_dot.fit_generator(train_dataloader, steps_per_epoch=train_steps, epochs=20, validation_data=test_dataloader, validation_steps=valid_steps , callbacks=tensorboard_callback)


# In[22]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[57]:


get_ipython().run_line_magic('tensorboard', '--logdir  .')


# ## <font color='blue'>**Inference**</font>

# <font color='blue'>**Plot attention weights**</font>

# In[29]:


import seaborn as sns
def plot_attention(orig,pred , attention_weights):
  #Refer: https://www.tensorflow.org/tutorials/text/nmt_with_attention#translate

  x = orig
  y = pred
  

  #attention_weights = np.reshape(attention_weights , (attention_weights.shape[1] , attention_weights.shape[2]))
  plt.figure(figsize=(5,5))


  sns.heatmap(attention_weights[:,:len(pred)] , xticklabels = y , yticklabels =x)

  plt.show()


# In[ ]:


model_dot.layers


# In[ ]:


english_dict


# <font color='blue'>**Predict the sentence translation**</font>

# In[26]:


def predict(input_sentence , model):

  '''
  A. Given input sentence, convert the sentence into integers using tokenizer used earlier
  B. Pass the input_sequence to encoder. we get encoder_outputs, last time step hidden and cell state
  C. Initialize index of <start> as input to decoder. and encoder final states as input_states to decoder
  D. till we reach max_length of decoder or till the model predicted word <end>:
         predicted_out,state_h,state_c=model.layers[1](dec_input,states)
         pass the predicted_out to the dense layer
         update the states=[state_h,state_c]
         And get the index of the word with maximum probability of the dense layer output, using the tokenizer(word index) get the word and then store it in a string.
         Update the input_to_decoder with current predictions
  F. Return the predicted sentence
  '''
 
  pred = []

  attention_plot = np.zeros((20 , 20))

  inp_seq = tknizer_ita.texts_to_sequences([input_sentence])
  
  inp_seq = tf.keras.preprocessing.sequence.pad_sequences(inp_seq , padding='post' , maxlen = 20)

  en_state_h , en_state_c = model.layers[0].initialize_states(1)

  en_outputs , en_state_h , en_stae_c = model.layers[0](tf.constant(inp_seq) , [en_state_h , en_state_c])

  #print(en_outputs.shape)
  #print("States :",en_state_h.shape , en_state_c.shape)
  
  dec_state_h , dec_state_c = en_state_h , en_state_c

  cur_vec =tf.constant([[tknizer_eng.word_index['<start>']]])

  for i in range(20):

    infe_output , dec_state_h , dec_state_c , att_weights , _ = model.layers[1].onestep_decoder(cur_vec , en_outputs , dec_state_h , dec_state_c)

    attention_weights = tf.reshape(att_weights , (-1, ))

    attention_plot[i] = attention_weights.numpy()

    cur_vec = np.reshape(np.argmax(infe_output) , (1,1))

    pred.append(tknizer_eng.index_word[cur_vec[0][0]])

    if(pred[-1] == '<end>'):
      break
    
  translated = ' '.join(pred)

  return translated , attention_plot





# In[187]:


input_sentence ="devo vederlo"

sentence , attention_plot = predict(input_sentence , model_dot)
sentence


# In[207]:


plot_attention( "devo vederlo".split() , sentence.split() , attention_plot)


# In[209]:


from PIL import Image, ImageTk 

image = Image.open("ss_applied_ai.jpg") 
image


# In[35]:


x = validation.italian
x = x.reset_index()
x = x.drop("index" , axis = 1)
x


# In[ ]:


y


# <font color='blue'>**Calculate BLEU score**</font>

# In[33]:


#Create an object of your custom model.
#Compile and train your model on dot scoring function.
# Visualize few sentences randomly in Test data
# Predict on 1000 random sentences on test data and calculate the average BLEU score of these sentences.
# https://www.nltk.org/_modules/nltk/translate/bleu_score.html

#Sample example

import nltk.translate.bleu_score as bleu
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def get_blue_and_plot(model):
  bleu_score = []
  for i in tqdm(range(0,1000)):
    prediction ,_= predict(x.italian[i] , model)
    real = y["english_out"][i]
    prediction_list = prediction.split()
    real_list = real.split()
    #if(bleu.sentence_bleu(real_list , prediction_list) >= 0.8):
    #  print(i , x.italian[i])
    bleu_score.append(bleu.sentence_bleu(real_list , prediction_list))
  
  plt.figure(figsize=(8,8))
  plt.title("Bleu Score for 1000 italian-to-english sentences".upper())
  sns.kdeplot(bleu_score)
  plt.xlabel("BLEU-Scores")
  plt.grid(True)
  plt.show()


  


# In[218]:


get_blue_and_plot(model_dot)


# <font color='blue'>**Repeat the same steps for General scoring function**</font>

# In[24]:


#Compile and train your model on general scoring function.
# Visualize few sentences randomly in Test data
# Predict on 1000 random sentences on test data and calculate the average BLEU score of these sentences.
# https://www.nltk.org/_modules/nltk/translate/bleu_score.html

model_general  = encoder_decoder(embedding_size = 100,lstm_state = 268,input_length = 20,dec_units = 268,score_fun="general",att_units = 268,batch_size = 268)
optimizer = tf.keras.optimizers.Adam()
model_general.compile(optimizer=optimizer,loss= custom_lossfunction)
train_steps=train.shape[0]//1024
valid_steps=validation.shape[0]//1024
model_general.fit_generator(train_dataloader, steps_per_epoch=train_steps, epochs=20, validation_data=test_dataloader, validation_steps=valid_steps , callbacks=tensorboard_callback)


# In[27]:


input_sentence = "devo vederlo"
sentence , attention_plot = predict(input_sentence , model_general)
sentence


# In[30]:


plot_attention( "devo vederlo".split() , sentence.split() , attention_plot)


# In[36]:


get_blue_and_plot(model_general)


# <font color='blue'>**Repeat the same steps for Concat scoring function**</font>

# In[37]:


#Compile and train your model on concat scoring function.
# Visualize few sentences randomly in Test data
# Predict on 1000 random sentences on test data and calculate the average BLEU score of these sentences.
# https://www.nltk.org/_modules/nltk/translate/bleu_score.html
model  = encoder_decoder(embedding_size = 100,lstm_state = 268,input_length = 20,dec_units = 268,score_fun="concat",att_units = 268,batch_size = 268)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss= custom_lossfunction)
train_steps=train.shape[0]//1024
valid_steps=validation.shape[0]//1024
model.fit_generator(train_dataloader, steps_per_epoch=train_steps, epochs=20, validation_data=test_dataloader, validation_steps=valid_steps , callbacks=tensorboard_callback)


# In[38]:


input_sentence = "devo vederlo"
sentence , attention_plot = predict(input_sentence , model)
sentence


# In[39]:


plot_attention( "devo vederlo".split() , sentence.split() , attention_plot)


# In[40]:


get_blue_and_plot(model)


# In[26]:


# Write your observations on each of the scoring function
from prettytable import PrettyTable
    
x = PrettyTable()

x.field_names = ["Model-Architecture", "Maximum-Bleu-score"]

x.add_row(["simple encoder-decoder" , 0.6])
x.add_row(["---" , "--"])

x.add_row(["encoder-dot-decoder" , 0.8])
x.add_row(["---" , "--"])

x.add_row(["encoder-general-decoder" , 0.8])
x.add_row(["---" , "--"])

x.add_row(["encoder-concat-decoder" , 0.8])

print(x)

