
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import io
from sklearn.model_selection import train_test_split
import time
import re
import unicodedata
import numpy as np



def inputYesNo(message:str) -> bool:
    s_Input = ""
    while  s_Input != "Y" and s_Input != "N":
        s_Input = input("{message} (Y/N)? ".format(message=message)).upper()
    if s_Input == "Y":
        return True
    else:
        return False


dataset_dir = "./brodley-transcripts-of-carl-rogers-therapy-sessions"
transcript_file_path = os.path.join(dataset_dir, "brodley-carl-rogers-T1.txt")
transcript_file_path_p1 = os.path.join(dataset_dir, "brodley-carl-rogers-T")
transcript_file_path_p2 = 1
transcript_file_path_p3 = (".txt")
transcript_2_file_path = os.path.join(dataset_dir, "brodley-carl-rogers-T2.txt")


while True:
    transcript_file_path_next = transcript_file_path_p1 + str(transcript_file_path_p2) + transcript_file_path_p3
    try:
        transcript_text_next = io.open(transcript_file_path_next, encoding='UTF-8').read().strip()
        print ("\nOpened transcript: " + transcript_file_path_next)
    except:
        break
    lines_next =  transcript_text_next.split('\n')
    if transcript_file_path_p2 == 1:
        # Each line in the file is one person speaking
        # We assume the therapist always talks first
        client_lines_all = [line for line_num, line in enumerate(lines_next, start=1) if line_num % 2 == 0]
        client_lines_all.insert(0,u'[C]') #Client leads conversation with empty token
        therapist_lines_all = [line for line_num, line in enumerate(lines_next, start=1) if line_num % 2 != 0]

    else:
        # Each line in the file is one person speaking
        # We assume the therapist always talks first
        client_lines_next = [line for line_num, line in enumerate(lines_next, start=1) if line_num % 2 == 0]
        client_lines_next.insert(0,u'[C]') #Client leads conversation with empty token
        therapist_lines_next = [line for line_num, line in enumerate(lines_next, start=1) if line_num % 2 != 0]
        client_lines_all += client_lines_next
        therapist_lines_all += therapist_lines_next
    transcript_file_path_p2+= 1
pair_lines = list(zip(client_lines_all,therapist_lines_all))

# transcript_text = io.open(transcript_file_path, encoding='UTF-8').read().strip()
# transcript_2_text = io.open(transcript_2_file_path, encoding='UTF-8').read().strip()
# lines =  transcript_text.split('\n')
# lines_2 =  transcript_2_text.split('\n')
# # Each line in the file is one person speaking
# # We assume the therapist always talks first

# client_lines = [line for line_num, line in enumerate(lines, start=1) if line_num % 2 == 0]
# therapist_lines = [line for line_num, line in enumerate(lines, start=1) if line_num % 2 != 0]

# #Client leads conversation with empty token
# client_lines.insert(0,u'[C]')

# client_lines_2 = [line for line_num, line in enumerate(lines_2, start=1) if line_num % 2 == 0]
# therapist_lines_2 = [line for line_num, line in enumerate(lines_2, start=1) if line_num % 2 != 0]

# print(therapist_lines_2[0:3])
# print(therapist_lines[0:3])

#Client leads conversation with empty token
# client_lines_2.insert(0,u'[C]')

# client_lines = client_lines + client_lines_2
# # client_lines.append(client_lines_2)
# therapist_lines = therapist_lines + therapist_lines_2
# therapist_lines.append(therapist_lines_2)

# pair_lines = list(zip(client_lines,therapist_lines))
# print (pair_lines)
# for (c_Line, t_Line) in pair_lines[0:2]:
#     print(c_Line,t_Line)


#Now clean up each line


''' Converts the unicode file to ascii '''
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    # print (w)
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"\[t\]", " ", w )
    w = re.sub(r"\[c\]", " ", w)
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w) #add spaces around punctuation
    
    
    w = re.sub(r'[" "]+', " ", w) ##collapse multiple spaces into one
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿\-\']+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


clean_pair_lines = [(preprocess_sentence(c_clean), preprocess_sentence(t_clean)) for c_clean, t_clean in pair_lines ]
for c_Line, t_Line in clean_pair_lines[0:2]:
    print(c_Line,t_Line)


clean_client_lines, clean_therapist_lines = zip(*clean_pair_lines)

#Now we must encode?
#Yes. Instead of doing a index mapping we can use a tokeniser to make this easy

client_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    # We specify no filters, as the default will filter out punctuation
therapist_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

#Generate the encodings by using our input texts as samples
client_tokenizer.fit_on_texts(clean_client_lines)
therapist_tokenizer.fit_on_texts(clean_therapist_lines)

#Convert our two "languages"
client_tensor = client_tokenizer.texts_to_sequences(clean_client_lines)
client_tensor = tf.keras.preprocessing.sequence.pad_sequences(client_tensor, padding="post")
therapist_tensor = therapist_tokenizer.texts_to_sequences(clean_therapist_lines)
therapist_tensor = tf.keras.preprocessing.sequence.pad_sequences(therapist_tensor, padding="post")

max_length_targ, max_length_inp = therapist_tensor.shape[1], client_tensor.shape[1]

print("max targ: ", max_length_targ, "\nmax inp: ", max_length_inp)
input("This ok?")
#Our data is now in a form that we can use it.
#We will want to split the data into training and validation data sets.
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(client_tensor, therapist_tensor, test_size=0.2)
input_tensor_train, target_tensor_train = client_tensor, therapist_tensor

# For efficient processing, and to help with overfitting, we want our training data to be shuffled
# into batches.
# We want the result to be a tf.Dataset

#Before this we will want set up some parameters and hyperparameters.
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 20
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 256
in_vocab_size = len(client_tokenizer.word_index) + 1
out_vocab_size = len(therapist_tokenizer.word_index) + 1
print("Client vocab size: ", in_vocab_size, "Therapist vocab size: ", out_vocab_size)
input("This ok?")
#Now we have defined our parameters and hyper parameters, we can create batches

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)
# print("Dataset structure: {ds}".format(ds=dataset.element_spec))
#We have already padded our inputs, no reason to do this now.

#Next up is to create our encoder model.
#If I remember correctly, we will have an embedding layer
#This is fed an encoded (integer) input word
#It then makes a prediction of the next word too?
#Both are fed into the next layer?
#Correct, and to do this intuitively we create an extension to tf.keras.Model.
# This way we can easily define what happens each step

class Encoder(tf.keras.Model):
    """
    
    """
    def __init__ (self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units

        self.embedding = tf.keras.layers.Embedding(input_dim = self.vocab_size, output_dim = embedding_dim)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences = True,
                                        return_state = True,
                                        recurrent_initializer='glorot_uniform')
                        # We want sequences as these are the outputs we pay attention to

    def call(self, x, prev_hidden):
        x = self.embedding(x)
        # now we have embedding, we want to generate a new state and output
        output_sequence, hidden_state  = self.gru(x, initial_state = prev_hidden)
        return output_sequence, hidden_state

    
    def initialise_hidden_state(self):
        # To do this we need to have some insight about the shape of the state.
        # We are trying to encode some information about the words with 
        # respect to the sequence.
        # That I think is why we land on batch_sz * enc_units
        return tf.zeros((self.batch_sz,self.enc_units))
            

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self,units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):

        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score

        # Add a time axis to the query (decoder state) so the axis match that 
        # of the encoder sequence which has time (word index) as the second axis
        query_with_time = tf.expand_dims(query,1)

        # Calculate scores across time
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        
        self.W1(query_with_time) + self.W2(values)
        tf.nn.tanh(self.W1(query_with_time) + self.W2(values))
        score = self.V(tf.nn.tanh(self.W1(query_with_time) + self.W2(values)))

        # To provide us with a probability distribution over each word, so that the total of the weghts
        # is equal to one, we apply a softmax function over the time axis.
            # Note this could be different amounts of the 1st axis e.g. for batch_size = 4, max_sequence = 4
            # [0.1, 0.3, 0.1, 0.5]
            # [0.0, 0.4, 0.2, 0.4]
            # [0.3, 0.3, 0.1, 0.3]
            # [0.1, 0.6, 0.3, 0.0]
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences = True, return_state = True, recurrent_initializer = 'glorot_uniform')
        self.attention = BahdanauAttention(self.batch_sz)

        self.fc = tf.keras.layers.Dense(vocab_size)


    def call(self, enc_outputs, x, hidden_state):
        context_vector, attention_weights = self.attention(hidden_state, enc_outputs)

        x = self.embedding(x)

        # attention_vector =
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


encoder = Encoder(in_vocab_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(out_vocab_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

checkpoint_dir = './wellbeing_checkpoints_2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                encoder=encoder,
                                decoder=decoder)



def loss_function( real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)



@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_outputs, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden

        # Create starting word
        dec_in = tf.expand_dims([therapist_tokenizer.word_index["<start>"]] * BATCH_SIZE, 1)

        for t in range(1,targ.shape[1]):
            predictions, dec_hidden, _ = decoder(enc_outputs,dec_in,dec_hidden)

            loss += loss_function(targ[:,t], predictions)

            dec_in = tf.expand_dims(targ[:,t], 1)
            

    #average loss across the sequence
    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients,variables))

    return batch_loss



def training_loop():
    # Loop over epoch index
    EPOCHS = 10
    total_loss = 0
    # Reset loss for the epoch
    for epoch in range(EPOCHS):
        epoch_loss = 0
        start = time.time()

        enc_hidden = encoder.initialise_hidden_state()

        # For each epoch we are taking a given number of batches
        # We have the batch size, but we are looping over the steps per epoch (num of batches)
        # As we made this a batched dataset, when take from the dataset a batch is returned.
        # So we take the number of batches we need
        for batch, (inp,targ) in enumerate(dataset.take(steps_per_epoch)):
            
            # Reset the batch loss
            batch_loss = train_step(inp,targ, enc_hidden)
            epoch_loss += batch_loss

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                        batch,
                                                        batch_loss.numpy()))

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # We display some information as we go
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                epoch_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        total_loss += (epoch_loss / steps_per_epoch)
    total_loss = total_loss / EPOCHS
    print('Total Loss {:.4f}'.format(total_loss))

b_load = inputYesNo("Load checkpoints?")
if not b_load:
    training_loop()
else:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

'''Evaluation is very similar to the training step.
    - The evaluate function is similar to the training loop,
    except we don't use teacher forcing here. The input to the decoder
    at each time step is its previous predictions along with the hidden
    state and the encoder output.
    - Stop predicting when the model predicts the end token.
    - And store the attention weights for every time step.
'''
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    
    inputs = [client_tokenizer.word_index[i] for i in sentence.split(' ')]
    
    
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=max_length_inp,
                                                                padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''

    #!!! No - this has batch size > 1
    # enc_hidden = encoder.initialise_hidden_state()
    # We want batch size = 1 as we are making a prediction of one sentence
    
    enc_hidden = [tf.zeros((1, units))]
    enc_outputs, enc_hidden = encoder(inputs, enc_hidden)
    dec_hidden = enc_hidden

    dec_in = tf.expand_dims([therapist_tokenizer.word_index["<start>"]] , 0)
    # We expand at axis 0 here to give batch size dimension.
    result = ''
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(enc_outputs, dec_in, dec_hidden)

        # predictions are of size (batch_sz, vocab_size) - here (1, vocab_size)
        sampled_indices = tf.random.categorical(predictions, num_samples=1)
        # print (sampled_indices)
        sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
        # print ("Sampled indices ",sampled_indices)
        prediction_id = tf.math.argmax(predictions[0,:]).numpy()
        # print ("prediction_id ",prediction_id)
        result += therapist_tokenizer.index_word[sampled_indices[0]] + ' '
        
        if therapist_tokenizer.index_word[sampled_indices[0]] == "<end>":
            return result,sentence

        dec_in = tf.expand_dims([sampled_indices[0]], 0)

    return result,sentence








# def evaluate( sentence):
#     # Attention plot is the matrix relating how strongly a target word is related
#     # to an input word -showing how much attention we should pay based on this relationship
#     attention_plot = np.zeros((max_length_targ, max_length_inp))

#     # Start and end tags and space correctly
#     sentence = preprocess_sentence(sentence)

#     # Convert sentence into array of tokens
#     inputs = [client_tokenizer.word_index[i] for i in sentence.split(' ')]
#     # Pad of course
#     inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
#                                                             maxlen=max_length_inp,
#                                                             padding='post')
#     inputs = tf.convert_to_tensor(inputs)

#     result = ''

#     # Initial encoder state
#     hidden = [tf.zeros((1, units))]

#     # Apply encoder to the input, getting the final hidden state and a final encoded output
#     enc_out, enc_hidden = encoder(inputs, hidden)

#     #Initialise decoder state using encoder final state
#     dec_hidden = enc_hidden

#     # Create start tag as initial word seen by decoder
#     dec_input = tf.expand_dims([therapist_tokenizer.word_index['<start>']], 0)

#     for t in range(max_length_targ):
#         predictions, dec_hidden, attention_weights = decoder(dec_input,
#                                                             dec_hidden,
#                                                             enc_out)

#         # storing the attention weights to plot later on
#         attention_weights = tf.reshape(attention_weights, (-1, ))
#         attention_plot[t] = attention_weights.numpy()

#         predicted_id = tf.argmax(predictions[0]).numpy()

#         result += therapist_tokenizer.index_word[predicted_id] + ' '

#         if therapist_tokenizer.index_word[predicted_id] == '<end>':
#             return result, sentence, attention_plot

#         # the predicted ID is fed back into the model
#         dec_input = tf.expand_dims([predicted_id], 0)

#     return result, sentence, attention_plot

def chat( sentence):
    # result, sentence, attention_plot = evaluate(sentence)
    result, sentence= evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Answer: {}'.format(result))

    # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]



b_Manual = inputYesNo("Try a phrase of your own")
if b_Manual:
    s_Phrase = input()
    while s_Phrase != "":
        # chat(s_Phrase)
        chat(s_Phrase)
        s_Phrase = input()