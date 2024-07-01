from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

epochs = 1
# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the wikihow dataset
wikihow_dataset = load_dataset('wikihow')

# Define your dataset
class MyDataset(tf.data.Dataset):
    def _generator(texts):
        for text in texts:
            encoding = tokenizer.encode_plus(text, return_tensors='tf', padding='max_length', truncation=True, max_length=512)
            yield encoding['input_ids'], encoding['attention_mask']

    def __new__(cls, texts):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(texts, ),
            output_signature=(
                tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 512), dtype=tf.int32)
            )
        )

# Create your dataset
texts = ["Hello, I'm a language model.", "I'm training from scratch."]
dataset = MyDataset(wikihow_dataset['train']).batch(2)

class LLUE(tf.keras.Model):
    def __init__(self):
        super(LLUE, self).__init__()
        self.gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2')
        self.state = None
        # Initialize input_ids with 512 columns, first element is -1, others are 0
        self.input_ids = [[-1] + [0] * 511]

    def call(self, inputs):
        input_ids, attention_mask, new_tokens = inputs
        start_index = input_ids[0, 0]
        if start_index == -1:
            # Update input_ids
            input_ids = tf.concat([[1], new_tokens], axis=1)
            # Update attention_mask: 1 for the new token and new_tokens, 0 for the rest
            attention_mask = tf.concat([[1], tf.ones_like(new_tokens)], axis=1)
        if start_index == 1 and 512 - tf.shape(input_ids)[1] > tf.shape(new_tokens)[1]:
            # Update input_ids
            input_ids = tf.concat([1, new_tokens, input_ids[:, 1:]], axis=1)
            # Update attention_mask
            attention_mask = tf.concat([[1], tf.ones_like(new_tokens), attention_mask[:, 1:]], axis=1)
        if start_index == 1 and 512 - tf.shape(input_ids)[1] < tf.shape(new_tokens)[1]:
            # Determine the remaining space in the input_ids
            remaining_space = 512 - tf.shape(input_ids)[1]
            # Split the new tokens into two parts
            new_tokens_part1 = new_tokens[:, :remaining_space]
            new_tokens_part2 = new_tokens[:, remaining_space:]
            # Update input_ids
            new_tokens_part1_length = tf.shape(new_tokens_part1)[1]
            input_ids = tf.concat([512 - new_tokens_part1_length, new_tokens_part2, input_ids[:, 1:]], axis=1)
            input_ids[:, -new_tokens_part1_length:] = new_tokens_part1
            # Update attention_mask
            attention_mask = tf.concat([[1] * (512 - new_tokens_part1_length), tf.ones_like(new_tokens_part2), attention_mask[:, 1:]], axis=1)
            attention_mask[:, -new_tokens_part1_length:] = tf.ones_like(new_tokens_part1)
        if start_index != 1:
            # Update input_ids
            new_tokens_length = tf.shape(new_tokens)[1]
            input_ids[:, -start_index:-start_index + new_tokens_length] = new_tokens
            input_ids[:, 0] = -new_tokens_length
            # Update attention_mask
            attention_mask[:, -start_index:-start_index + new_tokens_length] = tf.ones_like(new_tokens)

        # num_new_tokens = tf.shape(new_tokens)[1]
        # # Remove the oldest tokens and add the new tokens
        # input_ids = tf.concat([input_ids[:, num_new_tokens:], new_tokens], axis=1)
        # # Update the last node to indicate the new starting point of the sequence
        # input_ids[:, -1] = tf.math.mod(input_ids[:, -1] + num_new_tokens, tf.shape(input_ids)[1])
        # # The attention mask should be updated in the same way
        # attention_mask = tf.concat([attention_mask[:, num_new_tokens:], tf.ones_like(new_tokens)], axis=1)
        # # rest of the function...
        # if self.state is None:
        #     # If there's no previous state, process the entire input
        #     output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        #     self.state = output
        # else:
        #     # If there's a previous state, process the new tokens and combine them with the previous state
        #     new_output = self.gpt2(input_ids=input_ids[:, -num_new_tokens:], attention_mask=attention_mask[:, -num_new_tokens:])
        #     output = tf.concat([self.state, new_output], axis=1)
        #     self.state = output
        # return output

# Create your model
model = LLUE()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=10000)

# Split the dataset into training and validation sets
train_dataset = dataset.take(int(len(dataset) * 0.8))
val_dataset = dataset.skip(int(len(dataset) * 0.8))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(dataset, epochs=epochs, validation_data=val_dataset, callbacks=[early_stopping, model_checkpoint])