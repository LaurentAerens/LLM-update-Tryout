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

    def call(self, inputs):
        input_ids, attention_mask = inputs
        if self.state is None:
            # If there's no previous state, process the entire input
            output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            self.state = output
        else:
            # If there's a previous state, process the new tokens and combine them with the previous state
            new_output = self.gpt2(input_ids=input_ids[:, -self.new_tokens:], attention_mask=attention_mask[:, -self.new_tokens:])
            output = tf.concat([self.state, new_output], axis=1)
            self.state = output
        return output

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