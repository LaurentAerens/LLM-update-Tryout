# LLM Update Tryout

This repository contains the code for training a Language Model from scratch using the GPT-2 architecture. The model is trained on the WikiHow dataset.
The model is called LLUE (Efficient Update Language Model) this is based on NNUE (Efficient Neural Network Update) which is a neural network architecture used in computer chess.
I based myself on GPT-2 and updated the input layer and methode in order to make it more efficient and faster.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- transformers
- tensorflow
- datasets

You can install them using pip:

```shell
pip install transformers tensorflow datasets
```

## Getting Started

1. Clone this repository:

```shell
git clone https://github.com/laurentaerens/LLM-update-Tryout.git
```

2. Change into the project directory:

```shell
cd LLM-update-Tryout
```

3. Run the Python script:

```shell
python main.py
```

## Training

The training process consists of the following steps:

1. Load the pre-trained GPT-2 tokenizer.
2. Load the WikiHow dataset.
3. Define a custom dataset class.
4. Create the dataset using the custom class.
5. Define the LLUE model.
6. Compile the model.
7. Train the model using the dataset.

## Results

The trained model will be saved as `best_model.h5` and can be used for various natural language processing tasks.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
