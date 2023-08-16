# U-Net Model for Segmentation

The U-Net model is a popular architecture for image segmentation tasks. It consists of an encoder and a decoder, which are connected by skip connections. The model takes an input image and produces a segmentation map as the output.

## Architecture

The U-Net architecture is composed of the following components:

- **DoubleConv**: This is a basic building block of the U-Net, which consists of two 3x3 convolutional layers with batch normalization and ReLU activation functions.

- **Encoder**: The encoder part of the U-Net consists of four down-sampling blocks. Each block applies a DoubleConv operation followed by max-pooling to reduce the spatial dimensions of the input feature maps.

- **Decoder**: The decoder part of the U-Net consists of four up-sampling blocks. Each block applies a transposed convolution (or deconvolution) followed by a DoubleConv operation to up-sample the feature maps and concatenate them with the corresponding feature maps from the encoder.

- **Skip Connections**: Skip connections are established by concatenating the feature maps from the encoder with the corresponding feature maps in the decoder. This helps in preserving spatial information and aids in better segmentation accuracy.

- **Output Layer**: The final output layer is a 1x1 convolutional layer with a sigmoid activation function. This produces the final segmentation map, where each pixel represents the predicted class for the corresponding input pixel.

## Usage

To use the U-Net model for segmentation in PyTorch, you can define an instance of the `UNet` class provided in the code. You can then pass input images through the model using the `forward` method, which will produce the segmentation maps as outputs.

```python
model = UNet(in_channels, out_channels)  # Initialize the U-Net model
output = model(input)                    # Pass the input through the model
