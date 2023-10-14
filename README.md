# Machine Learning-Naive Bayes Dataset Analysis
The goal is to use machine learning to identify different types of bees from a huge bee images dataset.
- This notebook walks through building a simple deep learning model that can automatically detect honey bees and bumble bees and then loads a pre-trained model for evaluation

### Import Libraries
- Import `keras`, the deep learning library you'll be using.
- Import the function `Sequential` from the `models` module of `keras`. This is the model type I'll use.
- Import the functions  `Dense`, `Dropout`, `Flatten`, `Conv2D`, `MaxPooling2D` from the `layers` module of `keras`. These will form the different layers of your ***convolutional neural network*.

```
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
```

### Load Image Labels
Now that we have all of our imports ready, it is time to look at the labels for our data. We will load our labels.csv file into a DataFrame called `labels`, where the index is the image name (e.g. an index of 1036 refers to an image named 1036.jpg) and the **genus** column tells us the bee type. Genus takes the value of either 0.0 (Apis or honey bee) or 1.0 (Bombus or bumble bee).

### Instructions
Load the DataFrame of labels and image names, explore the dataset, then assign image labels to y.

- Using read_csv function from `pandas`, load the labels.csv file which lives in the datasets folder. Be sure to set index_col=0 so that the images names are loaded as the index.
- Print the value counts of genus in the labels DataFrame. The dataset is imbalanced between the two classes.
- Assign the genus column values (the array of image labels) to y.

Recall labels.csv lives within the datasets folder, so it can be loaded with labels = pd.read_csv('datasets/labels.csv', index_col=0). Remember to set index_col=0 so that the image id is the index. A numpy array of the values of a column be accessed from a pandas DataFrame with .values so labels.genus.values return the values of the genus column as an array.

### Eamine RGB values in an image matrix
Image data can be represented as a matrix. The width of the matrix is the width of the image, the height of the matrix is the height of the image, and the depth of the matrix is the number of channels. Most image formats have three color channels: red, green, and blue.
<p>For each pixel in an image, there is a value for every channel. The combination of the three values corresponds to the color, as per the <a href="https://en.wikipedia.org/wiki/RGB_color_model">RGB color model</a>. Values for each color can range from 0 to 255, so a purely blue pixel would show up as (0, 0, 255).

### Instructions
Load the first image from your DataFrame and explore its shape and RGB values.
- Load the first image from the labels DataFrame index using Image.open, the image loading function from `scikit-image` and assign it to example_image
- Display example_image using plt.imshow()
- Print the shape of example_imageto see that it is 50 by 50 pixels and has 3 channels.
- Print the R, G, and B values for the top left pixel of the example_image. Recall that the image has shape (X, Y, Z) and that the that the X and Y coordinates for this pixel are (0, 0).

###### HINT
Load your example image with Image.open()
- Remember that arrays can be sliced just like lists. The RGB channels for the top left pixel of the image can be accessed with `example_image[0, 0, :]`

