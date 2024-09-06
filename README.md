# Image Classifier
Simple image classification using PyTorch. 
## How To Use
### Testing
in order to simply run the model, run the `tester.py` file. 

If you want to test your own images, at the moment they need to be 32 x 32 pixels. Once your images are that size, add them to the `\testImages` folder, and update line 11 in tester to add your own images. The line should look something like this:
```
image_paths = ['./testImages/Frog.jpg','./testImages/cat.jpg']  # Add paths to your images here
 ```
### Training
To retrain the model yourself, simply run `classifier.py`, and set the number of training epochs to fit your needs.
## Purpose
This was mainly made to help me learn more about PyTorch, as a lot of the code was not my own, and I've been mainly tinkering with everything to learn how it works. This is part of a course I'm taking on Deep learning: _A deep understanding of deep learning_ by Mike X Cohen on Udemy.
