# Predicting fetal lung response

Here, we are building robust machine learning models from ultrasound and physiological data to support efforts to create a fully-automated artificial placenta: https://www.sciencedirect.com/science/article/abs/pii/S0002937819304727?via%3Dihub

Authors: Suhaas Bhat, Garyk Brixi, Pranam Chatterjee

# How to use:
The inference.py script loads a model, and yields predictions for a given directory of fetal lung images. You can choose the model type - either SVM or RFC.
The input looks like:

`python inference.py [folder of inference data] [svm or rfc]`

So for example, to get predictions with our trained SVM on a folder named "test_images", one would run:

`python inference.py test_images svm`
