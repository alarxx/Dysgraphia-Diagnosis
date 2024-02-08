# DysgraphiaRMAT
Research Methods and Tools.

Application of image processing and deep learning technologies for the diagnosis of dysgraphia.

A dataset of Slovak manuscripts collected by a group of 120 school-aged children, 63 healthy examples and 57 examples with dysgraphia, was used. Data collected from Wacom tablets were converted into digital images. Computer Vision and Deep Learning methods were used to process, analyse images and detect dysgraphia. 

Models such as VGG and ResNet were emphasised. The dataset was divided into a training, validation and test dataset to confirm the reliability of the results. 
To avoid overfitting the model, I used augmentation techniques such as random cropping, rotating the image and augmenting the data by removing a random percentage of words from the image.
The results turned out to be about 80% accurate given such a small dataset.

A web application was developed as an interface to upload images and get a diagnosis from our classification model.

Good results were obtained. Future research may include data collection with Cyrillic writing: Russian, Kazakh. 

Additional data on sample participants:

![Subjects_Data](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/Subjects_Data.png)

Convert raw raw data (.svc, not .csv) to graphical vector view form:

![Dataset visualisation](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/sentence.png)

Healthy Example:

![Healthy_Example](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/Healthy_Example.png)

Dysgraphia Example:

![Dysgraphia_Example](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/Dysgraphia_Example.png)


Convert to Image Form:

![Image conversion](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/image%20with%20handwriting.png)

Words Detection:

![Words_Detection](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/Words_Detection.jpg)

Augmenting the data by removing a random percentage of words from the image:

![Remove_Words](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/Remove_Words.jpg)

![Test](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/test_code.jpg)
![Results](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/evaluation.png)
![All_Results](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/All_Results.png)

An interface for uploading images and obtaining a diagnosis from our classification model:

![Web_Application](https://github.com/Alar-q/DysgraphiaRMAT/blob/main/git_images/Web_Application.png)

## References

[GitHub](https://github.com/peet292929/Dysgraphia-detection-through-machine-learning) This repository contains data for the paper Dysgraphia detection through machine learning by M. Dobes and P. Drotar published in Scientific Reports. Cite as: Drotár, P., Dobeš, M. Dysgraphia detection through machine learning. Sci Rep 10, 21541 (2020). https://doi.org/10.1038/s41598-020-78611-9
