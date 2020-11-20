# Computer-Vision
UCF Computer Vision CAP 5415 from Fall 2019


## Detection and Classification of ATR Dataset

Using ATR Dataset(Infrared dataset with various sizes of objects at different ranges at which data is being captured and with high level of clutter per target ratio, the problem is quiet challenging.

Used UNet shaped 3DCNN fed with the difference in consecutive image as input. The temporal dimensionality resulted in providing better output. 
IoU of region detected by UNet and actual detection marks the performance of the model.
The detected region is then fed to the ResNeXt model for classifying to various vehicle classes(10 Classes of Army vehicles'

Used a resneXt for classification after detection in 10 classes of the object

## Digit Classification Using Harris Detector and CNN
Classification digit Dataset is being classified using the Harris corner Detector to detect features and classify.
Then the resutls are compared to the results of a CNN network trained on the Digit Dataset.

## Using Fisher Analysis and CNN on Image

Using Fisher Analysis to analyze images
