# Comparative-Analysis-of-SuperHero-Classification-using-VGG-and-ResNet-with-Transfer-Learning-

Superheroes play a significant role in popular culture, movies, comics, and in various media.This project explores the classification of superheroes, mainly concentrated on five distinct characters. We leverage the power of transfer learning to implement two widely used deep learning architectures: VGG and ResNet. These pre-trained models have demonstrated exceptional performance on large-scale image classification tasks and offer a strong foundation for fine-tuning on specific datasets. The final model we came up with uses VGG as backbone. We utilized transfer learning by freezing all the layers and using only the last few to train on our smaller dataset. Since, VGG is a very good image classification model, it can definitely be used to solve the task at hand.

# Description
Initially we wanted to explore different models and tried using VGG 16 and Resnet 18. We used a dataset of 250 images. 200 for training and 50 for validation. We used two classes, one is Batman and the other is Not a hero. This Not a hero class consisted of images of humans, taken from different angles to help the model recognize ordinary humans properly. The Batman dataset consisted of images of batman from various movies, however we had to filter the batman data a bit, by manually churning to unusable images or images which have fanfic creations and are too far from a batman or images where the batman was without proper costume this cleaning process was done for every superhero dataset.

Here we observed the performance of both ResNet and VGG with transfer learning. For VGG we freezed everything except the last 3 Dense layers and modified the output to have 6 neurons equal to the number of classes (5 Superheroes and 1 Not a hero). For Resnet, we froze everything except the last Dense layer and convolution block. Although, both the models have the same 100% accuracy, ResNet seems to overfit the validation set which can be seen from the high training loss. Making us pick the VGG for the rest of the project.

The chosen network architecture is a modified VGG16 convolutional neural network, renowned for its effectiveness in image classification tasks. To adapt the architecture for the task, the last three layers of the pre-trained VGG16 model were unfrozen to allow fine-tuning. Additionally, the final fully connected layer was replaced with a dense layer consisting of six neurons, corresponding to the six target classes. The base VGG16 model, pre-trained on ImageNet, was sourced from official PyTorch or TensorFlow repositories to leverage its learned features and enhance performance on the new dataset.
For training, the model was configured with the Stochastic Gradient Descent (SGD) optimizer, utilizing a cross-entropy loss function. The learning rate was set to 0.001, with a momentum of 0.9, and the model was trained over ten epochs. The dataset comprised 1500 images, evenly distributed across six classes. The training split included 200 images per class, while the validation and testing splits each contained 25 images per class.

# Result
In the first step, we focused on classifying whether the given image was of Batman or not. Our model successfully identified the human images as not being a superhero, even when presented with images of Batman without his mask. This differentiation was a good initial validation of our approach. We expanded our classification further with four more distinct characters: Captain Marvel, Superman, Wonder Woman, and Deadpool. Our model successfully identified these characters, showing its ability to handle a broader range of superhero images.




