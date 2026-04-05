Visual Speech Activity Detection

I’m very pleased to see that you note where your expectations were not matched (regarding ResNet18), instead of reformulating to give the appearance that this was intended.
Regarding the feedback from the peer reviews: it could perhaps have been made more explicit in the main body of the text that classification was based on isolated frames, rather than frames in context.
The results with ResNet are quite impressive, all the more so because the frames are isolated.

The group trained a speech activity detection system based on video frames, comparing performance among a number of methods. Their baseline, based on ResNet18, achieves quite high accuracy: considering the lack of context, this is particularly impressive.

SSL for TIMIT

Nitpick: a number of citations are confusingly presented without parentheses (i.e., "Yu et al. (2019)" instead of "(Yu et al., 2019)" when referred to indirectly.
Data augmentation techniques could be used to increase the training data.
Otherwise, I think the approach was solid, and you did quite well, given the limited amount of data available.

This project attempted to use supervised learning techniques for phoneme-based speech recognition. They emplyed a more capable, bidirectional model as a teacher model to train a unidirectional model. Though the results are far from impressive on the face of it, when the limited size of the dataset (TIMIT) is taken into account, the results are quite good.

Comparison of auto/manual ASR

This group trained an ASR system using self-supervised methods, also employing a teacher model. The group differentiated itself by using a hyperparameter search to find the optimal teacher model, and by comparing performance over multiple subsets of the data.