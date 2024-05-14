DAT255

1. **Deep Learning**: A subset of machine learning that involves artificial neural networks with multiple layers, allowing the model to learn intricate patterns and representations from data.
   
2. **Neural Network**: A computational model inspired by the structure and function of biological neural networks, composed of interconnected nodes or neurons organized in layers.

3. **Activation Function**: A function applied to the output of each neuron in a neural network layer, determining the neuron's output and introducing non-linearity to the model.

4. **Loss Function**: A function that quantifies the difference between predicted and actual values in a machine learning model, serving as a measure of the model's performance.

5. **Gradient Descent**: An optimization algorithm used to minimize the loss function by iteratively adjusting the model's parameters in the direction of the steepest descent of the loss gradient.

6. **Backpropagation**: A technique for calculating the gradients of the loss function with respect to the parameters of the neural network, enabling efficient optimization using gradient descent. "The algorithm for determening how a single training ecample would like to 'nudge' the weights and biases (parameters) - not just in term of whether they should go up or down, but in term of what relative porpotions to those changes cause the most rapid decrease to the cost"

7. **Mini-batch Gradient Descent**: An optimization variant where the gradient descent updates are computed using small subsets (mini-batches) of the training data, balancing computational efficiency and model convergence.

8. **Learning Rate**: A hyperparameter that determines the step size of parameter updates during gradient descent, influencing the convergence speed and stability of the optimization process.

9. **Overfitting**: A phenomenon where a machine learning model performs well on the training data but fails to generalize to unseen data, often caused by excessive complexity or lack of regularization.

10. **Regularization**: Techniques employed to prevent overfitting by imposing constraints or penalties on the model parameters. The main purpose of regularization is to slightly modify the learning algorithm so that the model generalizes better. This is achieved by adding a penalty on the size of the coefficients or the complexity of the model. Regularization helps to ensure that the model is robust and performs well not only on the training data but also on new, unseen data.
- *Types*
    1. L1 (Lasso) : Adds a penalty equal to the absolute value of the magnitude of coefficients. This can lead to some coefficients being exactly zero, which is a form of automatic feature selection. It is useful when we believe many features are irrelevant or when we prefer a sparse model.
    2. L2 (Ridge) Adds a penalty equal to the square of the magnitude of coefficients. This tends to spread the error among all terms and leads to smaller coefficients, but it does not necessarily eliminate coefficients entirely. It is useful when we believe that all features have an impact on the output.
    3. Dropout: A regularization technique that randomly sets a fraction of input units to 0 at each update during training time, which helps to prevent neurons from co-adapting too much.
    4. weight decay: A regularization technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function based on the magnitude of the model's weights. It encourages the model to learn simpler and smoother representations by penalizing large weight values.
    5. Early stopping:Involves stopping the training process before the training completes if the performance on a validation dataset starts to deteriorate or ceases to improve significantly. This prevents overfitting by not allowing the model to train too long and over-learn the training data.



11. **Data Augmentation**: A technique for increasing the diversity of training data by applying transformations such as rotation, scaling, and flipping, thereby enhancing the model's robustness and generalization.

12. **Transfer Learning**: A machine learning approach where knowledge gained from training one model on a particular task is transferred or adapted to a related task, often by fine-tuning the pre-trained model.

13. **Fine-tuning**: The process of further training a pre-trained model on new data or tasks, typically by adjusting its parameters while keeping some of the earlier learned representations intact.

14. **fastai**: A deep learning library built on top of PyTorch, providing high-level abstractions and utilities for simplifying the development and training of neural network models.

15. **DataBlock**: A component of fastai for flexible and customizable data preprocessing and loading, enabling seamless integration of various data sources and formats into the training pipeline.

16. **Learner**: The central object in fastai representing the model, data, optimization process, and training loop, providing an interface for training, evaluation, and inference.

17. **Image Classification**: A task in computer vision where the goal is to assign a label or category to an input image based on its visual content, often accomplished using deep learning models.

18. **Natural Language Processing (NLP)**: A field of artificial intelligence focused on enabling computers to understand, interpret, and generate human language, often employing deep learning techniques.

19. **Transformer**: A neural network architecture introduced in the "Attention is All You Need" paper by Google, widely used in NLP tasks for its parallelization capabilities and effectiveness in capturing long-range dependencies.

20. **GPT (Generative Pre-trained Transformer)**: A series of transformer-based models developed by OpenAI for natural language generation tasks, leveraging large-scale unsupervised pre-training followed by fine-tuning on specific tasks.

21. **Computer Vision**: A field of artificial intelligence focused on enabling computers to interpret and analyze visual information from the real world, often employing deep learning techniques for tasks like object detection and image segmentation.

22. **Object Detection**: A computer vision task involving the identification and localization of objects within an image, often achieved by predicting bounding boxes and class labels for each object instance.

23. **Segmentation**: A computer vision task where the goal is to partition an image into multiple segments or regions, typically representing different objects or regions of interest, often used in medical imaging and scene understanding.

23. **Reinforcement Learning**: A machine learning paradigm where an agent learns to make decisions by interacting with an environment, receiving rewards or penalties based on its actions, often used in game playing and robotics.



25. **fine_tune() method in fastai**: A method or function commonly found in deep learning libraries like fastai, used to further train a pre-trained neural network model on a new dataset or task. Fine-tuning typically involves adjusting the parameters of the pre-trained model while keeping some of the earlier learned representations intact, thereby leveraging the knowledge gained from pre-training to improve performance on the new task.
-* What id does*
    1. Pretrained Model Usage: Starts with a model that has already been trained on another dataset (usually a large and generic dataset like ImageNet).
    2. Freeze Training: Initially, all the layers of the model except the last few are frozen. This means that their weights are not updated during the first phase of training.
    3. Training the New Layers: The unfrozen layers are then trained on the new dataset for a specified number of epochs.
    4. Unfreezing and Fine-Tuning: Optionally, after the initial training of the new layers, the entire model can be unfrozen and all layers can be fine-tuned together for additional epochs.
- *Most important parameters*
    1. epochs: Total number of epochs to train the model. This is divided into two phases — initial training of the new layers and full model fine-tuning.
    2. base_lr: The learning rate to be applied. FastAI may apply this learning rate following its discriminative fine-tuning philosophy, where different layers can have different learning rates.
    3. freeze_epochs: Number of epochs during which the pretrained layers are frozen and only the last few layers are trained.


26. **DataBlock**: In the context of fastai, DataBlock is a high-level abstraction for defining the data processing pipeline in machine learning tasks. It allows users to specify how to transform raw data into a suitable format, e.g. bulding a dataset. DataBlock provides flexibility and customization options for preprocessing data and creating DataLoaders.

27. **DataLoaders**: DataLoaders are objects used to efficiently load and iterate over batches of data during the training, validation, and testing phases of a machine learning task. In fastai, DataLoaders are typically created using the DataBlock API, encapsulating the training and validation datasets along with data augmentation, batching, shuffling, and other data processing operations. They provide a convenient interface for feeding data to the training loop.

28. **Embedding**: In the context of deep learning, an embedding refers to a mapping from discrete categorical variables (such as words, tokens, or categories) to continuous vector representations in a lower-dimensional space. Embeddings are learned during the training process and capture semantic relationships between categorical values, enabling neural networks to effectively process and generalize from categorical input data.

29. **Learning Rate**: The learning rate is a hyperparameter that controls the step size of parameter updates during the training process of a neural network. It determines how much the model parameters are adjusted in the direction of the gradient during optimization. A suitable learning rate is crucial for achieving optimal convergence and performance in training neural networks, with values typically chosen through experimentation and validation.

30. **Learning Rate Finder**: A technique used to determine an appropriate learning rate for training neural networks, particularly in the context of stochastic gradient descent optimization algorithms. The learning rate finder involves gradually increasing the learning rate during the early stages of training while monitoring the loss function's behavior. The optimal learning rate is typically identified as the point where the loss begins to decrease most rapidly or levels off.
- *How It Works*
    1. Initial Setup: The learning rate finder starts with a very small learning rate and gradually increases it over a series of iterations.
    2. Training Mini-batches: For each mini-batch, the model is trained with the current learning rate, and the learning rate is then increased exponentially.
    3. Tracking Loss: It records the loss at each step. The idea is to see how the loss changes as the learning rate increases.
    4. Plotting Loss: After running through the mini-batches, the finder plots the loss against the learning rates.

31. **Metric**: A metric is a measure used to evaluate the performance of a machine learning model on a specific task or dataset. Metrics quantify different aspects of model performance, such as accuracy, precision, recall, F1-score, mean squared error, or area under the receiver operating characteristic curve (ROC-AUC). Metrics are essential for assessing the effectiveness and quality of trained models and guiding model selection and optimization processes.
- *Taskt and its most used metrics*
    1. Classification
        - Accuracy: The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined
        - Cross Entropy Loss (Log Loss):  A performance metric for evaluating the probabilities output by a classifier as opposed to its discrete   predictions. Lower log loss values indicate better performance, with perfect models having a log loss of zero.
    2. Classification, especially with imbalenced datasets
        - Precision and Recall
            - Precision: The ratio of true positive predictions to the total predicted positives - accuracy of positive predictions.
            - Recall: The ratio of true positive predictions to the actual positive cases - ability to detect positive instances.
    3.  Classification tasks where balancing precision and recall is important.
        - F1 Score: The harmonic mean of precision and recall. It is particularly useful when you need a balance between precision and recall, and there is an uneven class distribution.
    4. Regression (predicting an outcome:
        - MAE:The average of the absolute differences between the predicted values and the actual values. It gives an idea of how wrong the         predictions were; the lower the MAE, the better. 
        - MSE: The average of the squares of the differences between the predicted values and the actual values. It heavily penalizes larger errors compared to MAE.

32. **Generative AI**: Generative AI refers to artificial intelligence systems and algorithms that are capable of generating new content, such as images, text, audio, or video, that mimics or resembles human-created data. Generative AI models often learn to capture and replicate the underlying distribution of training data, enabling them to produce novel and realistic outputs. 

33. **Collaborative Filtering**: Collaborative filtering is a technique used in recommendation systems to predict a user's preferences or interests based on the preferences of similar users or items. It leverages the collective wisdom of a group of users to make personalized recommendations, typically by analyzing user-item interaction data such as ratings, reviews, or purchase history. Collaborative filtering can be implemented using different approaches, including user-based filtering, item-based filtering, and matrix factorization methods.


34. **ResNet**: ResNet, short for Residual Network, is a deep neural network architecture proposed by Kaiming He et al. in their 2015 paper "Deep Residual Learning for Image Recognition." ResNet introduced the concept of residual blocks, where the input to a block is added to its output, allowing for easier training of very deep neural networks. ResNet architectures have achieved state-of-the-art performance in various computer vision tasks, including image classification, object detection, and semantic segmentation.


35. **Hyperparameters**: Hyperparameters are parameters whose values are set before the training process begins and remain constant during training. Examples include learning rate, batch size, and the number of layers in a neural network. Tuning hyperparameters is a crucial part of optimizing the performance of machine learning models.


36. **`Resize()`** is a method commonly used in image processing and computer vision tasks to resize images to a specified size. It is used to standardize the dimensions of images in a dataset, ensuring that all images have the same width and height, which is often necessary for feeding them into neural networks for training or inference. 

37. **item_tfms()**: In libraries like fastai, "item_tfms()" refers to transformations applied to individual items or samples in a dataset. These transformations typically include resizing, cropping, normalization, and other preprocessing steps performed on each item before they are used for training or inference.

38. **batch_tfms()**: Similar to "item_tfms()", "batch_tfms()" in fastai refers to transformations applied to batches of data during the training process. These transformations are typically applied after data augmentation and are useful for tasks such as data normalization, data augmentation, and regularization.

39. **batch**: A batch is a subset of data samples from a dataset that are processed together during training or inference. Batching allows for more efficient computation, especially on hardware accelerators like GPUs, by parallelizing operations across multiple data samples.

40. **Confusion matrix**: A confusion matrix is a table used to evaluate the performance of a classification model. It presents a summary of the predicted and actual class labels for a classification problem, showing the number of true positives, true negatives, false positives, and false negatives.

41. **Gradient**: In the context of optimization algorithms, the gradient represents the direction and magnitude of the steepest increase of a function. It points in the direction of the greatest increase in the function's value, and its magnitude indicates the rate of change.

42. **SGD**: SGD stands for Stochastic Gradient Descent, an optimization algorithm commonly used in training machine learning models. It updates the model parameters iteratively by computing gradients on small random subsets of the training data, making it computationally efficient for large datasets.

43. **Tokenization**: Tokenization is the process of splitting text data into smaller units called tokens. Tokens can be words, subwords, characters, or any other meaningful units of text, depending on the task and the tokenizer used.

44. **gradual unfreezing**: Gradual unfreezing is a technique used in transfer learning, where layers of a pre-trained model are unfrozen incrementally during training. This allows the model to adapt to the new task while preserving the learned representations in the early layers.

45. **numericalization**: Numericalization is the process of converting categorical data into numerical form. It's are crucial for stabilizing and speeding up the training of neural networks 
    - *Types*
    1. One-hot Encoding: A preprocessing step that involves converting each categorical variable into a new binary variable for each category.
    2. Batch Normalization: Normalizes the activations of a previous layer by subtracting the batch mean and dividing by the batch standard deviation.

46. **Ensambling**: Ensemble methods are techniques in machine learning that combine multiple models to produce a better performing model than the individual models constituting the ensemble. 
    - Benefits: 
    1. Reduce overfitting: Different models will likely overfit different aspects of the data, and aggregation can help to average out these effects.
    2. Improve generalizatio: Ensambled models often perform better on unseen data compared to single models.

    - *Types*
    1. Bagging: It involves training multiple models independently on different subsets of the data. Each model learns from a slightly different version of the training data. After training, the predictions from each network can be averaged (for regression) or voted on (for classification) to produce the final output. 
    2. Boosting: Here multiple weak learners are combined sequentially to create a strong learner. Each new learner focuses on the mistakes made by the previous ones, gradually improving the overall performance of the model.
    3. Stacking: Different models are trained on the same dataset, and a new model, called a meta-model or blender, is then trained to make a final prediction based on the predictions of the previous models.The base models' predictions are used as input features for the meta-model, which then learns how best to combine these predictions to make the final prediction.

47. **What is a latent factor? Why is it "latent"?**: A latent factor refers to a variable that is not directly observed but is inferred or estimated from other observed variables within a model. These factors are typically used to explain patterns in the data that are not immediately apparent, and they play a crucial role in various types of analysis, including *factor analysis*, *principal component analysis (PCA)*, and *recommendation systems.*

48. **embedding matrix (embeddings)**: The embedding matrix is essentially a lookup table, typically used for transforming high-dimensional categorical data into a lower-dimensional space. Each row of the matrix corresponds to a specific category (such as a word in NLP). When a categorical item needs to be converted into an embedding, the process involves retrieving a row from the matrix. This row is the embedded representation of the categorical item and is learned from data during the training process of a machine learning model.
    - Embeddings vs. One-hot Encoding-
        1. *Efficiency*: Embeddings reduce the dimensionality significantly, which decreases model complexity and improves computational efficiency.
        2. *Semantic Representation*: Embeddings capture more information and relationships between categories (like semantic similarity between words) than one-hot vectors.
        3. *Trainability*: Unlike static one-hot vectors, embeddings are learned and optimized during training, allowing them to adapt to the specific task by capturing relevant patterns in the data.

49. **bootstrapping problem**: The bootstrapping problem refers to the challenge of estimating the reliability of predictions made by a model when the model's training data is limited or noisy. In recommendation systems, it refers to the initial challenge of generating useful recommendations when there is insufficient data available.
    - *Issues*
    1. Cold Start Problem: The bootstrapping problem is often linked to the cold start problem, where the system has little to no information about new users or new items. This lack of data makes it difficult to recommend items that the user might like.
    2. Data Scarcity: Early in the deployment of a recommendation system, there is typically limited interaction data (such as ratings, clicks, views, purchases). This scarcity of data hampers the system's ability to learn and generalize user preferences effectively.
    - *Soutions*
    1. Content-Based Filtering: Uses information about the items (e.g., metadata such as genre, description, etc.) to make recommendations, helping to alleviate the item cold start problem.
    2. Explicit Feedback: Asking new users to provide initial preferences or rate a few items to quickly gather data about their likes and dislikes.
    3. Implicit Feedback: Using indirect signals such as browsing history, click patterns, or time spent on different items to infer user preferences.
    4. Demographic-Based Recommendations: Using demographic information (age, gender, location) to recommend items that are popular among similar demographic groups.

50. **feature**: In machine learning, a feature refers to an individual measurable property or characteristic of a data sample that is used as input to a model for making predictions or classifications. Features can be numerical, categorical, or ordinal, and they capture relevant information about the data samples.

51. **Convolutional Neural Network (CNN)**: A type of neural network particularly suited for analyzing visual data, employing convolutional layers to extract spatial hierarchies of features.

52. **Convolutions**: Convolutions are mathematical operations commonly used in deep learning for tasks such as image processing and computer vision. In the context of convolutional neural networks (CNNs), convolutions involve applying a filter or kernel to an input image or feature map, performing a dot product between the filter and overlapping patches of the input, and producing an output feature map. Convolutions enable the network to extract spatial hierarchies of features, capturing patterns such as edges, textures, and shapes.


53. **channel**: In the context of convolutional neural networks (CNNs), a channel refers to a single feature map produced by applying a convolutional filter to an input image or feature map. Channels capture different aspects or patterns in the input data, such as edges, textures, or colors.

54. **padding**: Padding is a technique used in convolutional neural networks to ensure that the spatial dimensions of input data and output feature maps are preserved after applying convolutions. It involves adding extra pixels or values around the borders of the input data to achieve the desired output size.

55. **stride**: In convolutional neural networks, the stride is the step size used to slide the convolutional filter across the input data or feature map during the convolution operation. It determines the amount of spatial overlap between neighboring patches of the input data and affects the spatial dimensions of the output feature maps.

56. **MNIST CNN**: MNIST CNN refers to a convolutional neural network architecture designed for the task of classifying handwritten digits in the MNIST dataset. It typically consists of multiple convolutional layers followed by pooling layers and fully connected layers, achieving high accuracy on the MNIST benchmark task.


57. Tensors and Tensor Rank
    - Tensors: Tensors are multi-dimensional arrays that generalize scalars, vectors, and matrices. They are a fundamental data structure used in deep learning and numerical computing.
        Dimensions:
            0-D Tensor (Scalar): A single value, e.g., a number like 3 or -1.5.
            1-D Tensor (Vector): A one-dimensional array of values, e.g., [1, 2, 3].
            2-D Tensor (Matrix): A two-dimensional array of values, e.g., a table or grid of numbers.
            n-D Tensor: An n-dimensional array of values, where n can be any positive integer.
    - Tensor Rank: The rank of a tensor is the number of dimensions (axes) it has.
        Dimensions:
            Rank-0 Tensor: Scalar, e.g., 5.
            Rank-1 Tensor: Vector, e.g., [1, 2, 3].
            Rank-2 Tensor: Matrix, e.g., [[1, 2], [3, 4]].
            Rank-n Tensor: An n-dimensional array, e.g., a 3-D tensor might represent a volume in space or a stack of images.
        
- Usage in Deep Learning:
    1. Data Representation: Tensors are used to represent data (inputs, outputs, weights) in neural networks.
    2. Operations: Many operations in deep learning (e.g., matrix multiplication, convolution) involve manipulating tensors.