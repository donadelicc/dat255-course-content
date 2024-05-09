DAT255

1. **Deep Learning**: A subset of machine learning that involves artificial neural networks with multiple layers, allowing the model to learn intricate patterns and representations from data.
   
2. **Neural Network**: A computational model inspired by the structure and function of biological neural networks, composed of interconnected nodes or neurons organized in layers.

3. **Activation Function**: A function applied to the output of each neuron in a neural network layer, determining the neuron's output and introducing non-linearity to the model.

4. **Loss Function**: A function that quantifies the difference between predicted and actual values in a machine learning model, serving as a measure of the model's performance.

5. **Gradient Descent**: An optimization algorithm used to minimize the loss function by iteratively adjusting the model's parameters in the direction of the steepest descent of the loss gradient.

6. **Backpropagation**: A technique for calculating the gradients of the loss function with respect to the parameters of the neural network, enabling efficient optimization using gradient descent.

7. **Mini-batch Gradient Descent**: An optimization variant where the gradient descent updates are computed using small subsets (mini-batches) of the training data, balancing computational efficiency and model convergence.

8. **Learning Rate**: A hyperparameter that determines the step size of parameter updates during gradient descent, influencing the convergence speed and stability of the optimization process.

9. **Overfitting**: A phenomenon where a machine learning model performs well on the training data but fails to generalize to unseen data, often caused by excessive complexity or lack of regularization.

10. **Regularization**: Techniques employed to prevent overfitting by imposing constraints or penalties on the model parameters, such as L1 and L2 regularization.

11. **Dropout**: A regularization technique that randomly deactivates a fraction of neurons during training, reducing co-dependency between neurons and improving generalization performance.

12. **Convolutional Neural Network (CNN)**: A type of neural network particularly suited for analyzing visual data, employing convolutional layers to extract spatial hierarchies of features.

14. **Data Augmentation**: A technique for increasing the diversity of training data by applying transformations such as rotation, scaling, and flipping, thereby enhancing the model's robustness and generalization.

15. **Transfer Learning**: A machine learning approach where knowledge gained from training one model on a particular task is transferred or adapted to a related task, often by fine-tuning the pre-trained model.

16. **Fine-tuning**: The process of further training a pre-trained model on new data or tasks, typically by adjusting its parameters while keeping some of the earlier learned representations intact.

17. **fastai**: A deep learning library built on top of PyTorch, providing high-level abstractions and utilities for simplifying the development and training of neural network models.

18. **DataBlock API**: A component of fastai for flexible and customizable data preprocessing and loading, enabling seamless integration of various data sources and formats into the training pipeline.

19. **Learner**: The central object in fastai representing the model, data, optimization process, and training loop, providing an interface for training, evaluation, and inference.

20. **Callback**: A function or object in fastai that can be invoked at specific points during the training process, allowing for custom actions, monitoring, or modification of the training behavior.

21. **Image Classification**: A task in computer vision where the goal is to assign a label or category to an input image based on its visual content, often accomplished using deep learning models.

22. **Natural Language Processing (NLP)**: A field of artificial intelligence focused on enabling computers to understand, interpret, and generate human language, often employing deep learning techniques.

26. **Transformer**: A neural network architecture introduced in the "Attention is All You Need" paper, widely used in NLP tasks for its parallelization capabilities and effectiveness in capturing long-range dependencies.

28. **GPT (Generative Pre-trained Transformer)**: A series of transformer-based models developed by OpenAI for natural language generation tasks, leveraging large-scale unsupervised pre-training followed by fine-tuning on specific tasks.

29. **Computer Vision**: A field of artificial intelligence focused on enabling computers to interpret and analyze visual information from the real world, often employing deep learning techniques for tasks like object detection and image segmentation.
Bruke denne som “Give an example of topic in the course”??

30. **Object Detection**: A computer vision task involving the identification and localization of objects within an image, often achieved by predicting bounding boxes and class labels for each object instance.

31. **Segmentation**: A computer vision task where the goal is to partition an image into multiple segments or regions, typically representing different objects or regions of interest, often used in medical imaging and scene understanding.

34. **Reinforcement Learning**: A machine learning paradigm where an agent learns to make decisions by interacting with an environment, receiving rewards or penalties based on its actions, often used in game playing and robotics.

1. **fine_tune()**: A method or function commonly found in deep learning libraries like fastai, used to further train a pre-trained neural network model on a new dataset or task. Fine-tuning typically involves adjusting the parameters of the pre-trained model while keeping some of the earlier learned representations intact, thereby leveraging the knowledge gained from pre-training to improve performance on the new task.

2. **DataBlock**: In the context of fastai, DataBlock is a high-level abstraction for defining the data processing pipeline in machine learning tasks. It allows users to specify how to transform raw data into a format suitable for training, validation, and testing, including tasks such as image classification, text classification, and tabular data analysis. DataBlock provides flexibility and customization options for preprocessing data and creating DataLoaders.

3. **DataLoaders**: DataLoaders are objects used to efficiently load and iterate over batches of data during the training, validation, and testing phases of a machine learning task. In fastai, DataLoaders are typically created using the DataBlock API, encapsulating the training and validation datasets along with data augmentation, batching, shuffling, and other data processing operations. They provide a convenient interface for feeding data to the training loop.

4. **Embedding**: In the context of deep learning, an embedding refers to a mapping from discrete categorical variables (such as words, tokens, or categories) to continuous vector representations in a lower-dimensional space. Embeddings are learned during the training process and capture semantic relationships between categorical values, enabling neural networks to effectively process and generalize from categorical input data.

5. **Learning Rate**: The learning rate is a hyperparameter that controls the step size of parameter updates during the training process of a neural network. It determines how much the model parameters are adjusted in the direction of the gradient during optimization. A suitable learning rate is crucial for achieving optimal convergence and performance in training neural networks, with values typically chosen through experimentation and validation.

6. **Learning Rate Finder**: A technique used to determine an appropriate learning rate for training neural networks, particularly in the context of stochastic gradient descent optimization algorithms. The learning rate finder involves gradually increasing the learning rate during the early stages of training while monitoring the loss function's behavior. The optimal learning rate is typically identified as the point where the loss begins to decrease most rapidly or levels off.

7. **Metric**: A metric is a measure used to evaluate the performance of a machine learning model on a specific task or dataset. Metrics quantify different aspects of model performance, such as accuracy, precision, recall, F1-score, mean squared error, or area under the receiver operating characteristic curve (ROC-AUC). Metrics are essential for assessing the effectiveness and quality of trained models and guiding model selection and optimization processes.

8. **Tabular Data**: Tabular data refers to structured datasets organized in rows and columns, where each row represents a data sample or observation, and each column represents a feature or attribute. Tabular data commonly arises in various domains such as finance, healthcare, retail, and marketing, and may include numerical, categorical, or ordinal variables. Analyzing tabular data involves tasks such as classification, regression, and clustering, often using machine learning algorithms.

1. **Generative AI**: Generative AI refers to artificial intelligence systems and algorithms that are capable of generating new content, such as images, text, audio, or video, that mimics or resembles human-created data. Generative AI models often learn to capture and replicate the underlying distribution of training data, enabling them to produce novel and realistic outputs. Examples of generative AI models include generative adversarial networks (GANs), variational autoencoders (VAEs), and autoregressive models.

2. **Collaborative Filtering**: Collaborative filtering is a technique used in recommendation systems to predict a user's preferences or interests based on the preferences of similar users or items. It leverages the collective wisdom of a group of users to make personalized recommendations, typically by analyzing user-item interaction data such as ratings, reviews, or purchase history. Collaborative filtering can be implemented using different approaches, including user-based filtering, item-based filtering, and matrix factorization methods.

3. **Convolutions**: Convolutions are mathematical operations commonly used in deep learning for tasks such as image processing and computer vision. In the context of convolutional neural networks (CNNs), convolutions involve applying a filter or kernel to an input image or feature map, performing a dot product between the filter and overlapping patches of the input, and producing an output feature map. Convolutions enable the network to extract spatial hierarchies of features, capturing patterns such as edges, textures, and shapes.

4. **ResNet**: ResNet, short for Residual Network, is a deep neural network architecture proposed by Kaiming He et al. in their 2015 paper "Deep Residual Learning for Image Recognition." ResNet introduced the concept of residual blocks, where the input to a block is added to its output, allowing for easier training of very deep neural networks. ResNet architectures have achieved state-of-the-art performance in various computer vision tasks, including image classification, object detection, and semantic segmentation.

5. **"Freezing" parameters in the fine_tune() method**: When using the fine_tune() method in deep learning frameworks like fastai, "freezing" parameters refers to temporarily disabling the updates of certain layers or parameters during the fine-tuning process. This is often done to preserve the learned representations in the pre-trained layers while allowing the later layers to adapt to the new task or dataset. Freezing parameters helps prevent catastrophic forgetting and can improve the convergence and generalization performance of the fine-tuning process, especially when the new dataset is small or similar to the original pre-training data.

1. **head**: In the context of neural networks, especially when fine-tuning pre-trained models, the "head" refers to the top layers of the network that are responsible for making predictions. These layers are typically replaced or modified when adapting a pre-trained model to a new task or dataset.

2. **Hyperparameters**: Hyperparameters are parameters whose values are set before the training process begins and remain constant during training. Examples include learning rate, batch size, and the number of layers in a neural network. Tuning hyperparameters is a crucial part of optimizing the performance of machine learning models.


`Resize()` is a method commonly used in image processing and computer vision tasks to resize images to a specified size. It is used to standardize the dimensions of images in a dataset, ensuring that all images have the same width and height, which is often necessary for feeding them into neural networks for training or inference. 

In libraries such as Python's PIL (Python Imaging Library) or OpenCV, `Resize()` allows users to specify the desired dimensions (width and height) to which the images should be resized. This resizing process can help normalize the input data and improve the model's performance by reducing computational complexity and ensuring consistency across the dataset.


4. **item_tfms()**: In libraries like fastai, "item_tfms()" refers to transformations applied to individual items or samples in a dataset. These transformations typically include resizing, cropping, normalization, and other preprocessing steps performed on each item before they are used for training or inference.

5. **batch_tfms()**: Similar to "item_tfms()", "batch_tfms()" in fastai refers to transformations applied to batches of data during the training process. These transformations are typically applied after data augmentation and are useful for tasks such as data normalization, data augmentation, and regularization.

6. **batch**: A batch is a subset of data samples from a dataset that are processed together during training or inference. Batching allows for more efficient computation, especially on hardware accelerators like GPUs, by parallelizing operations across multiple data samples.

7. **Confusion matrix**: A confusion matrix is a table used to evaluate the performance of a classification model. It presents a summary of the predicted and actual class labels for a classification problem, showing the number of true positives, true negatives, false positives, and false negatives.

8. **RMSE**: RMSE stands for Root Mean Square Error, a common metric used to evaluate the accuracy of regression models. It measures the square root of the average squared differences between predicted and actual values, providing a measure of the model's prediction error.

9. **L1 norm**: The L1 norm, also known as the Manhattan norm or taxicab norm, is a measure of vector magnitude that represents the sum of the absolute values of the vector components. It is commonly used in regularization techniques such as Lasso regression.

10. **Gradient**: In the context of optimization algorithms, the gradient represents the direction and magnitude of the steepest increase of a function. It points in the direction of the greatest increase in the function's value, and its magnitude indicates the rate of change.

11. **SGD**: SGD stands for Stochastic Gradient Descent, an optimization algorithm commonly used in training machine learning models. It updates the model parameters iteratively by computing gradients on small random subsets of the training data, making it computationally efficient for large datasets.

12. **What is the value of log(-2)**: The natural logarithm of a negative number is undefined in the real number system, including log(-2). However, in the complex number system, log(-2) can be expressed as log(2) + iπ, where i is the imaginary unit and π is the mathematical constant pi.

13. **tokenization**: Tokenization is the process of splitting text data into smaller units called tokens. Tokens can be words, subwords, characters, or any other meaningful units of text, depending on the task and the tokenizer used.

14. **gradual unfreezing**: Gradual unfreezing is a technique used in transfer learning, where layers of a pre-trained model are unfrozen incrementally during training. This allows the model to adapt to the new task while preserving the learned representations in the early layers.

15. **numericalization**: Numericalization is the process of converting categorical data into numerical form, typically by assigning a unique integer index to each category. It is a common preprocessing step in machine learning tasks involving categorical variables.

16. **bagging**: Bagging, short for bootstrap aggregating, is an ensemble learning technique where multiple models are trained on random subsets of the training data with replacement. The final prediction is typically obtained by averaging or voting over the predictions of individual models.

17. **boosting**: Boosting is another ensemble learning technique where multiple weak learners are combined sequentially to create a strong learner. Each new learner focuses on the mistakes made by the previous ones, gradually improving the overall performance of the model.

18. **What is a latent factor? Why is it "latent"?**: In the context of recommendation systems and matrix factorization methods like collaborative filtering, a latent factor is a hidden or unobserved variable that represents an underlying characteristic or feature of users or items. These factors are called "latent" because they are not directly observable but are inferred from the patterns in the data.

19. **embedding matrix**: In natural language processing and other tasks involving categorical variables, an embedding matrix is a learnable matrix used to map discrete tokens or categories to continuous vector representations (embeddings) in a lower-dimensional space. Embedding matrices are trained alongside the model parameters during the training process.

20. **weight decay**: Weight decay is a regularization technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function based on the magnitude of the model's weights. It encourages the model to learn simpler and smoother representations by penalizing large weight values.

21. **bootstrapping problem**: The bootstrapping problem refers to the challenge of estimating the reliability of predictions made by a model when the model's training data is limited or noisy. It can lead to overconfidence in the model's predictions and poor generalization performance.

22. **feature**: In machine learning, a feature refers to an individual measurable property or characteristic of a data sample that is used as input to a model for making predictions or classifications. Features can be numerical, categorical, or ordinal, and they capture relevant information about the data samples.

23. **channel**: In the context of convolutional neural networks (CNNs), a channel refers to a single feature map produced by applying a convolutional filter to an input image or feature map. Channels capture different aspects or patterns in the input data, such as edges, textures, or colors.

24. **padding**: Padding is a technique used in convolutional neural networks to ensure that the spatial dimensions of input data and output feature maps are preserved after applying convolutions. It involves adding extra pixels or values around the borders of the input data to achieve the desired output size.

25. **stride**: In convolutional neural networks, the stride is the step size used to slide the convolutional filter across the input data or feature map during the convolution operation. It determines the amount of spatial overlap between neighboring patches of the input data and affects the spatial dimensions of the output feature maps.

26. **MNIST CNN**: MNIST CNN refers to a convolutional neural network architecture designed for the task of classifying handwritten digits in the MNIST dataset. It typically consists of multiple convolutional layers followed by pooling layers and fully connected layers, achieving high accuracy on the MNIST benchmark task.











Absolutt, her er en enkelt setning for hvert begrep:

1. **Gradient**: En vektor som peker i retningen av den raskeste økningen av en funksjon, brukt i optimeringsalgoritmer for å minimere eller maksimere funksjonsverdier.

2. **Loss Function**: En matematisk funksjon som beregner avstanden mellom prediksjoner gjort av et modell og de faktiske målene, brukt til å måle modellens ytelse under trening.

3. **Activation Function**: En ikke-lineær funksjon som introduseres i hver lag av et nevralt nettverk, som gir modellen muligheten til å lære og representere komplekse sammenhenger i data.

4. **Gradient Descent**: En optimeringsalgoritme som brukes til å minimere en tapfunksjon ved å justere parametrene i modellen i retning av de steepeste nedgangene av tapfunksjonen.

5. **Convolutional Layer**: Et lag i et konvolusjonsnettverk som utfører konvolusjon av inputdata med en sett av filtre for å generere feature-maps.

6. **Filter**: En liten matrise som blir skalert over inputdata i et konvolusjonslag for å utlede feature-maps, som hjelper nettverket med å lære distinkte visuelle mønstre.

7. **Stride**: Avstanden mellom hvor filteret blir flyttet over inputdataen i hvert steg under konvolusjonsoperasjonen.

8. **Segmentation**: Prosessen med å dele et bilde eller et datasett inn i forskjellige segmenter eller regioner basert på visuelle egenskaper, ofte brukt i bildegjenkjenning og medisinsk bildeanalyse.

9. **Tokenization**: Prosessen med å konvertere tekst eller data til mindre enheter, kalt tokens, som kan være ord, tegn eller andre definerte enheter, for videre analyse eller behandling.

10. **Numericalization**: Konvertering av tekst eller kategoriske data til numeriske verdier, som ofte brukes i maskinlæring for å tillate modeller å håndtere slike data.

11. **Embedding**: En representasjon av et objekt eller en enhet, vanligvis som en vektor med lavere dimensjoner, som inneholder meningsfull informasjon om objektets egenskaper eller relasjoner, ofte brukt i naturlig språkbehandling for å representere ord eller setninger.

1. **Boosting**: En ensemble-teknikk i maskinlæring hvor flere svake modeller bygges sekvensielt, hvor hver modell fokuserer på å rette feilene som den forrige modellen gjorde.

2. **Bagging**: En ensemble-teknikk i maskinlæring hvor flere modeller bygges uavhengig av hverandre, ofte ved å trene hver modell på forskjellige delsett av treningsdataene og deretter kombinere prediksjonene.

3. **fine_tune()**: En metode i dyplæring som justerer parametrene til et forhåndstreinert nevralt nettverk ved å trene det videre på en spesifikk oppgave med tilpasning av de siste lagene eller hele nettverket.

4. **Learning Rate**: En hyperparameter som styrer størrelsen på oppdateringene som gjøres på modellparametrene under trening, og dermed påvirker konvergenshastigheten og ytelsen til modellen.

5. **Transfer Learning**: En tilnærming i maskinlæring der kunnskapen som er lært av et forhåndstreinert nettverk på en stor datasett, blir overført eller tilpasset til å forbedre ytelsen på en annen, lignende oppgave med mindre datasett.

6. **DataBlock**: En del av rammeverket fastai som lar deg definere hvordan du skal lese inn og behandle dataene dine før de mate inn i modellen din.

7. **DataLoader**: En del av rammeverket PyTorch som laster dataene dine inn i nettverket ditt i små batcher, som gjør det mulig å trene modellen din effektivt og parallelt.

8. **DataLoaders**: En kombinasjon av DataLoaders fra PyTorch og DataBlock fra fastai, som lar deg enkelt definere og håndtere lastingen av trenings-, validerings- og testdataene dine i maskinlæringsprosjekter.

1. **Tabular Data**: Data organisert i tabellform med rader og kolonner, hvor hver rad representerer en observasjon eller et eksempel, og hver kolonne representerer en funksjon eller et attributt.

2. **Decoder**: En del av en nevralt nettverksarkitektur, spesielt vanlig i sekvensgenerering og maskinoversettelsesmodeller, som tolker den skjulte representasjonen fra encoderen og genererer ønsket output, for eksempel en oversettelse av en setning.

3. **Dot Product**: En matematisk operasjon som beregner produktet av tilsvarende elementer i to vektorer og deretter summerer resultatene, ofte brukt i nevrale nettverk for å måle graden av likhet eller samhørighet mellom to vektorer.

4. **Encoder**: En del av en nevralt nettverksarkitektur som tar inn inputdata og konverterer den til en skjult representasjon, ofte med høyere abstraksjonsnivå, som kan brukes av andre deler av nettverket for å utføre oppgaver som klassifisering eller generering.

5. **Metric**: En metode for å evaluere ytelsen til en modell ved å måle forskjellen mellom dens prediksjoner og de faktiske verdiene på testdataene, vanligvis ved hjelp av mål som nøyaktighet, presisjon, recall, eller F1-score, avhengig av den spesifikke oppgaven.

Selvfølgelig, her er 10 flere dyplæringsbegreper med enkle forklaringer:

11. **Convolutional Neural Network (CNN)**: En type nevralt nettverk spesielt egnet for bildebehandling, som bruker konvolusjonslag til å lære distinkte visuelle trekk fra dataene.

12. **Recurrent Neural Network (RNN)**: En type nevralt nettverk som er spesielt egnet for sekvensdata, da det har en tilbakevendende struktur som tillater minnet om tidligere informasjon i analysen av nye data.

13. **Generative Adversarial Networks (GANs)**: Et rammeverk innen dyplæring der to nettverk, en generator og en diskriminator, konkurrerer med hverandre for å forbedre genereringen av realistiske data fra en gitt distribusjon.

14. **Long Short-Term Memory (LSTM)**: En type rekurrent nevralt nettverk, spesielt utviklet for å håndtere problemet med gradientøkning og -fall i vanlige RNN-arkitekturer, ved å introdusere spesielle minneenheter som kan lagre og hente informasjon over lange tidsperioder.

15. **Batch Normalization**: En teknikk som normaliserer aktivisjonene i et nevralt nettverk lagvis under trening for å stabilisere læringsprosessen og øke konvergensen.

16. **Transfer Learning**: En tilnærming innen dyplæring der kunnskapen som er lært av et nettverk på en oppgave, blir overført eller gjenbrukt til å forbedre ytelsen på en annen, lignende oppgave.

17. **Attention Mechanism**: En teknikk som tillater nevrale nettverk å fokusere på spesifikke deler av inputdataen, vanligvis brukt i sekvensmodeller for å gi vekt til forskjellige deler av sekvensen.

18. **Encoder-Decoder Architecture**: En arkitektur i dyplæring som består av to hovedkomponenter: en encoder som konverterer inputdataen til en skjult representasjon, og en decoder som oversetter denne representasjonen til ønsket output.

19. **Mean Squared Error (MSE)**: En vanlig tapfunksjon som beregner gjennomsnittet av kvadratene til forskjellen mellom prediksjonene til modellen og de faktiske målene.

20. **Cross-Entropy Loss**: En tapfunksjon som brukes spesielt i klassifiseringsproblemer, som måler avstanden mellom prediksjoner og faktiske mål som sannsynlighetsfordelinger.
