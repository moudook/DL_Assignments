Dataset: MNIST subset, 5 classes, 28x28 images, flattened to 784-dim vectors
Train/val/test already separated (same as Assignment-3)
Same 5 classes as Assignment-3 (0, 4, 5, 6, 7)

Task-1: PCA dimension reduction
- Reduce dimensions to 32, 64, 128, 256
- Eigen vectors from training data
- Mean subtracted training, validation, test data projected onto eigen vectors
- Use mean vector from training set for validation and test mean subtraction
- FCNN classification model for each reduced dimension
- Present validation accuracy for different FCNN architectures
- Present test accuracy and confusion matrix for best architecture (based on validation)
- Observe best reduced dimension from test accuracies
- Compare with best results from Assignment-3

Task-2: Autoencoder reconstruction
- Build autoencoders with 1-hidden layer and 3-hidden layer architectures
- 3-hidden layer: 400 neurons in first and third layers
- Bottleneck (middle) layer: 32, 64, 128, 256 neurons
- Train with Adam optimizer
- Bottleneck layer is always linear (no activation)
- Sigmoid (logistic or tanh, use one consistently) for remaining hidden layers
- Observe average reconstruction errors for train, val, test (computed after training)
- Take one image per class from train, val, test; show reconstructed images for each architecture (with originals)

Task-3: Classification using 1-hidden autoencoder compressed representation
- Save output of middle layer (compressed) from each 1-hidden autoencoder encoder
- Obtain compressed representation for train, val, test
- Experiment for each reduced dimension representation
- FCNN classification model for each representation
- Same FCNN architectures as Task-1
- Present validation accuracy for different architectures
- Present test accuracy and confusion matrix for best architecture
- Observe best reduced dimension from test accuracies
- Compare with Assignment-3 and Task-1 results

Task-4: Classification using 2-hidden autoencoder compressed representation
- Save output of middle layer from 2-hidden autoencoder encoder
- Obtain compressed representation for train, val, test
- Experiment for each reduced representation
- FCNN classification model for each representation
- Same FCNN architectures as Task-1
- Present validation accuracy for different architectures
- Present test accuracy and confusion matrix for best architecture
- Observe best reduced dimension from test accuracies
- Compare with Assignment-3, Task-1, Task-3 results

Task-5: Denoising autoencoders
- 1-hidden layer autoencoder with 20% noise and 40% noise
- Bottleneck neurons based on best test accuracy of best reduced dimension from 1-hidden autoencoder
- Observe average reconstruction errors for train, val, test
- Take one image per class from train, val, test; show reconstructed images (with originals)
- Reduced dimension representation classification
- Same best architecture as Task-3 for this reduced representation
- Present validation and test accuracy

Task-6: Weight visualization
- Best compressed representation from 1-hidden autoencoder: plot inputs that maximally activate each hidden neuron (weights from input to compressed layer)
- Same for both denoising autoencoders
- Compare (a) and (b)

Code/Submission:
- Code in .py file
- Folder name: Group11_Assignment4_code
- Zip file: Group11_Assignment4_code.zip
- Report: Group11_Assignment4_report.pdf
- Upload code zip and report PDF to Moodle

Hidden constraints (from Assignment-3 code and context):
- Reproducible weight initialization (seed 42)
- Same initial weights for different optimizers/architectures
- Xavier Glorot Uniform weight initialization
- Full-batch evaluation for test set
- Checkpoint and resume support for long training runs
- Early stopping based on loss convergence (tol=1e-4)
- Adam optimizer for autoencoder training (specified in assignment)
- Bottleneck layer always linear (no activation)
- Sigmoid OR tanh consistently (not mixed)
- Reconstruction error computed post-training, not during
- One image per class from train/val/test for reconstruction visuals