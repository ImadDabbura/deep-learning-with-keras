import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def plot_loss_and_metric(history, metric_name='accuracy'):
    '''Plot training and validation loss and metric on two grids.'''
    acc = history.history[metric_name]
    loss = history.history['loss']
    val_acc = history.history['val_' + metric_name]
    val_loss = history.history['val_loss']
    epochs = len(acc)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(range(1, epochs + 1), loss, label='Training loss')
    axes[0].plot(range(1, epochs + 1), val_loss, label='Validation loss')
    axes[0].set_xlabel('Iteration', fontsize=18)
    axes[0].set_ylabel('Loss', fontsize=18)
    axes[0].set_title('Training and validation loss', fontsize=20)
    axes[0].legend(fontsize=14)
    axes[1].plot(range(1, epochs + 1), acc, label='Training accuracy')
    axes[1].plot(range(1, epochs + 1), val_acc, label='Validation accuracy')
    axes[1].set_xlabel('Iteration', fontsize=18)
    axes[1].set_ylabel('Accuracy', fontsize=18)
    axes[1].set_title('Training and validation accuracy', fontsize=20)
    axes[1].legend(fontsize=14)
    plt.tight_layout()


def feature_extraction(directory, conv_base, num_examples, batch_size=20):
    '''Compute extracted features using `conv_base` of pretrained model.'''
    data_gen = ImageDataGenerator(rescale=1/255)
    generator = data_gen.flow_from_directory(directory,
                                             target_size=(150, 150),
                                             batch_size=batch_size,
                                             class_mode='binary')
    features_extracted = np.zeros((num_examples, 4, 4, 512))
    labels = np.zeros((num_examples))

    i = 0
    for batch_input, batch_label in generator:
        features = conv_base.predict(batch_input)
        features_extracted[i * batch_size:(i + 1) * batch_size] = features
        labels[i * batch_size:(i + 1) * batch_size] = batch_label
        i += 1
        if i * batch_size >= num_examples:
            break

    return features_extracted, labels


def plot_conv_outputs(model, activations):
    # These are the names of the layers
    layer_names = [layer.name for layer in model.layers[:8]]
    # Since the filters are multiple of 16 --> Use 16 images per row
    images_per_row = 16

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # Get num of features, size, & num of columns (1,size,size,features)
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row

        # Initialize display grid
        grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_img = layer_activation[0,
                                               :, :,
                                               col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_img -= channel_img.mean()
                channel_img /= channel_img.std()
                channel_img *= 64
                channel_img += 128
                channel_img = np.clip(channel_img, 0, 255).astype('uint8')
                grid[col * size: (col + 1) * size,
                     row * size: (row + 1) * size] = channel_img

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * grid.shape[1],
                            scale * grid.shape[0]))
        plt.title(layer_name)
        plt.axis('off')
        plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.show()


def smooth_curve(points, factor=0.9):
    '''Add smoothness to set of points.'''
    smooth_points = []
    
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(factor * previous + (1 - factor) * point)
        
        else:
            smooth_points.append(point)
            
    return smooth_points
