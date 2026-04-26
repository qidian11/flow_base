import numpy as np

data = np.load('prior_cnn_res_model_double_precision_False_14_loss_history_coupling_layers_32_hidden_layers_6_hidden_channels_16_iterations_30000.npy', mmap_mode='r')
print(data[::100])