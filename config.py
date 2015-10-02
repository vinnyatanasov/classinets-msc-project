# ClassiNet configuration variables and directories live here.

num_targets = 700
train_data_size = 2000
agreement_data_size = 900
feat_space_size = 50000

# Note: ensure the following directories are created before running scripts!
output_dir = "output-700/" # assuming 700 target word ClassiNet
train_data_dir = output_dir + "target-train-data/"
weight_vect_dir = output_dir + "weight-vectors/"
expanded_data_dir = output_dir + "expanded-data/"
