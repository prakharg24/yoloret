import sys
import os

home_directory = sys.argv[1]

input_files = ['data_paths/voc_train_14910.txt', 'data_paths/voc_val_1641.txt', 'data_paths/voc_test_4952.txt', 'data_paths/voc_trainall_16551.txt']
output_files = ['voc_train_14910.txt', 'voc_val_1641.txt', 'voc_test_4952.txt', 'voc_trainall_16551.txt']

for inp, out in zip(input_files, output_files):
    read_data = open(inp, 'r').read().split("\n")[:-1]
    updated_data = []
    for line in read_data:
        ann = line.split(" ")
        ann[0] = os.path.join(home_directory, ann[0])
        new_line = " ".join(ann)
        updated_data.append(new_line)
    open(out, 'w').write("\n".join(updated_data))
