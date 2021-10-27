import sys
import os

home_directory = sys.argv[1]

input_files = ['data_paths/coco_train_105517.txt', 'data_paths/coco_val_11749.txt', 'data_paths/coco_test_4952.txt', 'data_paths/coco_trainall_117266.txt']
output_files = ['coco_train_105517.txt', 'coco_val_11749.txt', 'coco_test_4952.txt', 'coco_trainall_117266.txt']

for inp, out in zip(input_files, output_files):
    read_data = open(inp, 'r').read().split("\n")[:-1]
    updated_data = []
    for line in read_data:
        ann = line.split(" ")
        ann[0] = os.path.join(home_directory, ann[0])
        new_line = " ".join(ann)
        updated_data.append(new_line)
    open(out, 'w').write("\n".join(updated_data))
