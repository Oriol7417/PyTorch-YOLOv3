import os
import argparse
import numpy as np
import random


def get_file_list(path, extension):
    file_list = os.listdir(path)
    file_list_extension = [os.path.join(path, file).replace("\\","/") for file in file_list if file.endswith(extension)]
    return file_list_extension

def save_list(file_path, data):
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(data))
        print('Save', file_path)

def get_num_lines(file_path):
    lines = 0
    with open(file_path, 'r') as fp:
        lines = len(fp.readlines())
    return lines

def train_valid_split(X, valid_size=0.2, random_state=1004):   
    valid_num = int(len(X) * valid_size)
    train_num = len(X) - valid_num

    random.seed(random_state)
    random.shuffle(X)
    X_train = X[:train_num]
    X_valid = X[train_num:]
    X_train.sort()
    X_valid.sort()
    return X_train, X_valid

def edit_template(template_file_path, save_file_path, num_classes, num_filters):
    # Read template file
    with open(template_file_path) as fp:
        lines = fp.readlines()

    # Save modified file
    with open(save_file_path, 'w') as fp:
        for line in lines:
            fp.write(line.replace('NUM_FILTERS', num_filters).replace('NUM_CLASSES', num_classes))
        print('Save', save_file_path)

def run():
    parser = argparse.ArgumentParser(description="Generate custom dataset.")
    parser.add_argument("-n", "--name", type=str, default="mask", help="Set the name of the custom dataset.")
    parser.add_argument("-c", "--classes", type=str, default="data/mask/mask.names", help="Path to classes label file (.names)")
    parser.add_argument("-v", "--valid", type=float, default=0.3, help="Valid Ratio (0.3)")
    args = parser.parse_args()

    # Generate train.txt and valid.txt
    dataset_path = os.path.join('data', args.name).replace("\\","/")
    image_path = os.path.join(dataset_path, 'images').replace("\\","/")
    image_list = get_file_list(image_path, '.jpg')
    image_list_train, image_list_valid = train_valid_split(image_list, valid_size=args.valid)
    train_txt_file_path = os.path.join(dataset_path, 'train.txt').replace("\\","/")
    valid_txt_file_path = os.path.join(dataset_path, 'valid.txt').replace("\\","/")
    save_list(train_txt_file_path, image_list_train)
    save_list(valid_txt_file_path, image_list_valid)

    # Generate backup directory
    backup_path = os.path.join(dataset_path, "backup").replace("\\","/")
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)

    # Get number of classes and number of filters
    num_classes = get_num_lines(args.classes)
    num_filters = 3 * (num_classes + 5)

    # Generate yolov3-*.cfg
    config_path = os.path.join('config', args.name).replace("\\","/")
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    saved_file_path = os.path.join(config_path, 'yolov3-' + args.name + '.cfg').replace("\\","/")
    edit_template(os.path.join('config', 'yolov3-template.cfg').replace("\\","/"),  
                  saved_file_path, str(num_classes), str(num_filters))

    # Generate *.data
    data_file_path = os.path.join(config_path, args.name + '.data').replace("\\","/")
    with open(data_file_path, 'w') as fp:
        fp.write('classes='+str(num_classes)+'\n')
        fp.write('train='+train_txt_file_path+'\n')
        fp.write('valid='+valid_txt_file_path+'\n')
        fp.write('names='+args.classes+'\n')
        fp.write('backup='+backup_path+'\n')
        print('Save', data_file_path)
        
if __name__ == "__main__":
    run()