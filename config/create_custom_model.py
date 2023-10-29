import os
import argparse


def get_file_list(path, extension):
    file_list = os.listdir(path)
    file_list_extension = [os.path.join(path, file) for file in file_list if file.endswith(extension)]
    return file_list_extension

def save_list(file_path, data):
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(data))
        print('Save', file_path)

def create_train_txt(dataset_path):
    image_path = os.path.join(dataset_path, 'images')
    image_list = get_file_list(image_path, '.jpg')
    txt_list = get_file_list(image_path, '.txt')
    assert len(image_list) == len(txt_list), 'The number of image files and the number of text files are different in training dataset.'
    save_list(os.path.join(dataset_path, 'train.txt'), image_list)
    save_list(os.path.join(dataset_path, 'valid.txt'), txt_list)

def get_num_lines(file_path):
    lines = 0
    with open(file_path, 'r') as fp:
        lines = len(fp.readlines())
    return lines

def edit_template(template_file_path, save_file_path, num_classes, num_filters):
    # Read template file
    with open(template_file_path) as fp:
        lines = fp.readlines()

    # Save modified file
    with open(save_file_path, 'w') as fp:
        for line in lines:
            fp.write(line.replace('NUM_FILTERS', num_filters).replace('NUM_CLASSES', num_classes))
        print('Save', save_file_path)

def create_config(name, classes_file_path):
    num_classes = get_num_lines(classes_file_path)
    num_filters = 3 * (num_classes + 5)
    saved_file_path = os.path.join('config', 'yolov3-' + name + '.cfg')
    edit_template(os.path.join('config', 'yolov3-template.cfg'),  
                  saved_file_path, str(num_classes), str(num_filters))

def run():
    parser = argparse.ArgumentParser(description="Generate custom dataset.")
    parser.add_argument("-n", "--name", type=str, default="mask", help="Set the name of the custom dataset.")
    parser.add_argument("-c", "--classes", type=str, default="data/mask/mask.names", help="Path to classes label file (.names)")
    args = parser.parse_args()

    create_train_txt(os.path.join('data', args.name))
    create_config(args.name, args.classes)

if __name__ == "__main__":
    run()