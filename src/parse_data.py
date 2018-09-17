import os
import xml.etree.ElementTree
import json
import pickle as pkl


def load_label_mapping(label_num):
    label_mapping = json.load(open(label_num, 'r'))
    label_to_code = {}
    code_to_label = {}
    for key, value in label_mapping.items():
        label_to_code[int(key)] = value[0]
        code_to_label[value[0]] = int(key)

    with open('../resources/label_to_code.pkl', 'wb') as ff:
        pkl.dump(label_to_code, ff)
    with open('../resources/code_to_label.pkl', 'wb') as ff:
        pkl.dump(code_to_label, ff)

    return label_to_code, code_to_label


def main(data_path, label_path, label_num):
    data = {}
    label_to_code, code_to_label = load_label_mapping(label_num)
    
    for file in os.listdir(data_path):
        data[file.split('.')[0]] = [data_path+file, -1]

    xmls = [label_path+file for file in os.listdir(label_path)]
    for it, xml_file in enumerate(xmls):
        e = xml.etree.ElementTree.parse(xml_file).getroot()
        name = e.find('filename').text
        code = e.find('object').find('name').text
        data[name][1] = code_to_label[code]

        if it % 5000 == 0:
            print('{} / {} ({}%)'.format(it, len(xmls), 100*it/len(xmls)))

    with open('../resources/data.pkl', 'wb') as ff:
        pkl.dump(data, ff)


if __name__ == '__main__':

    data_path = '/media/dan/CAGE/ILSVRC/Data/CLS-LOC/val/'
    label_path = '/media/dan/CAGE/ILSVRC/Annotations/CLS-LOC/val/'
    label_num = '/media/dan/CAGE/ILSVRC/labels.json'

    main(data_path, label_path, label_num)