import csv
from xml.dom import minidom
from bs4 import BeautifulSoup as bs

TRAINING_DATA_PATH = 'Dataset\\Training\\'
TEST_DATA_PATH = 'Dataset\\Test\\'
OUTPUT_DATA_PATH = 'Dataset\\'

INPUT_DATA_EXTENSION = '.xml'

OUTPUT_DATA_EXTENSION = '.csv'
OUTPUT_DATA_FILENAME = ['training' , 'test']
OUTPUT_DATA_FIELDS = ['text' , 'class']

TARGET_CLASSES = ['3d_Printer' , 'Anime', 'Arduino', 'Astronomy']\
#                  'Biology' , 'Chess', 'Coffee', 'Cooking',
#                 'Law' , 'Space', 'Windows_Phone', 'Wood_Working' ]

MAX_NUMBER = 400

def parse_xml(data_path, target_class):

    parsed_documents = list()
    count = 1

    document = minidom.parse(data_path + target_class + INPUT_DATA_EXTENSION)
    document_items = document.getElementsByTagName('row')

    for item in document_items:
        count += 1
        parser = bs(item.attributes['Body'].value, 'lxml')
        text = parser.get_text()
        parsed_documents.append([text, target_class])
        if count >= MAX_NUMBER:
            break
    return parsed_documents


def write_csv(csv_writer, dataset):
    writer = csv.DictWriter(csv_writer, fieldnames=OUTPUT_DATA_FIELDS)
    writer.writeheader()

    for data in dataset:
        writer.writerow({OUTPUT_DATA_FIELDS[0]:data[0],
                        OUTPUT_DATA_FIELDS[1]: data[1] })
        #print(data)


def main():
    training_data = list()
    for target_class in TARGET_CLASSES:
        training_data += parse_xml(TRAINING_DATA_PATH, target_class)

    test_data = list()
    for target_class in TARGET_CLASSES:
        test_data += parse_xml(TEST_DATA_PATH, target_class)

    OUTPUT_DATA_FILES = [training_data , test_data]

    for file, filename in zip(OUTPUT_DATA_FILES, OUTPUT_DATA_FILENAME):
        csv_writer = open(OUTPUT_DATA_PATH + filename + OUTPUT_DATA_EXTENSION , 'w' , newline='', encoding='utf-8')
        write_csv(csv_writer, file)
        csv_writer.close()

if __name__ == '__main__':
    main()
