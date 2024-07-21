#if you wanna train it with your own image dataset u can do these first two codes otherwise if u have your own data skip first 2
#test data prep
import cv2
import os
from PIL import Image
import csv
import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt
class img2csv1:
    def start():
        x=0
        b=os.listdir()
        for filename in os.listdir():
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".svg") or filename.endswith(".jfif"):
                im = img=cv2.imread(str(filename))
                image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                x+=1
                cv2.imwrite(f"gray\\test_gray{x}.jpg",image_gray)
                with Image.open(f'gray\\test_gray{x}.jpg') as image2:
                    width, height = image2.size
                    print('width -->',str(width))
                    img2=cv2.imread(str(f'gray\\test_gray{x}.jpg'))
                    cv2.imshow("test",img)
                    cv2.imshow("test1",image_gray)
                    cv2.waitKey(900)
                    cv2.destroyAllWindows()
                    print("----------------------")
                    print(image_gray)
                    arr2 = image_gray
                    print(arr2)
                    print(len(arr2))
                    arr2.tofile(f'CSV\\testdata{x}.csv', sep=",")
    def conscsv(input_dir, output_file):
        header_written = False

        with open(output_file, 'w', newline='') as outfile:
            csv_writer = csv.writer(outfile)

            with open(output_file, 'w', newline='') as outfile:
                csv_writer = csv.writer(outfile)

                for root, _, files in os.walk(input_dir):
                    for file_name in files:
                        if file_name.endswith('.csv'):
                            file_path = os.path.join(root, file_name)
                            with open(file_path, 'r', newline='') as infile:
                                csv_reader = csv.reader(infile)
                                header = next(csv_reader, [])
                                csv_writer.writerow(header)
                                try:
                                    for row in csv_reader:
                                        targets[int(row[0])] = 0.99  # Assuming row[0] is the key
                                except IndexError:
                                    print(f"error: {row} has less than one element.")
                                except KeyError:
                                    print(f"error: {row[0]} is not a valid key.")

        print("CSV files success.")

img2csv1.start()
input_directory = 'embedpath/facetrain/CSV/'
output_file_path = 'embedpath/CSV/merged_testdata.csv'
img2csv1.conscsv(input_directory, output_file_path)
#-------------------------------------------------------------------------------------------------------
#train data prep
class img2csv2:
    def start():
        x=0
        b=os.listdir()
        for filename in os.listdir():
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".svg") or filename.endswith(".jfif"):
                im = img=cv2.imread(str(filename))
                image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                x+=1
                cv2.imwrite(f"gray\\train_gray{x}.jpg",image_gray)
                with Image.open(f'gray\\train_gray{x}.jpg') as image2:
                    width, height = image2.size
                    print('width -->',str(width))
                    img2=cv2.imread(str(f'gray\\train_gray{x}.jpg'))
                    cv2.imshow("train",img)
                    cv2.imshow("train1",image_gray)
                    cv2.waitKey(900)
                    cv2.destroyAllWindows()
                    print("----------------------")
                    print(image_gray)
                    arr2 = image_gray
                    print(arr2)
                    print(len(arr2))
                    arr2.tofile(f'CSV\\traindata{x}.csv', sep=",")
    def conscsv(input_dir, output_file):
        header_written = False

        with open(output_file, 'w', newline='') as outfile:
            csv_writer = csv.writer(outfile)

            with open(output_file, 'w', newline='') as outfile:
                csv_writer = csv.writer(outfile)

                for root, _, files in os.walk(input_dir):
                    for file_name in files:
                        if file_name.endswith('.csv'):
                            file_path = os.path.join(root, file_name)
                            with open(file_path, 'r', newline='') as infile:
                                csv_reader = csv.reader(infile)
                                header = next(csv_reader, [])
                                csv_writer.writerow(header)
                                try:
                                    for row in csv_reader:
                                        targets[int(row[0])] = 0.99  # Assuming row[0] is the key
                                except IndexError:
                                    print(f"error: {row} has less than one element.")
                                except KeyError:
                                    print(f"error: {row[0]} is not a valid key.")

        print("CSV files 1 success.")

img2csv2.start()
input_directory = 'embedpath/CSV/'
output_file_path = 'embedpath/CSV/merged_traindata.csv'
img2csv2.conscsv(input_directory, output_file_path)
#---------------------------------------------------------------------------------------------------------------------
#ann self
class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate
        
        self.activation_function = lambda x : expit(x)
        self.inverse_activation_function = lambda x : logit(x)

        self.w_i_h = np.random.default_rng().normal(0, pow(self.input_nodes, -0.5),
                                                    (self.hidden_nodes, self.input_nodes))
        self.w_h_o = np.random.default_rng().normal(0, pow(self.hidden_nodes, -0.5),
                                                    (self.output_nodes, self.hidden_nodes))
        pass


    def train(self, input_list, targets_list):
        inputs = np.array(input_list, ndmin=2).T

        x_hidden = np.dot(self.w_i_h, inputs)
        o_hidden = self.activation_function(x_hidden)

        x_output = np.dot(self.w_h_o, o_hidden)
        o_output = self.activation_function(x_output)

        targets = np.array(targets_list, ndmin=2).T
        output_errors = targets - o_output
        hidden_errors = np.dot(self.w_h_o.T, output_errors)

        self.w_h_o += self.learning_rate * np.dot((output_errors * o_output * (1-o_output)), o_hidden.T)
        self.w_i_h += self.learning_rate * np.dot((hidden_errors * o_hidden * (1-o_hidden)), inputs.T)
       

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        x_hidden = np.dot(self.w_i_h, inputs)
        o_hidden = self.activation_function(x_hidden)

        x_output = np.dot(self.w_h_o, o_hidden)
        o_output = self.activation_function(x_output)

        return o_output


    def reversequery(self, targets_list):
        o_output = np.array(targets_list, ndmin=2).T
        x_output = self.inverse_activation_function(o_output)
        o_hidden = np.dot(self.w_h_o.T, x_output)
        o_hidden -= np.min(o_hidden)
        o_hidden /= np.max(o_hidden)
        o_hidden *= 0.98
        o_hidden += 0.01

        x_hidden = self.inverse_activation_function(o_hidden)
        inputs = np.dot(self.w_i_h.T, x_hidden)
        inputs -= np.min(o_hidden)
        inputs /= np.max(o_hidden)
        inputs *= 0.98
        inputs += 0.01

        return inputs

#---------------------------------------------------------------------------------------------------------------------
#data prep
train_file= open('path_to_your_file\\your-csvfile.csv', 'r')
train_list = train_file.readlines()
train_file.close()

test_file= open('path_to_your_file\\your-csvfile.csv', 'r')
test_list = test_file.readlines()
test_file.close()

#---------------------------------------------------------------------------------------------------------------------
train_list
train_file
test_list
test_file
#---------------------------------------------------------------------------------------------------------------------
#training

# input_nodes = res of pics(example[28*28=784]), or the number of object in each row in dataset
# output nodes = number of test rows you have (10 pics)
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1
epochs = 50

nn = NeuralNetwork(input_nodes=input_nodes,
                   hidden_nodes=hidden_nodes, output_nodes= output_nodes, 
                   learning_rate=learning_rate)
# print("initial weights (W_input_hidden): ", nn.w_i_h)

for e in range (epochs):
    for row in train_list:
        row_data = row.split(',')
        inputs = (np.asfarray(row_data[1:]) / (255.0 * 0.98)) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(row_data[0])] = 0.99
        nn.train(inputs, targets)
    # print("\n\nweights (W_input_hidden) after a round of training: ", nn.w_i_h)
#---------------------------------------------------------------------------------------------------------------------
#testing
test_row_data = test_list[0].split(',')
print("Target number is: ", test_row_data[0])
image_data = np.asfarray(test_row_data[1:]).reshape((28,28))
image = plt.imshow(image_data, cmap='Greys')

nn.query((np.asfarray(test_row_data[1:]) / (255.0 * 0.98)) + 0.01)
