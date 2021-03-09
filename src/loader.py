import numpy as np
import cv2
import os
import glob

class Dataloader(object):
    def __init__(self, data_path, batch_size):
        # self parameters
        self.batch_size = batch_size
        self.x_train_dir = os.path.join(data_path, "x_train")        
        self.x_test_dir = os.path.join(data_path, "x_test")
        self.y_data_dir = os.path.join(data_path, "y_data")
        self.params_dir = os.path.join(data_path, "params")
        self.train_list = glob.glob(os.path.join(self.x_train_dir, "*.exr"))
        self.test_list = glob.glob(os.path.join(self.x_test_dir, "*.exr"))
        self.train_samples = len(self.train_list)
        self.test_samples = len(self.test_list)
        self.train_order = np.random.permutation(self.train_samples)
        self.test_order = np.random.permutation(self.test_samples)

    def fresh_batch_order(self):
        self.train_order = np.random.permutation(self.train_samples) 

    # function to get file_lists
    def get_filelist(self, file_list, begin, end, order):
        y_params_list = []
        y_data_list = []
        files = []
        for i in range(begin, end):
            # get exr file name
            exr_name = file_list[order[i]]
            files.append(exr_name)

            # get y_params_name and y_data_name from exr file name
            origin_name, file_name = get_file_name(exr_name)
            y_params_file = os.path.join(self.params_dir, "weight") + "/" + file_name + ".vismap"
            y_params_list.append(y_params_file)
            y_data_file = self.y_data_dir + "/" + origin_name + ".convdata"
            y_data_list.append(y_data_file)

        return files, y_params_list, y_data_list    

    # function to load exr files
    def load_exr(self, files):
        images = []
        for image in files:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            images.append(img[:, :, (0, 1)])
        return images

    # funciton to load groud truth
    def load_gt(self, y_params_list, y_data_list):
        decoder_output = []
        visible_weight = []

        for i in range(len(y_params_list)):
            decoder_data = np.zeros((32, 32, 500), dtype=np.float64)
            y_params = np.load(y_params_list[i])
            weight = np.where(y_params > 0, 10, 0.1)
            y_data = np.load(y_data_list[i]).reshape(100, 4, 32, 32)
            y_pos = y_data[:, 0:3, :, :].reshape(300, 32, 32)
            y_curv = y_data[:, -1, :, :]
            decoder_data = np.moveaxis(np.concatenate((weight, y_pos, y_curv)), 0, -1)
            decoder_output.append(decoder_data)
            visible_weight.append(np.moveaxis(y_params, 0, -1))

        return visible_weight, decoder_output

    # function to get train batches
    def get_train_batch(self, i):
        file_list = self.train_list
        n = self.batch_size        
        begin, end = i * n, (i + 1) * n

        if end >= self.train_samples:
            end = self.train_samples
        n = end-begin

        files, y_params_list, y_data_list = self.get_filelist(file_list, begin, end, self.train_order)

        # put all the data into big arrays
        encoder_input = np.zeros(shape=(n, 256, 256, 2), dtype=np.float64)
        decoder_output = np.zeros(shape=(n, 32, 32, 500), dtype=np.float64)

        x_train = self.load_exr(files)        
        _, y_train = self.load_gt(y_params_list, y_data_list)

        for i in range(n):
            encoder_input[i] = x_train[i]
            decoder_output[i] = y_train[i]

        return encoder_input, decoder_output

    # function to get test batches
    def get_test_data(self, i):
        file_list = self.test_list
        n = self.batch_size        
        begin, end = i * n, (i + 1) * n
        if end >= self.test_samples:
            end = self.test_samples
        n = end-begin

        files, y_params_list, y_data_list = self.get_filelist(file_list, begin, end, self.test_order)

        # put all the data into big arrays
        encoder_input = np.zeros(shape=(n, 256, 256, 2), dtype=np.float64)
        decoder_output = np.zeros(shape=(n, 32, 32, 500), dtype=np.float64)
        visible_weight = np.zeros(shape = (n, 32, 32, 100), dtype=np.float64)

        x_test = self.load_exr(files)
        vis_weight, y_test = self.load_gt(y_params_list, y_data_list)
        
        for i in range(n):
            encoder_input[i] = x_test[i]
            decoder_output[i] = y_test[i]
            visible_weight[i] = vis_weight[i]        
        
        return visible_weight, encoder_input, decoder_output        

def get_file_name(filepath):
    file_name = os.path.split(filepath)[1].split('.')[0]
    origin_name = file_name[0:file_name.rfind("_")]
    return origin_name, file_name