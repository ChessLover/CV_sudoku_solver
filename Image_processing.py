# Sudoku functions package
import numpy as np 
import matplotlib.pyplot as plt
import os, random
import cv2
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical
from PIL import Image


class Sudoku_Image: 
    
    def load_data(self, path):        
        data = os.listdir(path)
        data_X = []     
        data_y = []  
        data_classes = len(data)
        for i in range (0,data_classes):
            data_list = os.listdir(path +"/"+str(i))
            for j in data_list:
                pic = cv2.imread(path +"/"+str(i)+"/"+j)
                pic = cv2.resize(pic,(32,32))
                data_X.append(pic)
                data_y.append(i)
        
        # Labels and images
        data_X = np.array(data_X)
        data_y = np.array(data_y)
        
        return data_classes, data_X, data_y
    
    def split_train_test(self, data_X, data_y, split_size, valid_split_size): 
        
        #Spliting the train validation and test sets
        train_X, test_X, train_y, test_y = train_test_split(data_X,data_y,test_size=split_size)
        train_X, valid_X, train_y, valid_y = train_test_split(train_X,train_y,test_size=valid_split_size)
        print("Training Set Shape = ", np.array(train_X).shape)
        print("Validation Set Shape = ", np.array(valid_X).shape)
        print("Test Set Shape = ", np.array(test_X).shape)

        train_X = np.array(list(map(self.Prep, train_X)))
        test_X = np.array(list(map(self.Prep, test_X)))
        valid_X= np.array(list(map(self.Prep, valid_X)))

        #Reshaping the images
        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2],1)
        test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2],1)
        valid_X = valid_X.reshape(valid_X.shape[0], valid_X.shape[1], valid_X.shape[2],1)
        
        return train_X, train_y, test_X, test_y, valid_X, valid_y
    
    def one_hot_encoding(self, train_y, test_y, valid_y, data_classes):
        
        # One hot encoding of the labels
        train_y = to_categorical(train_y, data_classes)
        test_y = to_categorical(test_y, data_classes)
        valid_y = to_categorical(valid_y, data_classes)
        
        return train_y, test_y, valid_y
    
    
    def augmentation(self, train_X): 
        #Augmentation
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
        datagen.fit(train_X)
        
        return datagen
    
    def select_rand_sudoku(self, path):
        # Randomly select an image from the dataset 
        rand_img=random.choice(os.listdir(path))
        print(rand_img)
        rand_sudoku = cv2.imread(path+'/'+rand_img)
        rand_sudoku = cv2.resize(rand_sudoku, (450,450))
        
        plt.figure()
        plt.imshow(rand_sudoku)
        plt.show() 
        
        return rand_sudoku
    
    def find_outline_in_sudoku(self, sudoku_img, threshold):
        # Finding the outline of the sudoku puzzle in the image
        contour_1 = sudoku_img.copy()
        contour_2 = sudoku_img.copy()
        contour, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_1, contour,-1,(0,255,0),3)

        #Plot the image
        plt.figure()
        plt.imshow(contour_1)
        plt.show()
        
        biggest, maxArea = self.main_outline(contour)
        print("biggest = ", biggest)
        print("maxArea = ", maxArea)
        if biggest.size != 0:
            biggest = self.reframe(biggest)
            cv2.drawContours(contour_2,biggest,-1, (0,255,0),10)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
            matrix = cv2.getPerspectiveTransform(pts1,pts2)  
            imagewrap = cv2.warpPerspective(sudoku_img,matrix,(450,450))
            imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
            
        plt.figure()
        plt.imshow(imagewrap)
        plt.show()
        
        return imagewrap
    
    # Preprocessing the images for neuralnet
    def Prep(self, img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        img = cv2.equalizeHist(img) #Histogram equalization to enhance contrast
        img = img/255 #normalizing
        return img

    # Function to grayscale, blur and apply threshold
    def preprocess(self, sudoku_img):
        gray = cv2.cvtColor(sudoku_img, cv2.COLOR_BGR2GRAY) 
        plt.figure()
        plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
        plt.show() 
        blur = cv2.GaussianBlur(gray, (3,3),6) 
        plt.figure()
        plt.imshow(blur)
        plt.show() 
        threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
        return threshold_img

    def main_outline(self, contour):
        biggest = np.array([])
        max_area = 0
        for i in contour:
            area = cv2.contourArea(i)
            if area >50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i , 0.02* peri, True)
                if area > max_area and len(approx) ==4:
                    biggest = approx
                    max_area = area
        return biggest ,max_area

    def reframe(self, points):
        points = points.reshape((4, 2))
        points_new = np.zeros((4,1,2),dtype = np.int32)
        add = points.sum(1)
        points_new[0] = points[np.argmin(add)]
        points_new[3] = points[np.argmax(add)]
        diff = np.diff(points, axis =1)
        points_new[1] = points[np.argmin(diff)]
        points_new[2] = points[np.argmax(diff)]
        return points_new

    def splitcells(self, sudoku_image):
        rows = np.vsplit(sudoku_image,9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r,9)
            for box in cols:
                boxes.append(box)
        return boxes
    
    # The sudoku_cell's output includes the boundaries this could lead to misclassifications by the model 
    # I am cropping the cells to avoid that
    # sneeking in a bit of PIL lib as cv2 was giving some weird error that i couldn't ward off

    def CropCell(self, cells):
        Cells_croped = []
        for i in cells:
            
            img = np.array(i)
            img = img[4:46, 6:46]
            img = Image.fromarray(img)
            Cells_croped.append(img)
            
        return Cells_croped

    def predict_digit_in_cell(self, cell, model):

        result = []
        for i in cell:
            img = np.asarray(i)
            img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
            img = cv2.resize(img, (32, 32))
            img = img / 255
            img = img.reshape(1, 32, 32, 1)
            
            predictions = model.predict(img)
            classes_x=np.argmax(predictions,axis=1)
            probabilityValue = np.amax(predictions)
            print("Probability: ", probabilityValue)
            if probabilityValue > 0.65:
                result.append(classes_x[0])
            else:
                result.append(0)
                   
        result = np.asarray(result).reshape(9,9)
        return result
    
    
    def plot_image(self, image):
        plt.figure()
        plt.imshow(image)
        plt.show()

    def read_image(self, path):
        # Importing puzzle to be solved
        puzzle = cv2.imread(path)

        #let's see what we got
        plt.figure()
        plt.imshow(puzzle)
        plt.show()

        # Resizing puzzle to be solved
        puzzle = cv2.resize(puzzle, (450,450))
        
        return puzzle
    
    def prepare_puzzle(self, puzzle, model):
        
        # Preprocessing Puzzle 
        su_puzzle = self.preprocess(puzzle)
        su_imagewrap = self.find_outline_in_sudoku(puzzle, su_puzzle)

        sudoku_cell = self.splitcells(su_imagewrap)
        self.plot_image(sudoku_cell[2])

        sudoku_cell_croped= self.CropCell(sudoku_cell)
        self.plot_image(sudoku_cell_croped[2])

        grid = self.predict_digit_in_cell(sudoku_cell_croped, model)

        return grid

# --------------------------------------------------------------------------------------------------    

class NN_model: 
    
    def create_model(self):

        model = Sequential()

        # model.add((Conv2D(60,(5,5),input_shape=(32, 32, 1) ,padding = 'Same' ,activation='relu')))
        # model.add((Conv2D(60, (5,5),padding="same",activation='relu')))
        # model.add(MaxPooling2D(pool_size=(2,2)))

        # model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
        # model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
        # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # model.add(Dropout(0.5))

        # model.add(Flatten())
        # model.add(Dense(500,activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(10, activation='softmax'))
        
        model.add((Conv2D(60,(5,5),input_shape=(32, 32, 1) ,padding = 'valid' ,activation='relu')))
        model.add((Conv2D(60, (5,5),padding="same",activation='sigmoid')))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(500,activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        
        print(model.summary())
        
        return model
      
    def compile_fit_model(self, datagen, model, train_X, train_y, valid_X, valid_y, epoch):
        #Compiling the model
        optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon = 1e-08)
        model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
        
        #Fit the model
        history = model.fit(datagen.flow(train_X, train_y, batch_size=32),
                                      epochs = epoch, validation_data = (valid_X, valid_y),
                                      verbose = 2, steps_per_epoch= 200)
        
        return model
    

# --------------------------------------------------------------------------------------------------
class Sudoku: 
         
    # Solution
    def printSudoku(self, a):
        print('\n')
        print('\n'.join(['  '.join([str(cell) for cell in row]) for row in a]))
        # for i in range(9):
        #     print(a[i])        
     
    def checkNumberInCol(self, grid, col, number):
        for i in range(0, 9):
            if grid[i][col] == number:
                return True
        return False
    
    def checkNumberInRow(self, grid, row, number):
        for i in range(0, 9):
            if grid[row][i] == number:
                return True
        return False
        
    def checkNumberInBox(self, grid, startRow, startCol, number):
        for i in range(startRow,startRow+3):
            for j in range(startCol,startCol+3):
                if grid[i][j] == number:
                    return True
        return False
    
    def isValidPlace(self, grid, row, col, num):
        return not (self.checkNumberInCol(grid, col, num) or self.checkNumberInRow(grid, row, num) or self.checkNumberInBox(grid, row - row%3 , col - col%3, num))
        
        
    def Suduko(self, grid, row, col):
     
        if (row == 8 and col == 9):
            return True
        
        if col == 9:
            row += 1
            col = 0
            
        if grid[row][col] > 0:
            return self.Suduko(grid, row, col + 1)
        
        for num in range(1, 10): 
            if self.isValidPlace(grid, row, col, num):
                grid[row][col] = num
                
                if self.Suduko(grid, row, col + 1):
                    return True
                
            grid[row][col] = 0
        return False
    
    