from Image_processing import Sudoku_Image, NN_model, Sudoku

# -----------------------------------------------------------------------------
#Creating image object
ob_image = Sudoku_Image()

# Obtain digit classes, X and y. There are 10 classes, and each digit has 1016 images.
data_classes, data_X, data_y = ob_image.load_data(r"./digits")

# Splitting into train and test datasets, and for cross validation as well
train_X, train_y, test_X, test_y, valid_X, valid_y = ob_image.split_train_test(data_X, data_y, split_size = 0.05, valid_split_size= 0.2)

datagen = ob_image.augmentation(train_X)
train_y, test_y, valid_y = ob_image.one_hot_encoding(train_y, test_y, valid_y, data_classes)

ob_model = NN_model()
model = ob_model.create_model()

model_NN = ob_model.compile_fit_model(datagen, model, train_X, train_y, valid_X, valid_y, epoch=30)

score_test = model_NN.evaluate(test_X, test_y, verbose=0)
print('Test Score = ',score_test[0])
print('Test Accuracy =', score_test[1])

score_train = model_NN.evaluate(train_X, train_y, verbose=0)
print('Train Score = ',score_train[0])
print('Train Accuracy =', score_train[1])

rand_sudoku = ob_image.select_rand_sudoku(r"./sudokus/aug")
ob_image.plot_image(rand_sudoku)

prep_image = ob_image.preprocess(rand_sudoku)
ob_image.plot_image(prep_image)

imgwrap = ob_image.find_outline_in_sudoku(rand_sudoku, prep_image)
sudoku = Sudoku()

# -----------------------------------------------------------------------------
# Read image
puzzle = ob_image.read_image("./sudoku puzzle/9.png")

grid = ob_image.prepare_puzzle(rand_sudoku, model_NN)

grid =[[9, 1, 3, 0, 0, 0, 0, 0, 2],
[0, 6, 0, 4, 9, 0, 1, 0, 3],
[0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 9, 0, 0],
[3, 0, 1, 0, 0, 4, 0, 0, 0],
[0, 8, 0, 7, 0, 2, 4, 3, 0],
[1, 7, 8, 5, 0, 9, 3, 0, 0],
[0, 0, 0, 0, 0, 0, 7, 5, 9],
[0, 0, 0, 3, 6, 7, 0, 0, 0]]

print(grid)

if (sudoku.Suduko(grid, 0, 0)):
    sudoku.printSudoku(grid)
else:
    print("Solution does not exist:(")