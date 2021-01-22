import numpy as np

train_data = np.load('../dataset/jimmyli-logdata/train/train_data.npy')
test_data = np.load('../dataset/jimmyli-logdata/test/test_data.npy')

print("train data (number: %d   size: %d)" % (train_data.shape[0], train_data.shape[1]))
print("test data (number: %d   size: %d)" % (test_data.shape[0], test_data.shape[1]))

num = 0
for i in train_data:
    num += 1
    if num == 50:
        break;
    print(i)
