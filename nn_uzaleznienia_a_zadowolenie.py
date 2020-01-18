import numpy as np
#from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

input_file = 'ankieta.csv'
data = np.loadtxt(input_file, dtype='U25, U25, U32, U32, i4', delimiter=';')
np.random.shuffle(data)

sex_labels = ['Kobieta', 'Mężczyzna']
simple_labels = ['Nie', 'Tak']
multiple_labels = ['Nigdy', 'Rzadziej, niż kilka razy w roku', 'Kilka razy w roku', 'Około raz w miesiącu', 'Kilka razy w miesiącu', 'Kilka razy w tygodniu']

#papierosy, alkohol, slodycze, radosc 

sex, cigarettes, alcohol, sweets, happiness = zip(*data)

sex_encoder = preprocessing.LabelEncoder();
simple_encoder = preprocessing.LabelEncoder();
multiple_encoder = preprocessing.LabelEncoder();

sex_encoder.fit(sex_labels)
simple_encoder.fit(simple_labels)
multiple_encoder.fit(multiple_labels)

encoded_sex = sex_encoder.transform(sex)
encoded_cigarettes = simple_encoder.transform(cigarettes)
encoded_alcohol = multiple_encoder.transform(alcohol)
encoded_sweets = multiple_encoder.transform(sweets)

print('\nLable mapping sex:')
for i, item in enumerate(sex_encoder.classes_):
    print(item, '-->', i)

print('\nLable mapping simple:')
for i, item in enumerate(simple_encoder.classes_):
    print(item, '-->', i)

print('\nLable mapping multiple:')
for i, item in enumerate(multiple_encoder.classes_):
    print(item, '-->', i)

X = np.column_stack([encoded_sex, encoded_cigarettes,encoded_alcohol,encoded_sweets])
print(list(X))

num_training = int(0.95*len(happiness))
happiness_array = np.asarray(happiness)

X_tran, y_tran = X[:num_training], happiness_array[:num_training]
X_test, y_test = X[num_training:], happiness_array[num_training:]

model = keras.Sequential()

model.add(keras.layers.Dense(10, activation='relu', input_shape=(4,)))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(14, activation='relu'))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(11, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_tran, y_tran, epochs=2000)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy}, and loss: {test_loss}')
prediction = model.predict(X_test)
for i in range(len(y_test)):
    print(f"Powinno wyjsc: {y_test[i]}, a bylo: {np.argmax(prediction[i])}")

