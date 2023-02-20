import tensorflow as tf
import numpy as np
import pandas as pd
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])

dataset = tf.data.Dataset.from_tensor_slices(data)
batch_size = 2
dataset = dataset.batch(batch_size)

print('Reading:')
for batch in dataset:
    print(batch)
updated_data = data.copy()
updated_data[1, 1] = 0
updated_dataset = tf.data.Dataset.from_tensor_slices(updated_data)

print('Updating:')
for batch in updated_dataset.batch(batch_size):
    print(batch)

filtered_data = np.delete(data, 2, axis=0)
filtered_dataset = tf.data.Dataset.from_tensor_slices(filtered_data)

print('Delete:')
for batch in filtered_dataset.batch(batch_size):
    print(batch)

dataset_reshape = dataset.map(lambda x: tf.reshape(x, [-1, 6]))
print('Reshape:')
for element in dataset_reshape:
    print(element.numpy())

numpy_data = np.array(list(dataset.as_numpy_iterator()))
print('Converting into numpy: \n', numpy_data)

df = pd.DataFrame(columns=['A', 'B', 'C'])
for element in dataset:
    df_batch = pd.DataFrame(element.numpy(), columns=['A', 'B', 'C'])
    df = pd.concat([df, df_batch], ignore_index=True)
print('Converting into pd.DataFrame: ')
print(df)