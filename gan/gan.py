import pandas as pd, numpy as np, tensorflow as tf
from tensorflow.keras import optimizers,losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# Define parameters
latent_dim = 10  # Adjust this based on your data complexity
batch_size = 32
epochs = 100

# Read data from CSV
data = pd.read_csv("outputs/multi_agent/sac/2x2/average-speed/100eps/state.csv",header=0)
# Separate features
data = data.iloc[:,:-1]
# Loop through each column containing string values
for col in data.columns:
    # print(col)
    data[col] = data[col].apply(lambda element: np.array(element))
print(type(data['1'].head(1)[0]))
features = pd.concat([data['1'], data['2'], data['5'],data['6']], axis=0, ignore_index=True)
features = features.to_csv('gan/comb.csv', index=False)
# # Define the generator
generator = Sequential([
    Dense(128, activation="relu", input_shape=(latent_dim,)),
    Dense(256, activation="relu"),
    Dense(21),  # Adjust this to match the desired number of output values
])

# Define the discriminator
discriminator = Sequential([
    # Embed each feature separately (adjust embedding dimension if needed)
    Embedding(1, 128, input_length=21),
    LSTM(256, return_sequences=True),
    LSTM(128),
    # Adjust the layers and activation functions based on your data
    Dense(1, activation="sigmoid")
])

# Combine the generator and discriminator into a GAN model
gan_model = Sequential([generator, discriminator])

# Define loss function (binary cross-entropy is suitable for single output)
loss_function = losses.BinaryCrossentropy()

# Define optimizers
discriminator_optimizer = optimizers.Adam(learning_rate=0.0002)
generator_optimizer = optimizers.Adam(learning_rate=0.0002)

# Compile the generator and discriminator
discriminator.compile(loss=loss_function, optimizer=discriminator_optimizer)
gan_model.compile(loss=loss_function, optimizer=generator_optimizer)

# Train the GAN model
for epoch in range(epochs):
    # Generate latent vectors
    latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Generate fake features
    fake_features = generator.predict(latent_vectors)
    # print('fake features',fake_features, fake_features.shape)
    # Combine real and fake data
    real_data = np.array(features.sample(batch_size).reset_index(drop=True))
    real_data = list(map(lambda element: np.array(element), real_data))
    print('Real features',real_data,len(real_data), type(real_data))
    combined_data = np.concatenate([real_data, fake_features])

    # Train the discriminator
    discriminator_labels = np.concatenate([np.ones((batch_size, 1)),
                                         np.zeros((batch_size, 1))])
    discriminator.train_on_batch(combined_data, discriminator_labels)

    # Train the generator
    generator_labels = np.ones((batch_size, 1))
    gan_model.train_on_batch(latent_vectors, generator_labels)

# Generate new data samples
new_features = generator.predict(np.random.normal(size=(100, latent_dim)))

# Print the generated data (single array with 21 values)
print(new_features)

