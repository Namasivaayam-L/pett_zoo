import tensorflow as tf
import numpy as np
def _build_DQN(num_actions,num_layers,width):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(width, activation='relu'))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(width, activation='relu'))
    model.add(tf.keras.layers.Dense(num_actions, activation='linear'))
    return model

class DQN:
    def __init__(self, ts, num_actions, num_states, width, num_layers,batch_size, gamma, learning_rate, model_path, fine_tune_model_path=None):
        self.ts = ts
        self.num_actions = num_actions
        self.num_states = num_states
        self.batch_size = batch_size
        self.model_path = model_path
        self.gamma = gamma
        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model = _build_DQN(num_actions,num_layers,width)
        self.model.compile(optimizer=self.optimizer,loss=self.loss)
        if fine_tune_model_path != None:
            self.model = tf.keras.models.load_model(fine_tune_model_path)
            print("Model Loaded... from: ",fine_tune_model_path)
        
    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            q_values = np.random.randint(self.num_actions)
        else:
            q_values = np.argmax(self.model.predict(state[np.newaxis, ...]))
        return q_values
    
    def learn(self, experience):
        # Reshape individual observations into batches
        curr_state, action, reward, next_state = experience
        batch = tuple(map(lambda x: np.reshape(x,[self.batch_size,self.num_states]), (curr_state, next_state)))
        
        q_values = self.model.predict(batch[0])
        q_target = reward + self.gamma * np.max(self.model.predict(batch[1]))

        # Update target Q-values in the Q-values array
        q_target_vec = q_values.copy()
        q_target_vec[0][action] = q_target

        # Calculate loss and apply gradients
        with tf.GradientTape() as tape:
            q_predictions = self.model(batch[0])
            loss = self.loss(q_target_vec, q_predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.model.save(self.model_path+self.ts+'.keras')