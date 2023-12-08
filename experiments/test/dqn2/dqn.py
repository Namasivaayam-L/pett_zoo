import tensorflow as tf
import numpy as np
def _build_DQN(num_states,num_actions,num_layers,width):
    inputs = tf.keras.Input(shape=(None,num_states))
    x = tf.keras.layers.Dense(width, activation='relu')(inputs)
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(width, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DQN')
    return model

class DQN:
    def __init__(self, num_actions, num_states, width, num_layers,batch_size, gamma, learning_rate):
        self.num_actions = num_actions
        self.num_states = num_states
        self.batch_size = batch_size
        self.model = _build_DQN(num_states,num_actions,num_layers,width)
        self.gamma = gamma
        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def act(self, state, experience_replay, ts, eps):
        if np.random.rand() <= eps and len(experience_replay.buffer) > 0:
            q_values = experience_replay.sample_batch(1,ts)[1]
            return q_values
        else:
            q_values = self.model.predict(state[np.newaxis,...])
            return np.argmax(q_values)

    def learn(self, experience):
        # Reshape individual observations into batches
        curr_state, action, reward, next_state = experience
        # print(curr_state)
        # batch = tuple(map(lambda x: np.reshape(x,[self.batch_size,self.num_states]), (curr_state, next_state)))
        batch = (curr_state,next_state)
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
