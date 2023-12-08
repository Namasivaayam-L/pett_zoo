import tensorflow as tf
import numpy as np

def _build_SAC_actor(num_actions,num_layers,width):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(width, activation='relu'))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(width, activation='relu'))
    model.add(tf.keras.layers.Dense(num_actions, activation='tanh'))
    return model

def _build_SAC_critic(num_layers,width):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(width, activation='relu'))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(width, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model

class SAC:
    def __init__(self, num_actions, num_states, width, num_layers, batch_size, gamma, learning_rate, tau, log_file, fine_tune_actor_path=None, fine_tune_critic_path=None):
        self.num_actions = num_actions
        self.num_states = num_states
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        self.log_file = log_file

        # Create actor and critic networks
        self.actor = _build_SAC_actor(num_actions, width, num_layers)
        self.critic = _build_SAC_critic(width, num_layers)
        self.critic_target = _build_SAC_critic(width, num_layers)

        # Load fine-tuned models if provided
        if fine_tune_actor_path != None:
            self.actor = tf.keras.models.load_model(fine_tune_actor_path)
            print("Actor model loaded from: ", fine_tune_actor_path)
        if fine_tune_critic_path != None:
            self.critic = tf.keras.models.load_model(fine_tune_critic_path)
            print("Critic model loaded from: ", fine_tune_critic_path)

        # Initialize optimizer and loss functions
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

    def act(self, state):
        # Get action probability distribution from actor network
        action_probs = self.actor(state[np.newaxis, ...]).numpy()[0]
        action_probs = action_probs / action_probs.sum()
        # action_probs = np.clip(action_probs, 0, 1)
        # Sample an action from the action probability distribution
        action = np.random.choice(self.num_actions, p=action_probs)

        # Print action to log file
        print(f'Predicted action: {action}', file=self.log_file)

        return action


    def learn(self, experience):
        # Reshape individual observations into batches
        curr_state, action, reward, next_state = experience
        batch = tuple(map(lambda x: np.reshape(x, [self.batch_size, self.num_states]), (curr_state, next_state)))

        # Calculate target Q-values using critic target network
        q_target_next = self.critic_target(batch[1])
        q_target_values = reward + self.gamma * q_target_next

        # Calculate Q-values using critic network
        q_values = self.critic(batch[0])

        # Update critic network
        with tf.GradientTape() as tape:
            critic_loss = self.loss(q_target_values, q_values)
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Update actor network using policy gradient method
        with tf.GradientTape() as tape:
            actor_loss = -self.critic(batch[0], self.actor(batch[0]))
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        # Update critic target network with polyak averaging
        for target_param, param in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            target_param.assign(target_param * self.tau + param * (1.0 - self.tau))
