import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
import numpy as np

tf.keras.backend.set_floatx("float64")


class Actor(Model):
    def __init__(self, action_dim, epsilon, num_layers=5, width=32):
        super().__init__()
        self.action_dim = action_dim
        self.model = tf.keras.Sequential()
        for _ in range(num_layers):
            self.model.add(layers.Dense(width, activation=tf.nn.relu))
        self.mean_lyr = layers.Dense(self.action_dim)
        self.stdev_lyr = layers.Dense(self.action_dim)  # Assuming this is for stdev_layer
        self.epsilon = epsilon
        
    def normalize_action(self,action):
        norm_action = (4 - 0) * (action + 1) / 2 + 0 #(action_high - action_low)*(agent_output + 1) / 2 + action_low
        return int(np.clip(norm_action,0,3))
        
    def call(self, state):
        a = self.model(state)
        mu = self.mean_lyr(a)
        log_sigma = self.stdev_lyr(a)
        sigma = tf.exp(log_sigma)
        dist = tfp.distributions.Normal(mu, sigma)
        action_ = dist.sample()
        action = tf.tanh(action_)
        log_pi_ = dist.log_prob(action_)
        log_pi = log_pi_ - tf.reduce_sum(tf.math.log(1 - action**2 + self.epsilon), axis=1, keepdims=True)
        norm_actions = list(map(self.normalize_action,action[0]))
        return np.argmax(norm_actions), log_pi


class Critic(Model):
    def __init__(self, num_layers=3, width=32):
        super().__init__()
        self.model = tf.keras.Sequential()
        for _ in range(num_layers):
            self.model.add(layers.Dense(width, activation=tf.nn.relu))
        self.model.add(layers.Dense(1))

    def call(self, state, action):
        if len(action.shape) == 0:
            action = tf.broadcast_to(action, state.shape)
            print('broadcast')
        else: 
            action = tf.expand_dims(action, axis=1)
            print('expand')
        print(action)
        action = tf.cast(action, dtype=tf.float64)
        state_action = tf.concat([state, action], axis=1)
        return self.model(state_action)

class SoftActorCritic:
    def __init__(
        self,
        action_dim,
        num_layers,
        width,
        epsilon=1e-16,
        writer=None,
        epoch_step=1,
        learning_rate=0.0003,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
    ):
        self.policy = Actor(action_dim, epsilon, num_layers, width)
        self.q1 = Critic(num_layers, width)
        self.q2 = Critic(num_layers, width)
        self.target_q1 = Critic(num_layers, width)
        self.target_q2 = Critic(num_layers, width)
        self.writer = writer
        self.epoch_step = epoch_step
        self.alpha = tf.Variable(0.0, dtype=tf.float64)
        self.target_entropy = -tf.constant(action_dim, dtype=tf.float64)
        self.gamma = gamma
        self.polyak = polyak
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)
        
    def act(self, current_state):
        # current_state_ = np.array(current_state, ndmin=2)
        action, _ = self.policy(current_state[np.newaxis,...])
        return action

    def update_q_network(self, current_states, actions, rewards, next_states):
        critic_loss = self._calculate_critic_loss(
            current_states, actions, rewards, next_states, self.q1, self.target_q1
        ) + self._calculate_critic_loss(current_states, actions, rewards, next_states, self.q2, self.target_q2)

        self._apply_gradients(critic_loss, self.q1, self.critic1_optimizer)
        self._apply_gradients(critic_loss, self.q2, self.critic2_optimizer)

        return critic_loss

    def update_policy_network(self, current_states):
        actor_loss = self._calculate_actor_loss(current_states)
        self._apply_gradients(actor_loss, self.policy, self.actor_optimizer)
        return actor_loss

    def update_alpha(self, current_states):
        alpha_loss = self._calculate_alpha_loss(current_states)
        self._apply_gradients(alpha_loss, [self.alpha], self.alpha_optimizer)
        return alpha_loss

    def train(self, experience):
        current_states, actions, rewards, next_states = experience
        critic_loss = self.update_q_network(current_states, actions, rewards, next_states)
        actor_loss = self.update_policy_network(current_states)
        alpha_loss = self.update_alpha(current_states)
        
        return critic_loss, actor_loss, alpha_loss

    def update_weights(self):
        self._update_target_weights(self.q1, self.target_q1)
        self._update_target_weights(self.q2, self.target_q2)

    def _calculate_critic_loss(self, current_states, actions, rewards, next_states, critic, target_critic):
        with tf.GradientTape() as tape:
            q = critic(current_states, actions)
            pi_a, log_pi_a = self.policy(next_states)
            q_target = target_critic(next_states, pi_a)
            min_q_target = tf.minimum(q_target[0], q_target[1])
            soft_q_target = min_q_target - self.alpha * log_pi_a
            gamma_expanded = tf.convert_to_tensor(self.gamma, dtype=tf.float64)
            x = [rewards[i]+ gamma_expanded * soft_q_target[i] for i in range(rewards.shape[0])]
            # y = tf.stop_gradient(rewards + gamma_expanded * soft_q_target)
            y = tf.stop_gradient(tf.convert_to_tensor(x))
            critic_loss = tf.reduce_mean((q - y) ** 2)
        return critic_loss

    def _calculate_actor_loss(self, current_states):
        with tf.GradientTape() as tape:
            pi_a, log_pi_a = self.policy(current_states)
            print(pi_a.shape,pi_a)
            q1 = self.q1(current_states, pi_a)
            q2 = self.q2(current_states, pi_a)
            min_q = tf.minimum(q1, q2)
            soft_q = min_q - self.alpha * log_pi_a
            actor_loss = -tf.reduce_mean(soft_q)
        return actor_loss

    def _calculate_alpha_loss(self, current_states):
        with tf.GradientTape() as tape:
            pi_a, log_pi_a = self.policy(current_states)
            alpha_loss = tf.reduce_mean(-self.alpha * (log_pi_a + self.target_entropy))
        return alpha_loss

    def _apply_gradients(self, loss, model, optimizer):
        variables = model.trainable_variables
        grads = self._get_gradients(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

    def _get_gradients(self, loss, variables):
        with tf.GradientTape(persistent=True) as tape:
            grads = tape.gradient(loss, variables)
        grads = [tf.convert_to_tensor(g) for g in grads if g is not None]
        del tape
        return grads




    def _update_target_weights(self, model, target_model):
        for theta_target, theta in zip(target_model.trainable_variables, model.trainable_variables):
            theta_target.assign(self.polyak * theta_target + (1 - self.polyak) * theta)
