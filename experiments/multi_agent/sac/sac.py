import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp
import numpy as np

class Base(Model):
    def __init__(self, n_actions, num_layers, width, model_path, name="critic"):
        super(Base, self).__init__()
        self.n_actions = n_actions
        self.num_layers = num_layers
        self.width = width
        self.model_name = name
        self.model_path = model_path
        self.model = Sequential()
        for _ in range(num_layers):
            self.model.add(Dense(self.width, activation="relu"))
        self.optimizer = Adam()
        self.loss = MeanSquaredError()

class CriticNetwork(Base):
    def __init__(self, n_actions, num_layers, width, model_path, name):
        super(CriticNetwork, self).__init__(n_actions, num_layers, width, model_path, name)
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        act_val = self.model(tf.concat([state, action], axis=1))
        return self.q(act_val)

class ValueNetwork(Base):
    def __init__(self,num_layers, width,model_path, name ):
        super(ValueNetwork, self).__init__(None, num_layers, width,model_path, name)
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_val = self.model(state)
        return self.v(state_val)

class ActorNetwork(Base):
    def __init__(self, n_actions, num_layers, width, model_path, max_action, name):
        super(ActorNetwork, self).__init__(n_actions, num_layers, width, model_path, name)
        self.max_action = max_action
        self.noise = 1e-6
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):
        prob = self.model(state)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = tf.clip_by_value(sigma, self.noise, 1)
        return mu, sigma

    def sample_normal(self, state):
        mu, sigma = self.call(state)
        probs = tfp.distributions.Normal(mu, sigma)
        actions = probs.sample()
        # print(actions)
        action = tf.math.tanh(actions) * self.max_action
        log_probs = probs.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        # print(action)
        return action, log_probs

class Agent:
    def __init__(self, state_dim, action_dim, max_action, learning_rate, gamma, tau, model_path,num_layers=5, width=32, reward_scale=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.num_layers = num_layers 
        self.width = width 
        self.max_action = max_action
        self.model_path = model_path
        self.reward_scale = reward_scale
        self.actor = ActorNetwork(action_dim, self.num_layers, self.width, self.model_path, self.max_action, name='Actor')
        self.critic_1 = CriticNetwork(action_dim, self.num_layers, self.width, self.model_path, name='Critic-1')
        self.critic_2 = CriticNetwork(action_dim, self.num_layers, self.width, self.model_path, name='Critic-2')
        self.value = ValueNetwork(self.num_layers, self.width,self.model_path, name='Value')
        self.target_value = ValueNetwork(self.num_layers, self.width,self.model_path, name='Target Value')
        
        self.actor.compile(optimizer = Adam(learning_rate))
        self.critic_1.compile(optimizer = Adam(learning_rate))
        self.critic_2.compile(optimizer = Adam(learning_rate))
        self.value.compile(optimizer = Adam(learning_rate))
        self.target_value.compile(optimizer = Adam(learning_rate))
        
        # self.update_network_params(tau=1)
    def act(self, state):
        state = tf.convert_to_tensor([state])
        actions,_ = self.actor.sample_normal(state)
        return actions
        
    def update_network_params(self,tau=None):
        if tau is None:
            tau = self.tau
        wts = []
        targets  = self.target_value.weights
        for i, wt in enumerate(self.value.weights):
            wts.append(wt*tau + targets[i]*(1-tau))
        self.target_value.set_weights(wts)
    
    def learn(self, experience):
        states,actions,rewards,next_states, dones = tuple(map(lambda x: tf.convert_to_tensor(x,dtype=tf.float32),experience))
        
        with tf.GradientTape() as tape:
            val = tf.squeeze(self.value(states),1)
            next_val = tf.squeeze(self.target_value(next_states),1)

            curr_pol_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_pol = self.critic_1(states,curr_pol_actions)
            q2_new_pol = self.critic_2(states,curr_pol_actions)
            critic_val = tf.squeeze(tf.math.minimum(q1_new_pol,q2_new_pol),1)
            
            val_target = critic_val - log_probs
            val_loss = 0.5* MeanSquaredError()(val, val_target)
        val_nwk_grad = tape.gradient(val_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(val_nwk_grad, self.value.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_pol_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_pol = self.critic_1(states,new_pol_actions)
            q2_new_pol = self.critic_2(states,new_pol_actions)
            critic_val = tf.squeeze(tf.math.minimum(q1_new_pol,q2_new_pol),1)
            
            actor_loss = log_probs - critic_val
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_nwk_grad = tape.gradient(actor_loss, self.value.trainable_variables)
        print(actor_nwk_grad, actor_loss)
        self.value.optimizer.apply_gradients(zip(actor_nwk_grad, self.value.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale * rewards + self.gamma * next_val * (1 - dones)
            q1_old_pol = tf.squeeze(self.critic_1(states,actions),1)
            q2_old_pol = tf.squeeze(self.critic_2(states,actions),1)
            critic_1_loss = 0.5 * MeanSquaredError(q1_old_pol, q_hat)
            critic_2_loss = 0.5 * MeanSquaredError(q2_old_pol, q_hat)
        critic_1_nwk_grad = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_nwk_grad = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_nwk_grad, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_nwk_grad, self.critic_2.trainable_variables))
        
        self.update_network_params()
        
    def save_models(self):
        print("Saving Models...")
        self.actor.save_weights(self.model_path+self.actor.model_name)
        self.critic_1.save_weights(self.model_path+self.actor.model_name)
        self.critic_2.save_weights(self.model_path+self.actor.model_name)
        self.value.save_weights(self.model_path+self.actor.model_name)
        self.target_value.save_weights(self.model_path+self.actor.model_name)

    def load_models(self):
        print("Saving Models...")
        self.actor.load_weights(self.model_path+self.actor.model_name)
        self.critic_1.load_weights(self.model_path+self.actor.model_name)
        self.critic_2.load_weights(self.model_path+self.actor.model_name)
        self.value.load_weights(self.model_path+self.actor.model_name)
        self.target_value.load_weights(self.model_path+self.actor.model_name)
        