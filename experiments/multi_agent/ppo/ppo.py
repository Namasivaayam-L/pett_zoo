import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, Sequential
from tensorflow.kears.layers import Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp

import numpy as np

class Base(Model):
    def __init__(self, num_layers, width, model_path, name):
        super(Base, self).__init__()
        self.num_layers = num_layers
        self.width = width
        self.model_path = model_path
        self.model_name = name
        self.model = Sequential()
        for _ in range(num_layers):
            self.model.add(Dense(self.width, activation='relu'))
        
    def save_model(self):
        self.model.save(self.model_path+self.model_name+'.keras')

    def load_model(self):
        self.model = keras.models.load_model(self.model_path+self.model_name+'.keras')
        
class ActorNetwork(Base):
    def __init___(self, action_dims, learning_rate, num_layers, width, model_path, name):
        super(ActorNetwork, self).__init__(num_layers, width, model_path, name)
        self.model.add(Dense(action_dims, activation='softmax'))
        self.model.compile(optimizer=Adam(learning_rate))
    def call(self, state):
        return self.model(state)
    
class CriticNetwork(Base):
    def __init__(self, learning_rate, num_layers, width, model_path, name):
        super(CriticNetwork, self).__init__(num_layers, width, model_path, name)
        self.model.add(Dense(1, activation=None))
        self.model.compile(optimizer=Adam(learning_rate))
    def call(self, state):
        return self.model(state)
        
class Agent:
    def __init__(self, ip_dims, action_dims, learning_rate=0.0003, gamma=0.99, gae_lambda=0.0003, policy_clip=0.2,num_layers=3, width=5, model_path=None):
        self.ip_dims = ip_dims
        self.action_dims = action_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.num_layers = num_layers
        self.width = width
        self.model_path = model_path
        
        self.actor = ActorNetwork(action_dims, learning_rate, num_layers, width, model_path, name='Actor')
        self.critic = CriticNetwork(learning_rate, num_layers, width, model_path, name='Critic')
    
    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()
    
    def load_models(self):
        self.actor.load_model()
        self.critic.load_model()
        
    def act(self,state):
        state = tf.convert_to_tensor([state])
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        val = self.critic(state)
        
        return list(map(lambda x: x.numpy()[0],[action, log_prob, val]))
    
    def learn(self, experience):
        states, actions, old_probs, vals, rewards, dones, batches = experience
        values = vals
        advantage = np.zeros(len(rewards),dtype=np.float32)

        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k]+self.gamma * values[k+1])* (1-int(dones[k])) - values[k]
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        
        for batch in batches:
            with tf.GradientTape(persistent=True) as tape:
                states, actions, old_probs, vals, rewards, dones = list(map(lambda x: tf.convert_to_tensor(x[batch]),[experience[:-1]]))
                
                probs = self.actor(states)
                dist = tfp.distributions.Categorical(probs)
                new_probs = dist.log_prob(actions)
                
                critic_val = tf.squeeze(self.critic(states),1)
                
                prob_ratio = tf.math.exp(new_probs - old_probs)

                wt_probs = advantage[batch] * prob_ratio
                clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                
                wt_clipped_probs = clipped_probs * advantage[batch]
                actor_loss = tf.math.reduce_mean(-tf.math.minimum(wt_probs, wt_clipped_probs))
                
                returns = advantage[batch] + values[batch]
                critic_loss = MeanSquaredError(critic_val, returns)
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_gradients,self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(critic_gradients,self.critic.trainable_variables))
        