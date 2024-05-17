import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Define the trading environment
class TradingEnvironment:
    def __init__(self, initial_balance, price_history):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.price_history = price_history
        self.current_step = 0
        self.max_steps = len(price_history) - 1
        self.position = 0  # Current position (0 for neutral, 1 for long, -1 for short)
        self.trades = []  # Record of trades

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0
        self.trades = []

    def get_state(self):
        return [self.balance, self.price_history[self.current_step], self.position]

    def take_action(self, action):
        # Execute the action (buy, sell, hold)
        if action == 0:  # Buy
            self.balance -= self.price_history[self.current_step]
            self.position = 1
            self.trades.append(('buy', self.price_history[self.current_step], self.current_step))
        elif action == 1:  # Sell
            self.balance += self.price_history[self.current_step]
            self.position = -1
            self.trades.append(('sell', self.price_history[self.current_step], self.current_step))
        else:  # Hold
            self.trades.append(('hold', self.price_history[self.current_step], self.current_step))
        self.current_step += 1

    def step(self, action):
        self.take_action(action)
        reward = self.balance - self.initial_balance
        done = self.current_step == self.max_steps
        next_state = self.get_state()
        return next_state, reward, done


# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000, batch_size=64, memory_size=10000,
                 target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.target_update_counter = 0

        # Main model
        self.model = self.build_model()

        # Target model (for stability)
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(self.state_size,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
    	if len(self.memory) < self.batch_size:
        	return
    	minibatch = random.sample(self.memory, self.batch_size)
    	for state, action, reward, next_state, done in minibatch:
        	if state is None or next_state is None:
            		continue  # Skip this transition if state or next_state is None

        	state = np.array(state)
        	next_state = np.array(next_state)

        	# Reshape states if needed
        	target = reward
        	if not done:
            		target = reward + self.discount_factor * np.amax(self.target_model.predict(next_state)[0])
        
        # Predict Q-values for the current state
        	try:
            		predicted_targets = self.model.predict(state)
            		predicted_targets[0][action] = target
            		self.model.fit(state, predicted_targets, epochs=1, verbose=0)
        	except Exception as e:
            		print(f"Error occurred during model prediction or fitting: {e}")

    	if self.epsilon > self.epsilon_min:
        	self.epsilon -= self.epsilon_decay

    def update_target_model(self):
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter += 1


# Training the DQN agent
def train_agent(env, agent, episodes=15):
    rewards = []
    balances = []

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state,[1])
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        agent.replay()
        agent.update_target_model()

        rewards.append(total_reward)
        balances.append(env.balance)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    return rewards, balances


# Plotting functions
def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Total Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()


def plot_balances(balances):
    plt.figure(figsize=(10, 6))
    plt.plot(balances)
    plt.title('Balance Over Time')
    plt.xlabel('Step')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.show()


def plot_action_distribution(trades):
    actions = [trade[0] for trade in trades]
    sns.countplot(actions)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.show()


def plot_q_values(agent, state_size=3, action_size=3):
    q_values = np.zeros((state_size, action_size))
    for i in range(state_size):
        state = np.zeros((1, state_size))
        state[0, i] = 1
        q_values[i] = agent.model.predict(state)
    plt.figure(figsize=(10, 6))
    sns.heatmap(q_values, annot=True, cmap='coolwarm')
    plt.title('Q-Value Heatmap')
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.xticks(ticks=np.arange(action_size), labels=['Buy', 'Sell', 'Hold'])
    plt.yticks(ticks=np.arange(state_size), labels=['Balance', 'Price', 'Position'])
    plt.show()


def plot_position_heatmap(trades, max_steps):
    positions = np.zeros(max_steps)
    for trade in trades:
        positions[trade[2]] = 1 if trade[0] == 'buy' else (-1 if trade[0] == 'sell' else 0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(positions.reshape(1, -1), cmap='coolwarm', cbar=False)
    plt.title('Position Heatmap')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.show()


# Main
if __name__ == "__main__":
    # Generate simulated price history
    np.random.seed(0)
    price_history = np.random.normal(loc=100, scale=5, size=100)

    # Initialize trading environment and agent
    env = TradingEnvironment(initial_balance=10000, price_history=price_history)
    agent = DQNAgent(state_size=3, action_size=3)

    # Train the agent and collect rewards and balances
    rewards, balances = train_agent(env, agent)

    # Plotting
    plot_rewards(rewards)
    plot_balances(balances)
    plot_action_distribution(env.trades)
    plot_q_values(agent)
    plot_position_heatmap(env.trades, len(price_history))

