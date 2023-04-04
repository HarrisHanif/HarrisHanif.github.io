import pandas as pd
import neat
import keras 
import backtrader as bt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pickle
import tensorflow as tf
import neat.reporting
import neat.checkpoint
import matplotlib.pyplot as plt
from neat.statistics import StatisticsReporter
import graphviz
import time
from neat.reporting import StdOutReporter
import os
import random
from statistics import mean, stdev 
from neat.genome import DefaultGenome as Genome
import visualize 
from neat.checkpoint import Checkpointer
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, Activation
from keras.optimizers import Adam
import gym

# Load the data
train_data = pd.read_csv('training_data.csv').drop('Date', axis=1)
test_data = pd.read_csv('testing_data.csv').drop('Date', axis=1)
val_data = pd.read_csv('validation_data.csv').drop('Date', axis=1)


# Normalize the data
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data)
test_data_norm = scaler.transform(test_data)
val_data_norm = scaler.transform(val_data)

# Reshape the data
timesteps = 10
X_train = []
y_train = []
for i in range(timesteps, len(train_data_norm)):
    X_train.append(train_data_norm[i-timesteps:i])
    y_train.append(train_data_norm[i][3])
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []
for i in range(timesteps, len(test_data_norm)):
    X_test.append(test_data_norm[i-timesteps:i])
    y_test.append(test_data_norm[i][3])
X_test = np.array(X_test)
y_test = np.array(y_test)

X_val = []
y_val = []
for i in range(timesteps, len(val_data_norm)):
    X_val.append(val_data_norm[i-timesteps:i])
    y_val.append(val_data_norm[i][3])
X_val = np.array(X_val)
y_val = np.array(y_val)

# Reshape the input data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 4))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 4))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 4))


sequence_length = 10  # Replace with your desired sequence length



plt.switch_backend('TkAgg')

checkpoint_folder = 'C:/Users/Harris Hanif/Desktop/checkpoints4'


def custom_save_checkpoint(population, species_set, generation, filename_prefix):
    """Save the current simulation state."""

    filename = f"{filename_prefix}_g{generation}.pkl"
    print(f"Saving checkpoint to {filename}")

    data = {
        'generation': generation,
        'population': population,
        'species_set': species_set,
        'rndstate': random.getstate()
    }

    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



checkpoint_folder = os.path.join(os.path.expanduser('~'), 'Desktop', 'checkpoints4')
final_checkpoint_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'checkpoints4', 'checkpoint_end.pkl')


os.makedirs(checkpoint_folder, exist_ok=True)


# Set a seed for reproducibility
random.seed(22)




# Load your gold price data (OHLC) and volume data here
data = pd.read_csv('data3_train.csv')
print(data.head())

# Manually parse the date string and convert it to datetime
dates = []
for date_str in data['Date']:
    dt = datetime.strptime(date_str, '%d/%m/%Y')
    dates.append(dt)

# Replace the 'Date' column in the dataframe with the parsed dates
data['Date'] = dates




# Add a new column to represent whether the bot is holding a position or not
data['holding_position'] = 0

class GoldTradingStrategy(bt.Strategy):
    params = (
        ('max_positions', 3),
        ('risk_percentage', 0.01),
    )

    def __init__(self, nn, scaler, data, data_scaled):
        self.nn = nn
        self.scaler = scaler
        self.data = data
        self.data_scaled = data_scaled
        self.position_count = 0
        self.num_trades = 0
        self.profitable_trades = []
        self.unprofitable_trades = []
        print("GoldTradingStrategy initialized")

    def notify_trade(self, trade):
        if trade.isclosed:
            if trade.pnl > 0:
                self.profitable_trades.append(trade)
            else:
                self.unprofitable_trades.append(trade)

    @staticmethod
    def calculate_consecutive_trades(strategy, trades_analyzer, profitable=True):
        trades = trades_analyzer.get_analysis()['closed']
        max_consecutive_trades = 0
        consecutive_trades = 0

        for trade in trades:
            if (trade['pnl'] > 0) == profitable:
                consecutive_trades += 1
            else:
                consecutive_trades = 0

            if consecutive_trades > max_consecutive_trades:
                max_consecutive_trades = consecutive_trades

        return max_consecutive_trades

    def get_input_data(self):
        input_data = np.hstack((self.data_scaled[-1, :], self.data['holding_position'][-1], self.data['dist_to_prev_close'][-1]))
        input_data = self.scaler.transform(input_data.reshape(1, -1))
        return input_data


    def next(self):
        current_position = self.position.size
        self.data['holding_position'][-1] = 1 if current_position else 0
        input_data = self.get_input_data()
        output = self.nn.activate(input_data)[0]

        # Add print statements to display input_data and output
        print(f"Input data: {input_data}")
        print(f"Output value: {output}")

        if self.position_count < self.params.max_positions:
            if output > 0.1 and current_position <= 0:  # Buy signal
                stop_loss = self.data['low'][-1] * 0.99
                distance = self.data['close'][-1] - stop_loss
                lot_size = (self.broker.get_value() * self.params.risk_percentage) / distance
                self.buy(size=lot_size)
                self.sell(size=lot_size, exectype=bt.Order.Stop, price=stop_loss)
                self.position_count += 1
                self.num_trades += 1
                print(f"[{self.datetime.date()}] BUY order executed, Price: {self.data.close[0]}, Size: {lot_size}")

            
            elif output < -0.1:  # Sell signal
                stop_loss = self.data['high'][-1] * 1.01
                distance = stop_loss - self.data['close'][-1]
                lot_size = (self.broker.get_value() * self.params.risk_percentage) / distance
                self.sell(size=lot_size)
                self.buy(size=lot_size, exectype=bt.Order.Stop, price=stop_loss)
                self.position_count += 1
                self.num_trades += 1
                print(f"[{self.datetime.date()}] SELL order executed, Price: {self.data.close[0]}, Size: {lot_size}")

        
        else:
            if output < -0.1 and current_position > 0:  # Close buy position
                self.close()
                self.position_count -= 1
                self.num_trades += 1
                print(f"[{self.datetime.date()}] Close BUY position, Price: {self.data.close[0]}")

            elif output > 0.1 and current_position < 0:  # Close sell position
                self.close()
                self.position_count -= 1
                self.num_trades += 1
                print(f"[{self.datetime.date()}] Close SELL position, Price: {self.data.close[0]}")

                

    @property
    def trades_count(self):
        return self.num_trades

# Define the evaluation function
def evaluate(net, X_val=X_val, y_val=y_val):
    y_pred = net.predict(X_val)
    print('Predictions:', y_pred)
    mse = np.mean((y_pred - y_val)**2)
    return 1 / mse


class KerasModel:
    
    def __init__(self, genome, config):
        # Set the input shape
        input_shape = (len(train_data_norm[0]),)

        # Create the input layer
        self.input_layer = Input(shape=input_shape)

        # Create a dictionary of hidden layers
        hidden_layers = {}

        # Add the first hidden layer
        for node_key in genome.nodes.keys():
            if node_key == 0:
                continue

            node = genome.nodes[node_key]

            # If the node is not enabled, skip it
            if not getattr(node, 'enabled', True):
                continue

            # Add the layer to the dictionary of hidden layers
            hidden_layers[node_key] = Dense(units=node.num_outputs, activation=Activation(node.activation).__name__, input_shape=input_shape)





        # Add the hidden layers to the model in the correct order
        for node_key, layer in sorted(hidden_layers.items()):
            self.input_layer = layer(self.input_layer)

        # Add the output layer
        self.output_layer = Dense(1, activation='linear')(self.input_layer)

        # Create the Keras model
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)

        # Compile the model
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, train_data, val_data):
        self.model.fit(train_data[:, :-1], train_data[:, -1], validation_data=(val_data[:, :-1], val_data[:, -1]),
                       epochs=self.config.num_epochs, batch_size=self.config.batch_size, verbose=0)

    def evaluate(self, test_data):
        mse = self.model.evaluate(test_data[:, :-1], test_data[:, -1], verbose=0)
        return mse








# Define the fitness 
def eval_genomes(genomes, config, checkpointer, p):
    # Load the data
    train_data = pd.read_csv('training_data.csv').drop('Date', axis=1)
    test_data = pd.read_csv('testing_data.csv').drop('Date', axis=1)
    val_data = pd.read_csv('validation_data.csv').drop('Date', axis=1)

    # Normalize the data
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data_norm = scaler.transform(train_data)
    test_data_norm = scaler.transform(test_data)
    val_data_norm = scaler.transform(val_data)

    for genome_id, genome in genomes:
        # Set the default fitness value to None instead of 0
        genome.fitness = None

        # Create the Keras model
        model = KerasModel(genome, config)

        # Train the model on the training data
        model.train(train_data_norm, val_data_norm)

        # Evaluate the model on the test data
        mse = model.evaluate(test_data_norm)

        # Set the genome fitness to negative MSE
        genome.fitness = -mse

    # Save the checkpoint
    checkpointer.save_checkpoint(config, p.population, p.species, p.generation)







    
       
        


def create_population(config, initial_state=None):
    if initial_state:
        genomes, genome_type, species_set_type = initial_state
        return neat.Population(config, (genomes, genome_type, species_set_type))
    else:
        return neat.Population(config)


def plot_fitness(scores):
    generation = list(range(len(scores)))
    max_fitness = [s[0] for s in scores]
    avg_fitness = [s[1] for s in scores]
    min_fitness = [s[2] for s in scores]

    plt.plot(generation, max_fitness, label='Max Fitness')
    plt.plot(generation, avg_fitness, label='Average Fitness')
    plt.plot(generation, min_fitness, label='Min Fitness')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc='upper right')
    plt.title('Fitness through Generations')
    plt.show()

class CustomReporter(StdOutReporter):
    def __init__(self, show_species_detail=False, output_folder="fitnessgenplots"):
        super().__init__(show_species_detail)
        self.generations = []
        self.max_fitness = []
        self.avg_fitness = []
        self.min_fitness = []

        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def start_generation(self, generation):
        super().start_generation(generation)
        self.generations.append(generation)

    def end_generation(self, config, population, species_set):
        fitness_values = [genome.fitness for genome in population.values()]

        # Filter out genomes with None fitness values
        valid_fitness_values = [fitness for fitness in fitness_values if fitness is not None]

        max_fitness = max(valid_fitness_values)
        min_fitness = min(valid_fitness_values)
        avg_fitness = sum(valid_fitness_values) / len(valid_fitness_values)

        self.max_fitness.append(max_fitness)
        self.min_fitness.append(min_fitness)
        self.avg_fitness.append(avg_fitness)

        plt.clf()
        plt.plot(self.generations, self.max_fitness, label='Max Fitness')
        plt.plot(self.generations, self.min_fitness, label='Min Fitness')
        plt.plot(self.generations, self.avg_fitness, label='Avg Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend(loc='upper right')
        plt.title('Fitness through Generations')
        plt.pause(0.01)

        super().end_generation(config, population, species_set)
        plt.savefig(os.path.join(self.output_folder, f"fitness_plot_generation_{len(self.generations)}.png"))


def plot_mean_fitness(generation_statistics):
    gen_numbers = list(range(0, len(generation_statistics)))
    mean_fitnesses = [s[0] for s in generation_statistics]

    plt.plot(gen_numbers, mean_fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.title("Mean Fitness Over Generations")
    plt.show()

def custom_mean(values):
    return sum(map(float, values)) / len(values) if len(values) > 0 else 0





# Define the function to create and train the NEAT model
def create_and_train_neat_model():
    # Load the configuration file and create the population object
    config_path = os.path.join(os.path.dirname(__file__), 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    checkpointer = Checkpointer(1)
    p = neat.Population(config)

    # Load the initial state, if available
    initial_state = None
    if os.path.isfile('checkpoints4/checkpoint_9'):
        checkpointer = neat.Checkpointer.restore_checkpoint('checkpoints4/checkpoint_9')
        p = create_population(config, checkpointer.population)
        initial_state = (checkpointer.population, Genome, neat.DefaultSpeciesSet)
    else:
        p = create_population(config, initial_state)

    # Add the custom reporter
    dynamic_plot_reporter = CustomReporter(output_folder="fitnessgenplots")
    p.add_reporter(dynamic_plot_reporter)

    # Add reporters to track progress
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Load the data and scale it
    data = pd.read_csv('data3_train.csv')
    scaler = MinMaxScaler()
    

    # Run NEAT
    winner = p.run(lambda genomes, _: eval_genomes(genomes, config, checkpointer, p), n=num_generations)



    # Save the winner and stats
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)

    return winner, stats, checkpointer, p.population, p.species, p.generation








    # Save the winner and stats
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)

    return winner, stats, p.checkpointer, p.population, p.species_set, p.generation

def visualize_winner(winner, config):
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    visualize.draw_net(config, winner, True, node_names=None, filename="winner_net.png")   

    
...


        


class TradingBot:
    def __init__(self, winner, checkpointer, population, species_set, generation):
        self.best_genome_net = None
        config_path = "config"
        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
       
        self.winner = winner
        self.checkpointer = checkpointer
        self.population = population
        self.species = species_set
        self.generation = generation

    def run(self):
        # Load the best genome
        with open('best_genome.pkl', 'rb') as f:
            self.best_genome_net = neat.nn.FeedForwardNetwork.create(pickle.load(f), self.config)

        

        # Create a cerebro instance
        cerebro = bt.Cerebro()

        # Add the data feed to cerebro
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)

        # Add the strategy to cerebro
        cerebro.addstrategy(GoldTradingStrategy, nn=self.best_genome_net, scaler=MinMaxScaler(), data=data)

        # Add the sizer to cerebro
        cerebro.addsizer(bt.sizers.FixedSize, stake=1000)

        # Add the commission to cerebro
        cerebro.broker.setcommission(commission=0.001)

        # Add analyzer to cerebro
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')

        # Run the backtest
        results = cerebro.run()


        # Get Sharpe ratio
        sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis()['sharperatio']
        print('Sharpe Ratio:', sharpe_ratio)

        plt.ion()
        # Plot the results
        cerebro.plot(style='candlestick', iplot=False)
        
        # Show the plot
        plt.show()
    
       
    def load_neural_network_model(self):
        # Load your trained neural network model here
        model = tf.keras.models.load_model('trained_nn_model.h5')
        return model




def extract_fitness_scores(stats, config, num_generations):
    fitness_scores = []

    for gen in range(num_generations):
        if 'fitness' in stats.generation_statistics[gen]:
            max_fitness = max(stats.generation_statistics[gen]['fitness'])
            avg_fitness = stats.get_fitness_mean()[gen]
            min_fitness = min(stats.generation_statistics[gen]['fitness'])
            fitness_scores.append((max_fitness, avg_fitness, min_fitness))
        else:
            print(f"Skipping generation {gen}: 'fitness' value not found.")
            continue

    return fitness_scores



def plot_fitness_scores(fitness_scores):
    generations = range(len(fitness_scores))
    max_fitness = [score[0] for score in fitness_scores]
    avg_fitness = [score[1] for score in fitness_scores]
    min_fitness = [score[2] for score in fitness_scores]

    plt.plot(generations, max_fitness, label="Max Fitness")
    plt.plot(generations, avg_fitness, label="Average Fitness")
    plt.plot(generations, min_fitness, label="Min Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()



def main(config, num_generations):
    
    winner, stats, checkpointer, population, species_set, generation = create_and_train_neat_model()
    
    fitness_scores = extract_fitness_scores(stats, config, num_generations)
    plot_fitness_scores(fitness_scores)

    trading_bot = TradingBot(winner, checkpointer, population, species_set, generation)
    trading_bot.run()

if __name__ == '__main__':
    config_path = "config"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    num_generations = 10  # Replace this with the actual number of generations used during the training
    main(config, num_generations)
    winner, stats, checkpointer, population, species, last_generation = create_and_train_neat_model()

    # Add this line inside the if __name__ == "__main__": block, after calling create_and_train_neat_model()
    visualize_winner(winner, neat.Config(Genome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config'))



    