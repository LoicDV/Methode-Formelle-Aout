import gymnasium as gym # environnement.
import numpy as np # manipulation de matrices.
import matplotlib.pyplot as plt # affichage des graphiques.
import tensorflow as tf # deep learning.
from collections import deque # manipulation de listes.
import random # générateur d'aléatoire.

### NAIF ###
# observation : tuple (somme du joueur, carte visible du croupier, as utilisable par le joueur).
# action : 0 (rester) ou 1 (tirer).
def resolve_blackjack(observation: tuple) -> int:
    player_sum, dealer_card, usable_ace = observation
    if player_sum >= 17: # Au dessus de 17, risque élevé de perdre et le dealer est obligé de tirer pour être entre 17 et 21.
        return 0
    if player_sum <= 11: # En dessous de 11, le joueur ne peut pas perdre.
        return 1
    if dealer_card <= 6: # Si la carte visible du dealer est faible, soit il tire un As et se retrouvera à la limite soit il tirera en dessous de 17 et devra en repiocher une.
        return 0
    return 1 # Sinon, le joueur risque de perdre. (cas où la carte du dealer est supérieure à 6 et le joueur est entre 12 et 16).




### UTILITAIRE ###
# Fonction d'exploration/exploitation. (Fonction reprise d'un groupe de projet de cette année).
def creation_epsilon_schedule(total_episodes, epsilon=1.0, epsilon_min=1e-4):
    epsilon_decay = 1- 10 **(int(np.log10(total_episodes))-1)
    x = np.arange(total_episodes)+1
    y = np.full(total_episodes, epsilon)
    y = np.maximum((epsilon_decay**x)*epsilon, epsilon_min)
    return y

def barplot(total_win, total_loose, total_draw, total_episodes):
    # Affichage des résultats.
    plt.bar(['Win', 'Loose', 'Draw'], [total_win, total_loose, total_draw])
    plt.show()
    # Pourcentage des résultats.
    print('Win :', total_win / total_episodes * 100, '%')
    print('Loose :', total_loose / total_episodes * 100, '%')
    print('Draw :', total_draw / total_episodes * 100, '%')




### Q-LEARNING ###
# Fonction qui donne l'action qui maximise la Q-Table.
def max_Q(Q, observations):
    if Q[observations[0], observations[1], observations[2], 0] < Q[observations[0], observations[1], observations[2], 1]:
        return 1
    else:
        return 0

# Fonction d'action d'entrainement.
def action_train(env, Q, observations, epsilon, shield):
    if shield: # Contrôle par le biais de notre méthode naïve.
        if np.random.uniform(0, 1) < 0.5: # Proba de 0.5 de faire la méthode naïve lorsque le shield est activé.
            return resolve_blackjack(observations)
    if np.random.uniform(0, 1) < epsilon: # Exploration.
        return env.action_space.sample()
    else: # Exploitation.
        return max_Q(Q, observations)

# Fonction d'amélioration de notre Q-Table.
def update_Q_Table(alpha, gamma, Q, reward, first_obs, action, second_obs):
    """
    first_obs[0] : Somme des cartes du joueur avant l'action.
    first_obs[1] : Carte du croupier.
    first_obs[2] : Si on possède un As avant l'action
    action : Action effectuée.
    second_obs[0] : Somme des cartes du joueur après l'action.
    second_obs[1] : Carte du croupier.
    second_obs[2] : Si on possède un As après l'action.
    alpha : Taux d'apprentissage.
    gamma : Facteur d'actualisation
    Définition d'update du Q_Learning trouvé sur : https://en.wikipedia.org/wiki/Q-learning#Algorithm
    """
    Q[first_obs[0], first_obs[1], first_obs[2], action] += alpha * (reward + gamma * max(Q[second_obs[0], second_obs[1], second_obs[2], 0], Q[second_obs[0], second_obs[1], second_obs[2], 1]) - Q[first_obs[0], first_obs[1], first_obs[2], action])

# Fonction d'action de test.
def action_test(Q, observations):
    return max_Q(Q, observations)




### DEEP Q-LEARNING ###
# Classe qui représente notre agent.
class Agent():
    # Constructeur de notre agent.
    def __init__(self, env, alpha, gamma, shield):
        self.env = env
        self.alpha = alpha # taux d'apprentissage.
        self.gamma = gamma # facteur de réduction.
        self.model = None # DNN de notre agent.
        self.memory = deque(maxlen=100) # mémoire de notre agent.
        self.create_model() # crée le DNN de notre agent.
        self.shield = shield # booléen qui indique si on utilise le shield ou non.

    def create_model(self):
        # Crée un DNN.
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim=1, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))
        self.model = model

    # Ajoute une transition à la mémoire de notre agent.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # Retournz l'action à effectuer pendant l'entrainement.
    def action_training(self, state, epsilon):
        if self.shield: # Contrôle par le biais de notre méthode naïve.
            if np.random.uniform(0, 1) < 0.5: # Proba de 0.5 de faire la méthode naïve lorsque le shield est activé.
                return resolve_blackjack(state)
        if np.random.uniform(0, 1) < epsilon: # Exploration.
            return self.env.action_space.sample()
        else: # Exploitation.
            return self.action(state)

    # Retourne l'action à effectuer.
    def action(self, state):
        predictions = self.model.predict(state, verbose=0) # Prédictions du DNN.
        return np.argmax(predictions[0]) # Retourne l'action qui maximise la prédiction.
    
    # Entraîne le DNN de notre agent.
    def train(self, batch_size):
        # Si la mémoire de notre agent ne contient pas assez de transitions, on ne fait rien.
        if len(self.memory) < batch_size:
            return
        # On récupère un batch de transitions.
        batch = np.array(random.sample(self.memory, batch_size), dtype=object)
        for state, action, reward, next_state, done in batch:
            prediction = self.model.predict(state, verbose=0)
            # On ajuste la reward en fonction si la partie est terminée ou non.
            if not done:
                reward += self.gamma * np.max(self.model.predict(next_state, verbose=0))
            prediction[0][action] = reward
            # On entraîne le DNN.
            self.model.fit(np.array(state), prediction, epochs=1, verbose=0)

    # Sauvegarde le DNN de notre agent.
    def save(self, name):
        self.model.save(name, save_format='h5')

    # Charge le DNN de notre agent.
    def load(self, name):
        self.model = tf.keras.models.load_model(name)