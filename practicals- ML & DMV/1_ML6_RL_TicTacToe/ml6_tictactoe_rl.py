import random
import pickle
import os

# ------------------------------------------------------------------
# 1. Setting up the environment / defining the Tic-Tac-Toe game
# ------------------------------------------------------------------
class TicTacToe:
    WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),      # rows
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),      # columns
                 (0, 4, 8), (2, 4, 6)]                 # diagonals
    def __init__(self):
        self.reset()
    def reset(self):
        self.board = [" "] * 9
        self.current = "X"
        return self.state()
    def state(self):
        return "".join(self.board)
    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == " "]
    def winner(self):
        for a, b, c in self.WIN_LINES:
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        if " " not in self.board:
            return "Draw"
        return None
    def step(self, action):
        self.board[action] = self.current
        result = self.winner()
        self.current = "O" if self.current == "X" else "X"
        return self.state(), result
    def render(self):
        b = self.board
        print(f" {b[0]} | {b[1]} | {b[2]}\n---+---+---\n {b[3]} | {b[4]} | {b[5]}\n---+---+---\n {b[6]} | {b[7]} | {b[8]}\n")


# ------------------------------------------------------------------
# 2. Building the reinforcement learning model (Q-learning agent)
# ------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, symbol, alpha=0.3, gamma=0.9, epsilon=0.2):
        self.symbol = symbol
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.Q = {}                 # Q-table : (state, action) -> value
    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)
    def choose_action(self, state, actions, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(actions)                      # explore
        qs = [self.get_q(state, a) for a in actions]
        best = max(qs)
        return random.choice([a for a, q in zip(actions, qs) if q == best])  # exploit
    def update(self, state, action, reward, next_state, next_actions, done):
        old = self.get_q(state, action)
        future = 0.0 if done or not next_actions else max(self.get_q(next_state, a) for a in next_actions)
        # Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max Q(s',a') - Q(s,a) ]
        self.Q[(state, action)] = old + self.alpha * (reward + self.gamma * future - old)


# ------------------------------------------------------------------
# 3. Training the model (self-play)
# ------------------------------------------------------------------
def train(episodes=50000):
    env = TicTacToe()
    agents = {"X": QLearningAgent("X"), "O": QLearningAgent("O")}
    stats = {"X": 0, "O": 0, "Draw": 0}
    for ep in range(1, episodes + 1):
        state = env.reset()
        last = {"X": None, "O": None}           # (state, action) of each player's previous move
        while True:
            player = env.current
            agent = agents[player]
            actions = env.available_actions()
            action = agent.choose_action(state, actions)
            next_state, result = env.step(action)
            if result is None:
                # give the *opponent* feedback for its previous move (reward 0, game continues)
                opp = "O" if player == "X" else "X"
                if last[opp] is not None:
                    s, a = last[opp]
                    agents[opp].update(s, a, 0, next_state, env.available_actions(), False)
                last[player] = (state, action)
                state = next_state
                continue
            # game over -> terminal rewards
            opp = "O" if player == "X" else "X"
            if result == "Draw":
                agent.update(state, action, 0.5, next_state, [], True)
                if last[opp]:
                    agents[opp].update(*last[opp], 0.5, next_state, [], True)
            else:                                   # current player just won
                agent.update(state, action, 1.0, next_state, [], True)
                if last[opp]:
                    agents[opp].update(*last[opp], -1.0, next_state, [], True)
            stats[result] += 1
            break
        if ep % 10000 == 0:
            print(f"Episode {ep:>6}:  X wins={stats['X']}  O wins={stats['O']}  Draws={stats['Draw']}")
            stats = {"X": 0, "O": 0, "Draw": 0}
    return agents


# ------------------------------------------------------------------
# 4. Testing the model
# ------------------------------------------------------------------
def test_vs_random(agent, games=1000):
    env = TicTacToe()
    res = {"Agent": 0, "Random": 0, "Draw": 0}
    for _ in range(games):
        env.reset()
        while True:
            acts = env.available_actions()
            if env.current == agent.symbol:
                a = agent.choose_action(env.state(), acts, explore=False)
            else:
                a = random.choice(acts)
            _, result = env.step(a)
            if result:
                res["Draw" if result == "Draw" else ("Agent" if result == agent.symbol else "Random")] += 1
                break
    return res


def play_human(agent):
    env = TicTacToe()
    print("You are O. Enter a position 0-8:\n 0 | 1 | 2\n 3 | 4 | 5\n 6 | 7 | 8\n")
    while True:
        if env.current == agent.symbol:
            a = agent.choose_action(env.state(), env.available_actions(), explore=False)
            print("Agent plays", a)
        else:
            a = -1
            while a not in env.available_actions():
                try:
                    a = int(input("Your move: "))
                except ValueError:
                    pass
        _, result = env.step(a)
        env.render()
        if result:
            print("Result:", "Draw" if result == "Draw" else f"{result} wins")
            break


# ---------------- main program ----------------
random.seed(0)
print("=== Training two Q-learning agents by self-play ===")
agents = train(50000)
print("\nQ-table size for X:", len(agents["X"].Q), "state-action pairs")

print("\n=== Testing trained X agent against a random player (1000 games) ===")
r = test_vs_random(agents["X"])
print(r, f"-> win/draw rate = {(r['Agent'] + r['Draw']) / 10:.1f}%")

print("\n=== Demo game: trained X vs trained O ===")
env = TicTacToe()
env.reset()
while True:
    p = agents[env.current]
    _, result = env.step(p.choose_action(env.state(), env.available_actions(), explore=False))
    env.render()
    if result:
        print("Result:", result)
        break

with open("q_table_X.pkl", "wb") as f:
    pickle.dump(agents["X"].Q, f)
print("Q-table saved to q_table_X.pkl")

try:
    if input("\nPlay against the agent? (y/n): ").strip().lower() == "y":
        play_human(agents["X"])
except EOFError:
    pass
