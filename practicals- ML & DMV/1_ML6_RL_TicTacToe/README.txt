============================================================
ML6 - REINFORCEMENT LEARNING:
TIC-TAC-TOE USING Q-LEARNING
============================================================

## PROBLEM STATEMENT

Build a Tic-Tac-Toe game using reinforcement learning in
Python by using the following tasks:
1. Setting up the environment
2. Defining the Tic-Tac-Toe game
3. Building the reinforcement learning model
4. Training the model
5. Testing the model

============================================================
DATASET
============================================================

No dataset is required.

In reinforcement learning the agent creates its own
experience by playing games (self-play).

============================================================
OBJECTIVE
============================================================

* To understand reinforcement learning concepts:
  agent, environment, state, action, reward, policy.
* To implement the Q-learning algorithm.
* To train an agent that plays Tic-Tac-Toe.
* To test the agent against a random player and a human.

============================================================
LIBRARIES USED
============================================================

1. random  - exploration and random opponent
2. pickle  - save the learned Q-table
3. os      - file handling

Only Python standard library - nothing to install.

============================================================
HOW TO RUN
============================================================

python ml6_tictactoe_rl.py

* Trains for 50,000 games (about 10-20 seconds).
* Tests against a random player.
* Shows a demo game.
* Saves q_table_X.pkl.
* Asks "Play against the agent? (y/n)". Type y to play
  as O by entering positions 0-8:

   0 | 1 | 2
   3 | 4 | 5
   6 | 7 | 8

============================================================
THEORY
============================================================

## Reinforcement Learning (RL)

RL is a type of machine learning where an AGENT learns by
interacting with an ENVIRONMENT. It takes ACTIONS, gets
REWARDS, and learns which actions give the most total
reward over time - learning by trial and error.

Supervised     : learns from labelled examples
Unsupervised   : finds patterns without labels
Reinforcement  : learns from rewards / penalties

------------------------------------------------------------

## Key Terms

* Agent       : the learner/decision maker (our player)
* Environment : the world (Tic-Tac-Toe board)
* State (s)   : current situation (board, e.g. "X O  X   ")
* Action (a)  : a move (empty cell 0-8)
* Reward (r)  : feedback (+1 win, -1 loss, 0.5 draw,
                0 otherwise)
* Policy (pi) : strategy - which action to take in a state
* Episode     : one complete game

------------------------------------------------------------

## Markov Decision Process (MDP)

RL problems are modelled as MDP: (S, A, P, R, gamma).
The next state depends only on the current state and
action (Markov property).

------------------------------------------------------------

## Q-Learning

Q-learning is a model-free, off-policy RL algorithm.
It learns a Q-function:

Q(s, a) = expected total future reward of taking action a
          in state s and playing well afterwards.

Values are stored in a Q-TABLE (state, action) -> value.
Initially all values are 0.

Update rule (Bellman equation):

Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max Q(s',a') - Q(s,a) ]

* alpha (learning rate)  = 0.3 : how much new info
                                 overrides old
* gamma (discount factor)= 0.9 : importance of future
                                 rewards
* r + gamma * max Q(s',a') = target (TD target)
* target - Q(s,a)          = TD error

------------------------------------------------------------

## Exploration vs Exploitation (epsilon-greedy)

* With probability epsilon (0.2): random move (explore)
* Otherwise: move with highest Q-value (exploit)

Exploration finds new strategies; exploitation uses what
is already learned.

------------------------------------------------------------

## Role of Reward Signal

The reward tells the agent what is good or bad. Since a
win/loss happens only at the end, Q-learning propagates
this reward backwards to earlier moves through the
gamma * max Q(s',a') term.

------------------------------------------------------------

## Self-Play

Two agents (X and O) play against each other and both
learn. As one improves, the other must improve too.

============================================================
COMPLETE CODE EXPLANATION
============================================================

## CODE BLOCK 1 - Environment (class TicTacToe)

WIN_LINES = [(0,1,2), (3,4,5), ... (2,4,6)]
reset()             -> empty board, X starts
state()             -> board as 9-char string (Q-table key)
available_actions() -> list of empty cells
winner()            -> "X", "O", "Draw" or None
step(action)        -> places mark, switches player,
                       returns (new_state, result)
render()            -> prints the board

------------------------------------------------------------

## CODE BLOCK 2 - Agent (class QLearningAgent)

self.Q = {}                 # Q-table dictionary
get_q(state, action)        # default 0.0

choose_action():
    if random.random() < epsilon: random move   (explore)
    else: move with max Q, ties broken randomly (exploit)

update():
    future = 0 if done else max Q(next_state, a')
    Q[(s,a)] = old + alpha * (reward + gamma*future - old)

------------------------------------------------------------

## CODE BLOCK 3 - Training (train())

for each episode:
    reset board
    loop:
        current player chooses action, env.step()
        if game continues:
            update opponent's previous move with reward 0
        if game over:
            winner's last move  -> reward +1
            loser's last move   -> reward -1
            draw                -> both 0.5

* Prints win/draw statistics every 10,000 episodes.
* Draws increase as both agents get better.

------------------------------------------------------------

## CODE BLOCK 4 - Testing

test_vs_random(agent, 1000)
* Agent plays with explore=False (pure exploitation)
  against a random player.

Demo game: trained X vs trained O (usually a draw -
perfect Tic-Tac-Toe play always ends in a draw).

play_human(agent): you play against the trained agent.

pickle.dump(agents["X"].Q, ...) saves the Q-table.

============================================================
IMPORTANT FUNCTIONS USED
============================================================

1. random.random(), random.choice()
2. TicTacToe.step(), winner(), available_actions()
3. QLearningAgent.choose_action()  - epsilon-greedy
4. QLearningAgent.update()         - Bellman update
5. train(), test_vs_random(), play_human()
6. pickle.dump()

============================================================
ARCHITECTURE / WORKFLOW
============================================================

Define environment (board, rules, winner check)
     ↓
Create two Q-learning agents (X, O)
     ↓
Self-play 50,000 episodes
   state → epsilon-greedy action → step → reward
         → Q-table update
     ↓
Test vs random player (1000 games)
     ↓
Demo game + save Q-table
     ↓
Optional: human vs agent

============================================================
OUTPUT EXPLANATION
============================================================

Sample output observed:

Episode 10000: X wins=2937 O wins=2155 Draws=4908
...
Episode 50000: X wins=3036 O wins=1860 Draws=5104

Q-table size for X: 5059 state-action pairs

Against random player (1000 games):
{'Agent': 924, 'Random': 11, 'Draw': 65}
-> win/draw rate = 98.9%

Demo game X vs O -> Draw (both play well).

============================================================
VIVA QUESTIONS WITH ANSWERS
============================================================

(Questions 1-4 are from the lab manual)

1. What is reinforcement learning and how does it differ
   from supervised and unsupervised learning?

Ans:
RL is learning by interaction: an agent takes actions in
an environment and learns from rewards. Supervised
learning uses labelled input-output pairs; unsupervised
finds hidden patterns in unlabelled data; RL has no labels
- only delayed reward signals - and its actions affect
future data.

------------------------------------------------------------

2. What are some real-world applications of reinforcement
   learning?

Ans:
Game playing (AlphaGo, chess), robotics, self-driving
cars, recommendation systems, traffic signal control,
resource/energy management, trading, ChatGPT-style models
(RLHF).

------------------------------------------------------------

3. How does the Q-learning algorithm work, and what is the
   Q-function?

Ans:
Q-function Q(s,a) is the expected total discounted reward
for taking action a in state s. Q-learning starts with a
zero Q-table, chooses actions epsilon-greedily, and after
each step updates Q(s,a) using
Q(s,a) += alpha [r + gamma max Q(s',a') - Q(s,a)]
until values converge. The best action is argmax_a Q(s,a).

------------------------------------------------------------

4. What is the role of a reward signal in reinforcement
   learning?

Ans:
It defines the goal. It tells the agent how good an action
was. The agent's aim is to maximize cumulative reward;
here +1 for win, -1 for loss, 0.5 for draw.

------------------------------------------------------------

5. What is a state and an action in this game?

Ans:
State = board configuration (9-character string).
Action = index of an empty cell (0-8).

------------------------------------------------------------

6. What is epsilon-greedy?

Ans:
Choosing a random action with probability epsilon and the
best known action otherwise.

------------------------------------------------------------

7. What is alpha (learning rate)?

Ans:
How much a new estimate replaces the old Q-value (0-1).

------------------------------------------------------------

8. What is gamma (discount factor)?

Ans:
Weight of future rewards (0-1). Near 1 = far-sighted.

------------------------------------------------------------

9. What is a Q-table?

Ans:
A lookup table storing Q-value for every (state, action).

------------------------------------------------------------

10. What is the Bellman equation?

Ans:
Q(s,a) = r + gamma * max Q(s',a') - value of a state
equals immediate reward plus discounted best future value.

------------------------------------------------------------

11. Why is Q-learning called off-policy?

Ans:
It learns the value of the greedy (best) policy while
following an exploratory epsilon-greedy policy.

------------------------------------------------------------

12. What is an episode?

Ans:
One complete game from start to terminal state.

------------------------------------------------------------

13. Why do draws increase during training?

Ans:
Both agents improve; optimal Tic-Tac-Toe play always
results in a draw.

------------------------------------------------------------

14. What is the limitation of tabular Q-learning?

Ans:
Q-table becomes too large for big state spaces; Deep
Q-Networks (DQN) use neural networks instead.

------------------------------------------------------------

15. Difference between Q-learning and SARSA?

Ans:
Q-learning uses max Q(s',a') (off-policy); SARSA uses the
Q-value of the action actually taken next (on-policy).

============================================================
CONCLUSION
============================================================

A Tic-Tac-Toe environment was defined and two Q-learning
agents were trained by self-play for 50,000 games. The
trained agent won or drew 98.9% of games against a random
player and plays a draw against an equally trained agent.
Hence reinforcement learning enables an agent to learn a
game strategy purely from rewards.

============================================================
END OF FILE
============================================================
