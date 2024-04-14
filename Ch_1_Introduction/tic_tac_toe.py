from judger import Judger
from players import AIPlayer, HumanPlayer

def train(epochs: int, verbose: int = 500):
    player1 = AIPlayer(epsilon=0.01)
    player2 = AIPlayer(epsilon=0.01)
    judger = Judger(player1, player2)
    p1_wins = 0
    p2_wins = 0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        if i % verbose == 0:
            print(f'Epoch {i}, player 1 winrate: {p1_wins / i}, player 2 winrate: {p2_wins / i}')
        player1.backup()
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()

def compete(turns):
    player1 = AIPlayer(epsilon=0)
    player2 = AIPlayer(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    p1_wins = 0
    p2_wins = 0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        judger.reset()
    print(f'{turns}, player 1 winrate {p1_wins / turns}, player2 winrate {p2_wins / turns}')

def play():
    while True:
        player1 = HumanPlayer()
        player2 = AIPlayer(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print('You win!')
        else:
            print("It's a tie!")


if __name__ == '__main__':
    train(int(1e5))
    # compete(int(1e3))
    play()