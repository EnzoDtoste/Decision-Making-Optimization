from ..problem import Problem
from ..choice import Choice
import random
import numpy as np

def random_players(players, count, main_player):
    random_list = random.choices(players, k=count-1)
    position = random.randint(0, count-1)
    random_list.insert(position, main_player)
    return random_list, position

def random_hands(count):
    fs = [(x, y) for x in range(10) for y in range(x, 10)]
    
    total = count * 10
    if total > len(fs):
        raise ValueError()

    sfs = random.sample(fs, total)
    random.shuffle(sfs)

    hands = []
    index = 0
    for _ in range(count):
        hand = sfs[index:index + 10]
        index += 10
        hands.append(hand)

    return hands

def random_player(plays, table):
    copy = [play for play in plays]
    random.shuffle(copy)
    return copy

def bigger(plays, table):
    return sorted(plays, key= lambda f: f[0] + f[1], reverse=True)

def hardest(plays, table):
    return [p[1] for p in sorted([(sum([0 if e == 0 else 1 for e in table[play[1]]]), play) for play in plays], key= lambda e: e[0], reverse=True)]

def hardest_bigger(plays, table):
    return [p[1] for p in sorted([(sum([0 if e == 0 else 1 for e in table[play[1]]]), play) for play in plays], key= lambda e: e[0], reverse=True) if p[0] > 7] + sorted(plays, key= lambda f: f[0] + f[1], reverse=True)


class Domino(Problem):
    def __init__(self, choice : Choice, choiceParameters):
        super().__init__(choice, choiceParameters)

    def get_available_plays(self, hand, head):
        total = hand + [(f[1], f[0]) for f in hand]
        if head is None:
            return total
        return [f for f in total if f[0] in head or f[1] in head]

    def get_current_embedding(self, params):
        head, hand, _, table = params

        nhd = np.array([-1, -1]).flatten() if head is None else np.array(head).flatten()
        nh = np.array([[hand[i][0], hand[i][1]] if i < len(hand) else [-1, -1] for i in range(len(table))]).flatten()
        return np.concatenate((nhd, nh, np.array(table).flatten()))

    def order_choices(self, choices, params):
        head, _, order, table = params
        return order(choices, table)

    def get_choices(self, params):
        head, hand, _, _ = params
        return self.get_available_plays(hand, head)

    def run(self, hands, players, index_main_player):
        self.reset_embeddings()
        self.choiceParameters.reset_state()

        current_player = -1
        head = None
        no_play_count = 0

        table = [[0 for _ in range(10)] for _ in range(10)]

        while True:
            if no_play_count >= len(players):
                sums = [sum([f[0] + f[1] for f in hand]) for hand in hands] 
                minn = max(sums)
                indexes = []

                for i, s in enumerate(sums):
                    if s < minn:
                        indexes = [i]
                        minn = s
                    elif s == minn:
                        indexes.append(i)

                return indexes

            current_player += 1
            if current_player >= len(players):
                current_player = 0

            if index_main_player == current_player:
                play = self.select_choice([head, hands[current_player], players[current_player], table])
            else:
                plays = self.get_available_plays(hands[current_player], head)
                if len(plays) > 0:
                    play = players[current_player](plays, table)[0]
                else:
                    play = None

            if play is None:
                no_play_count += 1
                continue

            no_play_count = 0
            
            if head is None:
                head = play
                table[play[0]][play[1]] = current_player + 1
                table[play[1]][play[0]] = current_player + 1

            else:
                side = 0

                if play[0] == head[0]:
                    head = (play[1], head[1])
                elif play[0] == head[1]:
                    head = (head[0], play[1])
                elif play[1] == head[0]:
                    side = 1
                    head = (play[0], head[1])
                else:
                    side = 1
                    head = (head[0], play[0])

                if play[0] == play[1]:
                    table[play[0]][play[1]] = current_player + 1
                else:
                    if side == 0:
                        table[play[0]][play[1]] = -(current_player + 1)
                        table[play[1]][play[0]] = current_player + 1
                    else:
                        table[play[0]][play[1]] = current_player + 1
                        table[play[1]][play[0]] = -(current_player + 1)


            if play in hands[current_player]:
                hands[current_player].remove(play)
            else:
                hands[current_player].remove((play[1], play[0]))

            if len(hands[current_player]) == 0:
                return current_player