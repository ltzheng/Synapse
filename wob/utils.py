import re
import os
from typing import List
from math import inf as infinity


def truncate_response(obs: str, end_str: str) -> str:
    """Truncate response string of LLM by given ending substring."""
    end_idx = obs.find(end_str)
    if end_idx != -1:
        obs = obs[:end_idx]

    return obs


def get_plan_list(plan: str) -> List[str]:
    """Process raw plan string to a list of actions."""
    pattern = r"`[^`]*`"
    matches = re.finditer(pattern, plan)
    actions = [match.group(0)[1:-1] for match in matches]

    return actions


def log_conversation(query, response, path):
    """Save a conversation to a file"""
    with open(path, "a") as f:
        f.write("INPUT:\n" + query + "\nOUTPUT:\n" + response + "\n" + "-" * 30 + "\n")

    return


def save_result(info, path):
    """Save result to a file at the end of the episode"""
    filename = os.path.splitext(os.path.basename(path))[0]
    with open(path, "a") as f:
        f.write(info)
        new_file_path = path.with_name(f"{filename}_{info.lower()}.txt")

    os.rename(path, new_file_path)

    return


def wins(state, player):
    win_state = (
        [[state[i][j] for j in range(3)] for i in range(3)]
        + [[state[j][i] for j in range(3)] for i in range(3)]
        + [[state[i][i] for i in range(3)], [state[2 - i][i] for i in range(3)]]
    )
    return [player] * 3 in win_state


def minimax(state, depth=None, player=1):
    if depth is None:
        depth = 9 - sum(1 for row in state for cell in row if cell != 0)
    if depth == 9:
        return [1, 1, 0]
    if depth == 0 or wins(state, -1) or wins(state, 1):
        return [-1, -1, 1 if wins(state, 1) else -1 if wins(state, -1) else 0]

    best = [-1, -1, -infinity if player == 1 else infinity]

    empty_cells = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]

    for x, y in empty_cells:
        state[x][y] = player
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == 1:
            if score[2] > best[2]:
                best = score
        else:
            if score[2] < best[2]:
                best = score

    return best
