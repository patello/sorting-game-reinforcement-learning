import pytest

from ..sort_game import SortGame

def test_sort_game_init():
    sort_game = SortGame()
    pass

def test_sort_game_get_valid_moves():
    sort_game = SortGame()
    res = sort_game.get_valid_moves()
    assert(type(res[0]) == bool)
    assert(len(res) == 16)

def test_sort_game_step():
    sort_game = SortGame()
    first_brick = sort_game.currBricks[0]
    res = sort_game.step(0)
    assert(sort_game.board[0] == first_brick)

def test_sort_game_reset():
    sort_game = SortGame()
    sort_game.step(0)
    sort_game.reset()
    assert(sort_game.board[0]==sort_game.empty_pos_indicator)

@pytest.mark.parametrize("state_fun",[("int",20),("encoded",816)])
def test_sort_game_get_state(state_fun):
    sort_game=SortGame(state_fun=state_fun[0])
    res=sort_game.get_state()
    assert(len(res)==state_fun[1])

