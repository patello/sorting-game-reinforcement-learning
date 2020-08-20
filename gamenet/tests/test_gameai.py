import pytest

from ..gameai import GameAI

def test_gameai_init():
    gameai = GameAI()
    pass

def test_gameai_get_valid_moves():
    gameai = GameAI()
    res = gameai.get_valid_moves()
    assert(type(res[0]) == bool)
    assert(len(res) == 16)

def test_gameai_step():
    gameai = GameAI()
    first_brick = gameai.currBricks[0]
    res = gameai.step(0)
    assert(gameai.board[0] == first_brick)

def test_gameai_reset():
    gameai = GameAI()
    gameai.step(0)
    gameai.reset()
    assert(gameai.board[0]==gameai.empty_pos_indicator)

@pytest.mark.parametrize("state_fun",[("int",20),("encoded",816)])
def test_gameai_get_state(state_fun):
    gameai=GameAI(state_fun=state_fun[0])
    res=gameai.get_state()
    assert(len(res)==state_fun[1])

