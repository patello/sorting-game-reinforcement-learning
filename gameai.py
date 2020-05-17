import random

class GameAI:

    def __init__(self):
        self.reset()

    def get_state(self):
        b = self.board.copy()
        b.extend(self.randBricks[self.gameRound*4:self.gameRound*4+4])
        return b

    def get_valid_moves(self):
        return [self.board[i] == 0 for i in range(20)]
    
    def step(self,action):
        self.board[action] = self.randBricks[self.gameRound*4+self.brickPos]
        self.brickPos += 1
        if self.brickPos == 4:
            self.gameRound += 1
            self.brickPos = 0
 
        reward = 2
        row = action//4
        for i in range(row*4,row*4+3):
            if self.board[i] == 0 or self.board[i] > self.board[i+1]:
                reward -= 1
                break
        col = action%4
        for i in range(col,col+3*4,4):
            if self.board[i] == 0 or self.board[i] > self.board[i+1]:
                reward -= 1
                break
        return (reward,self.gameRound==4)

    def reset(self):
        self.board = [0] * 20
        self.randBricks = []
        while len(self.randBricks) < 16:
            r = random.randint(1,40)
            if r not in self.randBricks:
                self.randBricks.append(r)
        self.gameRound = 0
        self.brickPos = 0

if __name__ == "__main__":
    g = GameAI()
    print(g.get_state())
    print(g.step(13))
    print(g.get_state())
    print(g.step(14))
    print(g.get_state())

