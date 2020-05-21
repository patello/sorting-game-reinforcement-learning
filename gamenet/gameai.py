import random

class GameAI:

    def __init__(self, empty_pos_indicator = 0):
        self.empty_pos_indicator = empty_pos_indicator
        self.reset()

    def get_state(self):
        b = self.board.copy()
        b.extend(self.currBricks[0:4])
        return b

    def get_valid_moves(self):
        return [self.board[i] == self.empty_pos_indicator for i in range(16)]
    
    def step(self,action):
        self.board[action] = self.currBricks[0]
        self.brickPos += 1
        for i in range(4):
            self.currBricks[i] = self.currBricks[i+1]
        if self.brickPos == 4:
            self.gameRound += 1
            self.currBricks = [self.empty_pos_indicator]*5
            if self.gameRound != 4:
                for i in range(4):
                    self.currBricks[i] = self.randBricks[self.gameRound*4+i]
            self.brickPos = 0
 
        reward = 2
        row = action//4
        for i in range(row*4,row*4+3):
            if self.board[i] == self.empty_pos_indicator or self.board[i] > self.board[i+1]:
                reward -= 1
                break
        col = action%4
        for i in range(col,col+3*4,4):
            if self.board[i] == self.empty_pos_indicator or self.board[i] > self.board[i+4]:
                reward -= 1
                break
        #If only one move is available, do an additional step (the only remaining valid action) and add the reward.
        if sum(self.get_valid_moves())==1:
            (lastReward,_)=self.step(self.get_valid_moves().index(True))
            reward+=lastReward
        return (reward,self.gameRound==4)

    def reset(self):
        self.board = [self.empty_pos_indicator] * 16
        self.randBricks = []
        while len(self.randBricks) < 16:
            r = random.randint(1,40)
            if r not in self.randBricks:
                self.randBricks.append(r)
        self.currBricks = [self.empty_pos_indicator]*5
        for i in range(4):
            self.currBricks[i] = self.randBricks[i]
        self.gameRound = 0
        self.brickPos = 0

if __name__ == "__main__":
    g = GameAI()
    for i in range(10):
        print(g.get_state())
        print(g.step(i))

