import random

class SortGame:

    def __init__(self, empty_pos_indicator = 0, state_fun = "int"):
        self.empty_pos_indicator = empty_pos_indicator
        self.state_fun = state_fun
        self.reset()
        def get_state_int():
            b = self.board.copy()
            b.extend(self.currBricks[0:4])
            return b
        def get_state_encoded():
            encoding = []
            for i in range(16):
                #First position is filled/not filled, rest is encoded with if that position can be filled with a specific number
                encoding.append([0]*41)
            def _traverse_board(start,up_dir,side_dir):
                increment = side_dir if side_dir != 0 else up_dir*4
                if side_dir == -1:
                    end=start//4*4
                elif side_dir == 1:
                    end=start//4*4+3
                elif up_dir == -1:
                    end=start%4
                elif up_dir == 1:
                    end=start%4+12
                counter=0
                for i in range(start+increment,end+increment,increment):
                    if self.board[i] > 0:
                        limit=self.board[i]-(side_dir+up_dir)*counter
                        if side_dir+up_dir < 0:
                            return [0]*limit+[1]*(40-limit)
                        elif side_dir+up_dir > 0:
                            return [1]*(limit-1)+[0]*(41-limit)
                    counter += 1
                return [1]*40 

            mask_used = [1]*40
            for i in self.board:
                if i > 0:
                    mask_used[i-1]=0
            for i in range(len(encoding)):
                if self.board[i] > 0:
                    #If the board position is filled, then only set the first position, rest are zero
                    encoding[i][0] = 1
                else:
                    mask_up = _traverse_board(i,1,0)
                    mask_down = _traverse_board(i,-1,0)
                    mask_left = _traverse_board(i,0,1)
                    mask_right = _traverse_board(i,0,-1)

                    for j in range(40):
                        encoding[i][j+1]=mask_used[j]*mask_up[j]*mask_down[j]*mask_left[j]*mask_right[j]
            flat_encoding = [j for sub in encoding for j in sub]
            for i in range(4):
                brick_one_hot = [0]*40
                if self.currBricks[i] > 0:
                    brick_one_hot[self.currBricks[i]-1] =1
                flat_encoding.extend(brick_one_hot)
            return flat_encoding
        if state_fun == "int":
            self.get_state=get_state_int
        elif state_fun == "encoded":
            self.get_state=get_state_encoded

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
            if self.board[i] == self.empty_pos_indicator or self.board[i+1] == self.empty_pos_indicator or self.board[i] > self.board[i+1]:
                reward -= 1
                break
        col = action%4
        for i in range(col,col+3*4,4):
            if self.board[i] == self.empty_pos_indicator or self.board[i+4] == self.empty_pos_indicator or self.board[i] > self.board[i+4]:
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

