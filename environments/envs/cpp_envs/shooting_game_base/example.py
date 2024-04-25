import shooting_game_env
import random

#cd build && cmake .. && make && mv shooting_game_env.cpython-310-x86_64-linux-gnu.so .. && cd ..

def test():
    game = shooting_game_env.ShootingGame()
 #   game.init_render()
    game.reset()
    done = False
    print("hello test")
    
    step = 0
    cum_r = 0

    while not done:
  #      game.draw()
        action = (True, random.randint(0, 800), 35)
        state, reward, done, trunc, info = game.step(action)
        done = done or trunc
        cum_r +=reward
        print(f"Step: {step}, State: {state}, Reward: {reward}, Done: {done}, Cum_r: {cum_r}")
        step+=1


def main():
    test()

main()

