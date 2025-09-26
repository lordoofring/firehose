import logging
import time
import os
from gymnasium.wrappers import RecordVideo
#from stable_baselines3 import Maskab
from sb3_contrib import MaskablePPO

#from stable_baselines3 import PPO

from gym_env import FireEnv
from firehose.helpers import IgnitionPoints, IgnitionPoint
pre_train_dir = './pretrained_models'
model_path_list = os.listdir(pre_train_dir)
# index3: 40x40 fixed_ppo, index4: 40x40 random_ppo (all maskable)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    model = MaskablePPO.load(os.path.join(pre_train_dir,model_path_list[3]))
    eval_env = FireEnv(ignition_points=IgnitionPoints([IgnitionPoint(1100, 1)]),observation_type='forest_rgb')
    obs , info = eval_env.reset()
    eval_env = RecordVideo(eval_env, 
                           video_folder="videos",
                           name_prefix="Maskable_init",
                           episode_trigger=lambda x : x % 10 == 0)
    eval_env.start_recording('40x40_test_Masked_PPO')
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, truncated,terminated, info = eval_env.step(action)
        eval_env.render()

        print("\n", action, reward)
        time.sleep(0.025)
        if terminated or truncated:
            obs, info = eval_env.reset()
            break
        
            

    eval_env.stop_recording()
    eval_env.close()

