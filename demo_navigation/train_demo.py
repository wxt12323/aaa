import argparse
import os
import sys


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "gym-pybullet-drones"))
sys.path.append(os.path.join(ROOT, "demo_navigation"))

import pybullet as p
import imageio
from NavigateAviary import NavigateAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

# class GifRecordingCallback(BaseCallback):
#     """
#     自定义回调函数：在训练期间定期保存无人机的表现为 GIF。
#     """
#     def __init__(self, record_freq: int, output_dir: str, env_kwargs: dict, verbose=1):
#         super(GifRecordingCallback, self).__init__(verbose)
#         self.record_freq = record_freq
#         self.output_dir = output_dir
#         self.env_kwargs = env_kwargs

#     def _on_step(self) -> bool:
#         if self.n_calls % self.record_freq == 0:
#             self.save_gif()
#         return True

#     def save_gif(self):
#         if self.verbose:
#             print(f"[GIF Callback] Recording GIF at step {self.num_timesteps}...")
            
#         # 创建一个独立的、不需要GUI的测试环境用于截图
#         env = NavigateAviary(**self.env_kwargs, gui=False)
#         obs, info = env.reset()
#         frames = []
        
#         # 跑一个完整的回合
#         max_steps = int(env.EPISODE_LEN_SEC * env.CTRL_FREQ)
#         for step in range(max_steps):
#             action, _ = self.model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
            
#             # 每 2 帧截一张图，减少 GIF 体积
#             if step % 2 == 0:
#                 # 手动计算摄像机角度，盯住无人机
#                 view_matrix = p.computeViewMatrixFromYawPitchRoll(
#                     cameraTargetPosition=env.pos[0],
#                     distance=3.5,
#                     yaw=-45,
#                     pitch=-30,
#                     roll=0,
#                     upAxisIndex=2,
#                     physicsClientId=env.CLIENT
#                 )
#                 proj_matrix = p.computeProjectionMatrixFOV(
#                     fov=60.0,
#                     aspect=1.0,
#                     nearVal=0.1,
#                     farVal=100.0,
#                     physicsClientId=env.CLIENT
#                 )
#                 # 使用 ER_TINY_RENDERER (CPU渲染器)，它不需要你开启任何桌面GUI窗口就能截图
#                 _, _, rgbImg, _, _ = p.getCameraImage(
#                     width=400,
#                     height=400,
#                     viewMatrix=view_matrix,
#                     projectionMatrix=proj_matrix,
#                     renderer=p.ER_TINY_RENDERER,
#                     physicsClientId=env.CLIENT
#                 )
#                 frames.append(rgbImg[:, :, :3])  # 去掉 alpha 通道
                
#             if terminated or truncated:
#                 break
                
#         env.close()
        
#         if frames:
#             gif_path = os.path.join(self.output_dir, f"train_step_{self.num_timesteps}.gif")
#             imageio.mimsave(gif_path, frames, fps=15)
#             if self.verbose:
#                 print(f"[GIF Callback] Successfully saved GIF: {gif_path}")

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'demo_results'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('vel')
DEFAULT_AGENTS = 4
DEFAULT_TIMESTEPS = 6_000_000

def run(total_timesteps: int = DEFAULT_TIMESTEPS, n_envs: int = DEFAULT_AGENTS, gui: bool = DEFAULT_GUI):
    output_dir = os.path.join(ROOT, "demo_navigation", DEFAULT_OUTPUT_FOLDER)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    act_type = DEFAULT_ACT
    obs_type = DEFAULT_OBS

    def make_env():
        # Wrap with Monitor to track episodic returns and success rate (if info contains 'is_success')
        env = NavigateAviary(obs=obs_type, act=act_type, gui=gui)
        return Monitor(env, output_dir, info_keywords=("is_success",))

    train_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv  # <--- 使用多进程并行，提升收集数据的速度
    )
    
    eval_env = Monitor(NavigateAviary(obs=obs_type, act=act_type, gui=False), output_dir, info_keywords=("is_success",))

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=128,
        device="cpu", # 使用 CPU 进行训练
    )

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        verbose=1,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # # 每 50000 步通过内置 CPU 渲染器录制一个 GIF
    # gif_callback = GifRecordingCallback(
    #     record_freq=50000,
    #     output_dir=output_dir,
    #     env_kwargs={'obs': obs_type, 'act': act_type}
    # )

    # # 把两个 callback 打包起来组合使用
    # from stable_baselines3.common.callbacks import CallbackList
    # combined_callback = CallbackList([eval_callback, gif_callback])

    model.learn(total_timesteps=int(total_timesteps), 
    # callback=combined_callback)
    callback=eval_callback)
    model.save(os.path.join(output_dir, "final_model"))
    print(f"[INFO] Training done. Model saved at {output_dir}")

    # 画出训练曲线并保存
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        eval_file = os.path.join(output_dir, "evaluations.npz")
        if os.path.exists(eval_file):
            data = np.load(eval_file)
            timesteps = data["timesteps"]
            results = data["results"] # (N, n_eval_episodes)
            successes = data.get("successes", None)
            
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)
            
            plt.figure(figsize=(12, 5))
            
            # 绘制回报 (Return) 曲​​线
            plt.subplot(1, 2, 1)
            plt.plot(timesteps, mean_rewards, label="Mean Reward")
            plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title("Evaluations Return")
            plt.grid(True)
            plt.legend()
            
            # 绘制成功率 (Success Rate) 曲线
            if successes is not None:
                mean_success = np.mean(successes, axis=1)
                plt.subplot(1, 2, 2)
                plt.plot(timesteps, mean_success, color='orange', label="Success Rate")
                plt.xlabel("Timesteps")
                plt.ylabel("Success Rate")
                plt.title("Evaluations Success Rate")
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, "training_curves.png")
            plt.savefig(plot_file)
            print(f"[INFO] Learning curves saved at {plot_file}")
    except Exception as e:
        print(f"[WARNING] Could not plot training curves: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui', default=DEFAULT_GUI, action='store_true', help='Whether to use PyBullet GUI (default: False)')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, action='store_true', help='Whether to record a video (default: False)')
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS, help='Total timesteps (default: 5,000,000)')
    parser.add_argument("--n-envs", type=int, default=DEFAULT_AGENTS, help='Number of parallel environments (default: 4)')
    args = parser.parse_args()
    
    run(total_timesteps=args.timesteps, n_envs=args.n_envs, gui=args.gui)
