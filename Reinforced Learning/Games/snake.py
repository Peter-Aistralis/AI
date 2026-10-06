from collections import deque
import os
import random
import subprocess
import time

import cv2
from gym import Env
from gym.spaces import Box, Discrete
import mss
import numpy as np
from pynput.keyboard import Controller, Key
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==================== HYPERPARAMETERS ====================
STEP_TIME = 0.09
INPUT_SHAPE = (5, 60, 66) # changed to 5 channels instead of 4
ACTION_SIZE = 3

N_STEP = 3

PER_ALPHA = 0.6    # how much prioritization (0 = uniform, 1 = full)
PER_BETA  = 0.4    # importance sampling correction (0 = none, 1 = full)
PER_EPS   = 1e-6   # small constant so priorities never hit zero

LEARNING_RATE = 1e-4
REPLAY_MEMORY_SIZE = 50_000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998
TARGET_UPDATE_FREQ = 1000

EPSILON_DECAY_STEPS = 50_000


# ==================== 1. ENVIRONMENT ====================
class Snake(Env):
    
  def _save_debug_channels(self):
    """Save the exact channels the network sees."""
    raw_bgr, stacked = self._capture_frame()
    names = ["gray", "apple", "head", "body", "wall"]
    for i, name in enumerate(names):
        img = (stacked[i] * 255).astype(np.uint8)
        cv2.imwrite(f"debug_ch{i}_{name}.png", img)
        print(f"  ch{i} ({name}): max={img.max()}, mean={img.mean():.1f}")

  def __init__(self):
    super().__init__()
    self.output_dir = "game_screenshots"
    os.makedirs(self.output_dir, exist_ok=True)

    self.step_count = 0
    self.active_apple = None
    self.apples_eaten = 0
    self.last_head_pos = None
    self.keyboard = Controller()
    
    #self.n_step = N_STEP
    #self.n_step_buffer = deque(maxlen=N_STEP)

    self.direction = 0
    self.action_space = Discrete(ACTION_SIZE)
    self.observation_space = Box(
        low=0.0, high=1.0, shape=INPUT_SHAPE, dtype=np.float32
    )

    self.capture = mss.mss()
    self.game_loc = {"top": 235, "left": 400, "width": 660, "height": 600}
    self.replay_loc = {"top": 600, "left": 600, "width": 285, "height": 45}

    self.stacked_frames = np.zeros(INPUT_SHAPE, dtype=np.float32)

  def focus_chrome(self):
    try:
      script = 'tell application "Google Chrome" to activate'
      subprocess.run(["osascript", "-e", script], check=True)
      time.sleep(0.5)
      print("✅ Chrome activated")
      return True
    except Exception as e:
      print(f"❌ Could not focus Chrome: {e}")
      return False

  def _capture_frame(self):
    raw_bgra = np.array(self.capture.grab(self.game_loc))
    raw_bgr = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)

    # Channel 0: Grayscale -> walls and background
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    ch0 = cv2.resize(gray, (66, 60), interpolation=cv2.INTER_AREA)
    ch0 = ch0.astype(np.float32) / 255.0

    # Channel 1: Red -> Apple
    red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
          cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    ch1 = cv2.resize(red, (66, 60), interpolation=cv2.INTER_AREA)
    ch1 = ch1.astype(np.float32) / 255.0

    # Channel 2 -> Head (head has same color as body, so look for 'white' eyes, so you know this is the head)
    blue = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))

    # Find eye pixels (white) — always located on the head
    white = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 230)).astype(np.uint8) * 255
    
    # Dilate eyes ->  "head zone" around
    kernel = np.ones((25, 25), np.uint8)
    head_zone = cv2.dilate(white, kernel, iterations=1)
    
    # Head -> blue pixels inside the head zone
    head = cv2.bitwise_and(blue, head_zone)
    
    # Body -> blue pixels NOT in the head
    body = cv2.bitwise_and(blue, cv2.bitwise_not(head))
    
    ch2 = cv2.resize(head, (66, 60), interpolation=cv2.INTER_AREA)
    ch2 = ch2.astype(np.float32) / 255.0

    # Channel 3 -> body 
    ch3 = cv2.resize(body, (66, 60), interpolation=cv2.INTER_AREA)
    ch3 = ch3.astype(np.float32) / 255.0
    
    # Channel 4 -> walls
    wall = cv2.inRange(hsv, (35, 100, 80), (90, 200, 180))
    #print(f"  wall mask: {int(np.sum(wall > 0))} pixels (out of {wall.size})")
    ch4 = cv2.resize(wall, (66, 60), interpolation=cv2.INTER_AREA)
    ch4 = ch4.astype(np.float32) / 255.0

    stacked = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0)
    

    return raw_bgr, stacked

  def _wait_for_game_start(self, timeout=5.0):
    start = time.time()
    last_press = 0.0
    while time.time() - start < timeout:
        raw_bgra = np.array(self.capture.grab(self.game_loc))
        raw_bgr = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)

        blue_mask = (
            (hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130)
            & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 100)
        )
        blue_pixels = int(np.sum(blue_mask))

        if blue_pixels > 10:
            return True

        now = time.time()
        if now - last_press > 0.3:
            self.keyboard.press(Key.space)
            time.sleep(0.03)
            self.keyboard.release(Key.space)
            last_press = now

        time.sleep(0.1)

    print(f"⚠️  Game did not start within {timeout}s — continuing anyway")
    return False

  def reset(self, seed=None, options=None):
    self.step_count = 0
    super().reset(seed=seed)
    self.last_head_pos = None
    self.active_apple = None
    self.apples_eaten = 0
    self.direction = 0

    self.focus_chrome()
    time.sleep(0.3)

    self.keyboard.press(Key.space)
    time.sleep(0.05)
    self.keyboard.release(Key.space)
    time.sleep(0.3)

    self._wait_for_game_start(timeout=5.0)

    self.focus_chrome()
    time.sleep(0.2)

    self.keyboard.press(Key.right)
    time.sleep(0.05)
    self.keyboard.release(Key.right)
    time.sleep(0.3)

      # ─── DEBUG: snapshot channels right after game starts ───
    #self._save_debug_channels()

    raw_frame, processed_frame = self._capture_frame()
    self.stacked_frames = processed_frame
    return self.stacked_frames.copy(), {}

  def find_snake_head(self, frame):
    if frame is None or frame.size == 0:
      return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = (
        (hsv[:, :, 0] >= 100)
        & (hsv[:, :, 0] <= 130)
        & (hsv[:, :, 1] >= 100)
        & (hsv[:, :, 2] >= 100)
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
      largest_contour = max(contours, key=cv2.contourArea)
      M = cv2.moments(largest_contour)
      if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

  def is_snake_dead(self, raw_frame):
    # Disabled — blue-screen detection in game_over() is more reliable.
    return False

  def game_over(self, raw_frame):
    if raw_frame is None or raw_frame.size == 0:
      return False

    if self.is_snake_dead(raw_frame):
      return True

    finished_capture = np.array(self.capture.grab(self.replay_loc))
    bgr_replay = cv2.cvtColor(finished_capture, cv2.COLOR_BGRA2BGR)
    hsv_replay = cv2.cvtColor(bgr_replay, cv2.COLOR_BGR2HSV)

    blue_mask = (
        (hsv_replay[:, :, 0] >= 100)
        & (hsv_replay[:, :, 0] <= 130)
        & (hsv_replay[:, :, 1] >= 80)
        & (hsv_replay[:, :, 2] >= 80)
    )

    if np.mean(blue_mask) * 100 >= 80:
      print("GAME OVER (Blue Screen Detected)")
      return True

    return False

  def find_apple(self, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
      largest_contour = max(contours, key=cv2.contourArea)
      M = cv2.moments(largest_contour)
      if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        apple_color = self.get_apple_color(frame, cx, cy)
        return (cx, cy, apple_color)
    return None

  def get_apple_color(self, frame, x, y, radius=5):
    roi = frame[
        max(0, y - radius) : y + radius, max(0, x - radius) : x + radius
    ]
    if roi.size == 0:
      return None
    avg_color = np.mean(roi, axis=(0, 1)).astype(int)
    return tuple(avg_color)

  def check_apple_eaten(self, frame, apple_x, apple_y, snake_head_x=None, snake_head_y=None, radius=12):
    if snake_head_x is not None and snake_head_y is not None:
        distance = np.sqrt((snake_head_x - apple_x)**2 + (snake_head_y - apple_y)**2)
        if distance < 18:
            return True

    h, w, _ = frame.shape
    ymin, ymax = max(0, int(apple_y) - radius), min(h, int(apple_y) + radius + 1)
    xmin, xmax = max(0, int(apple_x) - radius), min(w, int(apple_x) + radius + 1)

    roi = frame[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        return True

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixel_count = np.sum(red_mask > 0)
    return red_pixel_count < 5

  def step(self, action):
    direction_shifts = [-1, 0, 1]
    self.direction = (self.direction + direction_shifts[action]) % 4

    dir_to_key = {0: Key.right, 1: Key.down, 2: Key.left, 3: Key.up}
    target_key = dir_to_key[self.direction]

    if action != 1:
      self.keyboard.press(target_key)
      time.sleep(0.02)
      self.keyboard.release(target_key)

    time.sleep(STEP_TIME)

    raw_frame, processed_frame = self._capture_frame()
    self.stacked_frames = processed_frame
    stacked_obs = processed_frame.copy()

    self.step_count += 1

    terminated = self.game_over(raw_frame)
    if terminated:
      self.active_apple = None
      self.last_head_pos = None
      return stacked_obs, -5.0, True, False, {}

    reward = 0.0
    curr_head = self.find_snake_head(raw_frame)
    prev_head = getattr(self, "last_head_pos", None)
    self.last_head_pos = curr_head

    if self.active_apple is not None:
        ax, ay, acolor = self.active_apple
        if self.check_apple_eaten(raw_frame, ax, ay):
            if curr_head is not None and abs(curr_head[0] - ax) < 30 and abs(curr_head[1] - ay) < 30:
                self.apples_eaten += 1
                if self.apples_eaten == 1:
                    reward = 10.0
                else:
                    reward = 30.0
                print(f"🎉 APPLE EATEN #{self.apples_eaten}")
                self.active_apple = None

    if self.active_apple is None:
        apple_info = self.find_apple(raw_frame)
        if apple_info is not None:
            self.active_apple = apple_info

    return stacked_obs, reward, False, False, {}


# ==================== 2. DUELING DQN NETWORK ====================
class DQNNetwork(nn.Module):

  def __init__(self, input_shape=INPUT_SHAPE, action_size=ACTION_SIZE):
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
    )

    with torch.no_grad():
      dummy_input = torch.zeros(1, *input_shape)
      conv_out = self.conv(dummy_input)
      self.flatten_dim = conv_out.view(1, -1).size(1)

    self.val_fc = nn.Sequential(
        nn.Linear(self.flatten_dim, 256), nn.ReLU(), nn.Linear(256, 1)
    )

    self.adv_fc = nn.Sequential(
        nn.Linear(self.flatten_dim, 256),
        nn.ReLU(),
        nn.Linear(256, action_size),
    )

  def forward(self, x):
    feat = self.conv(x)
    feat = feat.view(feat.size(0), -1)

    val = self.val_fc(feat)
    adv = self.adv_fc(feat)

    return val + (adv - adv.mean(dim=1, keepdim=True))


# ==================== 3. REPLAY BUFFER ====================
'''
class ReplayBuffer:

  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
    state, action, reward, next_state, done = zip(
        *random.sample(self.buffer, batch_size)
    )
    return (
        np.array(state, dtype=np.float32),
        np.array(action, dtype=np.int64),
        np.array(reward, dtype=np.float32),
        np.array(next_state, dtype=np.float32),
        np.array(done, dtype=np.float32),
    )

  def state_dict(self):
    return list(self.buffer)

  def load_state_dict(self, buffer_list):
    self.buffer = deque(buffer_list, maxlen=self.buffer.maxlen)

  def __len__(self):
    return len(self.buffer)

'''
class ReplayBuffer:

  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)
    self.priorities = deque(maxlen=capacity)
    self.max_priority = 1.0

  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))
    # New transitions get max priority so they're sampled at least once
    self.priorities.append(self.max_priority)

  def sample(self, batch_size):
    if len(self.buffer) == 0:
      return None

    priorities = np.array(self.priorities, dtype=np.float32)
    probs = priorities ** PER_ALPHA
    probs /= probs.sum()

    indices = np.random.choice(len(self.buffer), batch_size, p=probs)
    samples = [self.buffer[i] for i in indices]

    # Importance sampling weights (correct the bias)
    total = len(self.buffer)
    weights = (total * probs[indices]) ** (-PER_BETA)
    weights /= weights.max()
    weights = np.array(weights, dtype=np.float32)

    state, action, reward, next_state, done = zip(*samples)
    return (
        np.array(state, dtype=np.float32),
        np.array(action, dtype=np.int64),
        np.array(reward, dtype=np.float32),
        np.array(next_state, dtype=np.float32),
        np.array(done, dtype=np.float32),
        indices,
        weights,
    )

  def update_priorities(self, indices, td_errors):
    for idx, td in zip(indices, td_errors):
      p = abs(float(td)) + PER_EPS
      self.priorities[idx] = p
      if p > self.max_priority:
        self.max_priority = p

  def state_dict(self):
    return list(self.buffer), list(self.priorities)

  def load_state_dict(self, buffer_list, priority_list=None):
    self.buffer = deque(buffer_list, maxlen=self.buffer.maxlen)
    if priority_list is not None:
      self.priorities = deque(priority_list, maxlen=self.priorities.maxlen)
    else:
      self.priorities = deque([1.0] * len(self.buffer),
                              maxlen=self.priorities.maxlen)

  def __len__(self):
    return len(self.buffer)

# ==================== 4. DQN AGENT ====================
class DQNAgent:

  def __init__(self):
    self.action_size = ACTION_SIZE

    if torch.backends.mps.is_available():
      self.device = torch.device("mps")
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    print(f"⚡ Training Hardware Device: {self.device}")

    self.policy_net = DQNNetwork().to(self.device)
    self.target_net = DQNNetwork().to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    
    self.n_step = N_STEP
    self.n_step_buffer = deque(maxlen=N_STEP)

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    self.memory = ReplayBuffer(REPLAY_MEMORY_SIZE)
    self.epsilon = EPSILON_START
    self.training_steps = 0
    self.best_eval_apples = -1.0
    self.best_recent_avg = -float("inf")

  def act(self, state):
    if random.random() < self.epsilon:
      return random.randrange(self.action_size)

    state_t = (
        torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    )
    with torch.no_grad():
      q_values = self.policy_net(state_t)
    if self.epsilon == 0.0:
        q = q_values.cpu().numpy()[0]
        print(f"    Q: L={q[0]:+.2f} S={q[1]:+.2f} R={q[2]:+.2f} → action {q.argmax()}")
    return q_values.argmax(dim=1).item()

  def save_checkpoint(self, filepath, episode, total_reward):
    checkpoint = {
        "episode": episode,
        "policy_net_state_dict": self.policy_net.state_dict(),
        "target_net_state_dict": self.target_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "epsilon": self.epsilon,
        "training_steps": self.training_steps,
        "total_reward": total_reward,
        "best_eval_apples": getattr(self, "best_eval_apples", -1.0),
        "best_recent_avg": getattr(self, "best_recent_avg", -float("inf")),
    }
    tmp_path = filepath + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, filepath)
    print(f"💾 Checkpoint saved: {filepath}")

  def load_checkpoint(self, filepath, override_epsilon=None):
    if not os.path.exists(filepath):
        print(f"⚠️ No checkpoint found at {filepath}. Starting from scratch.")
        return 1, -float("inf")

    try:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("   Starting from scratch instead.")
        return 1, -float("inf")

    self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.training_steps = checkpoint["training_steps"]

    # Restore best tracking from the checkpoint — don't reset it.
    self.best_eval_apples = checkpoint.get("best_eval_apples", -1.0)
    self.best_recent_avg = checkpoint.get("best_recent_avg", -float("inf"))
    print(f"   (best_eval_apples={self.best_eval_apples:.2f}, best_recent_avg={self.best_recent_avg:.2f})")
    '''
    if "memory_state_dict" in checkpoint:
        self.memory.load_state_dict(checkpoint["memory_state_dict"])
        print(f"🧠 Restored {len(self.memory)} transitions.")
    else:
        print("🧠 No buffer in checkpoint — starting with empty replay buffer.")
'''
    if "memory_state_dict" in checkpoint:
        buf_state = checkpoint["memory_state_dict"]
        if isinstance(buf_state, tuple):
            self.memory.load_state_dict(buf_state[0], buf_state[1])
        else:
            self.memory.load_state_dict(buf_state)
        print(f"🧠 Restored {len(self.memory)} transitions.")
    else:
        print("🧠 No buffer in checkpoint — starting with empty replay buffer.")
        
        
    if override_epsilon is not None:
        self.epsilon = override_epsilon
    else:
        self.epsilon = checkpoint.get("epsilon", EPSILON_START)

    start_episode = checkpoint.get("episode", 0) + 1
    best_reward = checkpoint.get("total_reward", -float("inf"))
    print(f"📂 Loaded checkpoint from {filepath} (Resuming at Episode {start_episode}, Epsilon: {self.epsilon:.3f})")
    return start_episode, best_reward
  '''
  def train_step(self):
    if len(self.memory) < BATCH_SIZE:
      return

    states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

    states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
    actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
    dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

    q_values = self.policy_net(states_t).gather(1, actions_t)

    with torch.no_grad():
      best_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
      next_q_values = self.target_net(next_states_t).gather(1, best_actions)
      #target_q_values = rewards_t + (GAMMA * next_q_values * (1.0 - dones_t))
      target_q_values = rewards_t + (GAMMA ** N_STEP) * next_q_values * (1.0 - dones_t)

    loss = F.smooth_l1_loss(q_values, target_q_values)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    self.training_steps += 1

    if self.training_steps % TARGET_UPDATE_FREQ == 0:
      self.target_net.load_state_dict(self.policy_net.state_dict())
'''
  def train_step(self):
      if len(self.memory) < BATCH_SIZE:
        return
    
      batch = self.memory.sample(BATCH_SIZE)
      if batch is None:
        return
    
      states, actions, rewards, next_states, dones, indices, weights = batch
    
      states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
      actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
      rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
      next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
      dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
      weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)
    
      q_values = self.policy_net(states_t).gather(1, actions_t)
    
      with torch.no_grad():
        best_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
        next_q_values = self.target_net(next_states_t).gather(1, best_actions)
        target_q_values = rewards_t + (GAMMA ** N_STEP) * next_q_values * (1.0 - dones_t)
    
      # Weighted loss (importance sampling correction)
      td_errors = (q_values - target_q_values).detach().cpu().numpy().flatten()
      loss = (weights_t * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()
    
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    
      # Update priorities with the TD-errors from this batch
      self.memory.update_priorities(indices, td_errors)
    
      self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
      self.training_steps += 1
    
      if self.training_steps % TARGET_UPDATE_FREQ == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
      
  def push_transition(self, state, action, reward, next_state, done):
    """Add transition to n-step buffer. When buffer is full, compute n-step
    return and push the aggregated transition into the replay buffer."""
    self.n_step_buffer.append((state, action, reward, next_state, done))

    # Not enough transitions yet
    if len(self.n_step_buffer) < self.n_step:
        return

    # Compute n-step discounted reward from the buffer
    R = 0.0
    for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
        R += (GAMMA ** i) * r
        if d:
            break

    # The aggregated transition:
    #   state = oldest state in buffer
    #   action = oldest action
    #   next_state = newest next_state
    #   done = True if any transition in the window was terminal
    s0 = self.n_step_buffer[0][0]
    a0 = self.n_step_buffer[0][1]
    ns_n = self.n_step_buffer[-1][3]
    done_n = any(t[4] for t in self.n_step_buffer)

    self.memory.push(s0, a0, R, ns_n, float(done_n))
    
  def flush_n_step(self):
    """Force-flush remaining transitions at episode end."""
    while len(self.n_step_buffer) > 0:
        R = 0.0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (GAMMA ** i) * r
            if d:
                break

        s0 = self.n_step_buffer[0][0]
        a0 = self.n_step_buffer[0][1]
        ns_n = self.n_step_buffer[-1][3]
        done_n = any(t[4] for t in self.n_step_buffer)

        self.memory.push(s0, a0, R, ns_n, float(done_n))
        self.n_step_buffer.popleft()


# ==================== 5. TRAINING LOOP ====================
if __name__ == "__main__":
  import time
  from collections import deque

  env = Snake()
  agent = DQNAgent()

  CHECKPOINT_DIR = "checkpoints"
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  LATEST_PATH = os.path.join(CHECKPOINT_DIR, "snake_dqn_latest.pth")
  BEST_PATH = os.path.join(CHECKPOINT_DIR, "snake_dqn_best.pth")
  RECENT_BEST_PATH = os.path.join(CHECKPOINT_DIR, "snake_dqn_recent_best.pth")

  # ─── Interactive menu ──────────────────────────────────────
  has_checkpoint = os.path.exists(LATEST_PATH)

  print("\n" + "=" * 55)
  print("  SNAKE DQN — TRAINING MODE")
  print("=" * 55)
  if has_checkpoint:
      print(f"  📂 Existing checkpoint found: {LATEST_PATH}")
  else:
      print("  ⚠️  No existing checkpoint found.")
  print("=" * 55)

  while True:
      print()
      print("  1) Continue training from latest checkpoint")
      print("  2) Start new training from scratch")
      prompt = input("Choose [1=continue / 2=fresh]: ").strip()
      if prompt in ("1", "2"):
          choice = prompt
          break
      print("Please enter 1 or 2.")

  while True:
      try:
          n_episodes = int(input("How many episodes to train? [e.g. 500]: ").strip())
          if n_episodes > 0:
              break
          print("Please enter a positive integer.")
      except ValueError:
          print("Please enter a valid integer.")

  # ─── Apply choice ──────────────────────────────────────────
  if choice == "1":
      if not has_checkpoint:
          print("❌ No checkpoint exists. Falling back to fresh start.")
          choice = "2"
      else:
          start_episode, max_reward = agent.load_checkpoint(LATEST_PATH, override_epsilon=0.10)
          print(f"♻️  Continuing from episode {start_episode}")

  if choice == "2":
      if has_checkpoint:
          archive_dir = os.path.join(CHECKPOINT_DIR, "archive")
          os.makedirs(archive_dir, exist_ok=True)
          stamp = time.strftime("%Y%m%d_%H%M%S")
          for p in (LATEST_PATH, BEST_PATH, RECENT_BEST_PATH):
              if os.path.exists(p):
                  base = os.path.basename(p).replace(".pth", f"_{stamp}.pth")
                  os.rename(p, os.path.join(archive_dir, base))
          print(f"📦 Archived previous checkpoints to {archive_dir}/")

      start_episode = 1
      max_reward = -float("inf")
      print("🆕 Starting fresh")

  end_episode = start_episode + n_episodes
  print(f"\n🚀 Training episodes {start_episode} -> {end_episode - 1}\n")

  print(f"{'Episode':>8} | {'Reward':>8} | {'Apples':>6} | {'Steps':>5} | "
        f"{'Epsilon':>7} | {'Buffer':>7} | {'Avg25 R':>8} | {'Avg25 🍎':>8}")
  print("-" * 88)

  rolling_rewards = deque(maxlen=25)
  rolling_apples = deque(maxlen=25)
  rolling_lengths = deque(maxlen=25)

  # ─── Main training loop ────────────────────────────────────
  for episode in range(start_episode, end_episode):
    state, _ = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    while not done:
      action = agent.act(state)
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      #agent.memory.push(state, action, reward, next_state, float(done))
      agent.push_transition(state, action, reward, next_state, float(done))
      agent.train_step()

      state = next_state
      total_reward += reward
      step_count += 1
      
    agent.flush_n_step()

    apples = env.apples_eaten
    rolling_rewards.append(total_reward)
    rolling_apples.append(apples)
    rolling_lengths.append(step_count)

    avg_r = sum(rolling_rewards) / len(rolling_rewards)
    avg_a = sum(rolling_apples) / len(rolling_apples)

    print(f"{episode:8d} | {total_reward:+8.2f} | {apples:6d} | "
          f"{step_count:5d} | {agent.epsilon:7.4f} | {len(agent.memory):7d} | "
          f"{avg_r:+8.2f} | {avg_a:8.2f}")

    # ─── Save latest every 5 episodes ───────────────────────
    if episode % 5 == 0:
      agent.save_checkpoint(LATEST_PATH, episode, total_reward)

    # ─── Save a "recent best" when 25-ep avg improves ───────
    if len(rolling_rewards) == 25:
      if avg_r > agent.best_recent_avg:
        agent.best_recent_avg = avg_r
        agent.save_checkpoint(RECENT_BEST_PATH, episode, total_reward)
        print(f"  🏆 NEW RECENT BEST: Avg25 R = {avg_r:.2f}, Avg25 🍎 = {avg_a:.2f}")

    # ─── Evaluation every 50 episodes ───────────────────────
    if episode > 0 and episode % 50 == 0:
      print(f"\n{'=' * 55}")
      print(f"  📊 EVAL @ Episode {episode} (greedy, 5 eps)")
      print(f"{'=' * 55}")

      backup_epsilon = agent.epsilon
      agent.epsilon = 0.0

      eval_apples_list = []
      eval_rewards_list = []

      for ev in range(5):
        state, _ = env.reset()
        ev_reward = 0
        ev_done = False
        while not ev_done:
          action = agent.act(state)
          next_state, reward, terminated, truncated, _ = env.step(action)
          ev_done = terminated or truncated
          state = next_state
          ev_reward += reward
        eval_apples_list.append(env.apples_eaten)
        eval_rewards_list.append(ev_reward)
        print(f"  Eval ep {ev+1}: reward {ev_reward:+.2f}, apples {env.apples_eaten}")

      agent.epsilon = backup_epsilon

      avg_eval_apples = sum(eval_apples_list) / len(eval_apples_list)
      avg_eval_reward = sum(eval_rewards_list) / len(eval_rewards_list)
      print(f"  → AVG: reward {avg_eval_reward:+.2f}, apples {avg_eval_apples:.2f}")

      if avg_eval_apples > agent.best_eval_apples:
          agent.best_eval_apples = avg_eval_apples
          agent.save_checkpoint(BEST_PATH, episode, avg_eval_reward)
          print(f"  🏆 NEW BEST EVAL MODEL (eval apples {avg_eval_apples:.2f})")
      else:
          print(f"  (no new best eval; current best = {agent.best_eval_apples:.2f})")

      print(f"{'Episode':>8} | {'Reward':>8} | {'Apples':>6} | {'Steps':>5} | "
            f"{'Epsilon':>7} | {'Buffer':>7} | {'Avg25 R':>8} | {'Avg25 🍎':>8}")
      print("-" * 88)
