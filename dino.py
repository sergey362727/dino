"""
DQN-агент для https://wayou.github.io/t-rex-runner/

Игра запускается через Playwright. Локальный index.js (из этой же папки)
подменяет сетевой скрипт игры, а игровой цикл переводится в пошаговый режим:
Python выбирает действие -> JS делает ровно один кадр -> Python получает
новое состояние. Это позволяет обучаться в сотни раз быстрее реального времени,
а затем запустить игру "бесконечно" в обычном темпе.

Режимы:
    python dino.py train   - только обучение
    python dino.py play    - только бесконечная игра по обученной модели
    python dino.py auto    - обучение, затем бесконечная игра (по умолчанию)

Зависимости: numpy, playwright (уже установлены). torch/tensorflow не нужны.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
GAME_URL = "https://wayou.github.io/t-rex-runner/"
LOCAL_JS = BASE_DIR / "index.js"
MODEL_FILE = BASE_DIR / "dqn_model.npz"

FRAME_MS = 1000.0 / 60.0
FRAME_STACK = 3
N_ACTIONS = 3                 # 0 - ничего, 1 - прыжок, 2 - приседание
MAX_EPISODE_STEPS = 4000      # длина отрезка жизни, затем делается новый reset

# Обучение
TRAIN_STEPS = 120_000
MIN_REPLAY = 2_000           # минимальный размер буфера перед первым обновлением
REPLAY_CAPACITY = 60_000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 5e-4
TARGET_SYNC_EVERY = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 25_000
EXPERT_STEPS = 120_000         # сначала копируем надёжную эвристику (без неё DQN ломает политику)
SAVE_EVERY = 10_000
SEED = 42

# Воспроизведение
PLAY_SPEED = 1.0              # 1.0 = реальное время, 0 = как можно быстрее
HEADLESS_TRAIN = True
HEADLESS_PLAY = False


# ---------------------------------------------------------------------------
# JS-мосты
# ---------------------------------------------------------------------------

# Псевдо-часы: переопределяем performance.now(), чтобы управлять deltaTime вручную.
INIT_SCRIPT = r"""
(() => {
  const originalNow = performance.now.bind(performance);
  window.__dinoNow = null;
  try {
    Object.defineProperty(performance, 'now', {
      configurable: true,
      value: function () {
        return window.__dinoNow === null ? originalNow() : window.__dinoNow;
      }
    });
  } catch (e) {
    window.__dinoTimePatchError = String(e);
  }
})();
"""

# Захватываем синглтон Runner, выключаем его собственный requestAnimationFrame
# и делаем функцию step(), которая выполняет ровно один кадр игры.
BRIDGE_SCRIPT = r"""
(() => {
  const R = window.Runner;
  const r = R && R.instance_;
  if (!r) return { ok: false, error: 'Runner.instance_ не найден' };

  if (r.raqId) {
    cancelAnimationFrame(r.raqId);
    r.raqId = 0;
  }
  r.updatePending = false;
  r.scheduleNextUpdate = function () { this.updatePending = false; };
  // Показываем игру так же, как после нажатия Space на сайте: убираем
  // сообщение "Press Space to start", раскрываем контейнер до полной ширины
  // и включаем arcade-режим с масштабированием под окно браузера.
  r.playIntro = function () {
    this.playingIntro = false;
    this.tRex.playingIntro = false;
    this.playing = true;
    this.activated = true;
    this.containerEl.style.webkitAnimation = '';
    this.containerEl.style.width = this.dimensions.WIDTH + 'px';
    this.containerEl.style.height = this.dimensions.HEIGHT + 'px';
    var box = document.getElementById('messageBox');
    if (box) box.style.visibility = 'hidden';
    document.body.classList.add(R.classes.ARCADE_MODE);
    this.setArcadeModeContainerScale();
  };
  r.playIntro();
  try { r.stopListening(); } catch (e) {}

  // При разворачивании окна движок сам вызывает stop() внутри adjustDimensions().
  // Возвращаем состояние игры после пересчёта размеров, чтобы динозавр не замирал.
  const adjustDimensionsOriginal = r.adjustDimensions.bind(r);
  r.adjustDimensions = function () {
    adjustDimensionsOriginal();
    if (!this.crashed) {
      this.playing = true;
      this.paused = false;
      this.raqId = 0;
      this.updatePending = false;
    }
    this.setArcadeModeContainerScale();
  };

  window.__dinoNow = 1000;
  const FRAME = 1000 / 60;

  function emptyObstacle() {
    return { kind: 0, type: 'NONE', dx: 2, dt: 2,
             width: 0, y: 0, height: 0, speedOffset: 0 };
  }

  function snapshot() {
    const t = r.tRex;
    const dinoWidth = t.ducking ? t.config.WIDTH_DUCK : t.config.WIDTH;
    const front = t.xPos + dinoWidth;

    function describe(o) {
      const type = o.typeConfig.type;
      const kind = type === 'CACTUS_SMALL' ? 1 :
                   type === 'CACTUS_LARGE' ? 2 : 3;
      const dx = o.xPos - front;
      const speed = Math.max(2, r.currentSpeed + (o.speedOffset || 0));
      return {
        kind: kind,
        type: type,
        dx: dx,
        dt: dx / speed,
        width: o.width,
        y: o.yPos,
        height: o.typeConfig.height,
        speedOffset: o.speedOffset || 0,

        // Полная геометрия нужна планировщику для точной проверки столкновений.
        xPos: o.xPos,
        size: o.size,
        typeWidth: o.typeConfig.width,
        boxes: o.collisionBoxes.map(function (b) {
          return { x: b.x, y: b.y, w: b.width, h: b.height };
        })
      };
    }

    const raw = (r.horizon.obstacles || []).slice(0, 4);
    const obstacles = raw.map(describe);
    while (obstacles.length < 3) obstacles.push(emptyObstacle());

    // В движке столкновение проверяется только с obstacles[0], пока тот
    // полностью не скрылся за динозавром. Поэтому "цель" - первое ещё не
    // пройденное препятствие, но в состоянии оставляем и сырой список.
    // Препятствие больше не опасно, когда его правый край зашёл за
    // левый край динозавра (xPos TREX = 44). Движок ещё держит его в
    // obstacles[0] до xPos+width<=0, но столкновение уже невозможно.
    const blockerRaw = raw.find(function (o) {
      return o.xPos + o.width > t.xPos + 2;
    }) || raw[0];
    const blocker = blockerRaw ? describe(blockerRaw) : emptyObstacle();

    return {
      dead: !!r.crashed,
      dist: r.distanceRan,
      score: Math.ceil(r.distanceRan),
      speed: r.currentSpeed,
      dino: {
        y: t.yPos,
        ground: t.groundYPos,
        heightAbove: t.groundYPos - t.yPos,
        vy: t.jumping ? t.jumpVelocity : 0,
        jumping: !!t.jumping,
        ducking: !!t.ducking,
        speedDrop: !!t.speedDrop,
        x: t.xPos,
        width: dinoWidth
      },
      obstacles: obstacles,
      engine: obstacles,
      blocker: blocker
    };
  }

  window.__dino = {
    reset: function () {
      window.__dinoNow += FRAME;
      r.crashed = false;
      r.paused = false;
      r.playing = true;
      r.playingIntro = false;
      r.activated = true;
      r.inverted = false;
      r.invertTimer = 0;
      r.invertTrigger = false;
      r.runningTime = 0;                        // как на сайте: препятствия через CLEAR_TIME (3 c)
      r.distanceRan = 0;
      r.currentSpeed = r.config.SPEED;
      r.time = window.__dinoNow;
      r.horizon.reset();
      r.tRex.reset();
      r.tRex.playingIntro = false;
      r.tRex.jumpCount = 2;                     // не даём сработать playIntro()
      r.gameOverPanel = null;
      return snapshot();
    },

    step: function (action) {
      const t = r.tRex;
      if (t.ducking && action !== 2) t.setDuck(false);

      if (action === 1) {
        if (!t.jumping) {
          t.startJump(r.currentSpeed);
          t.endJump();
        }
      } else if (action === 2) {
        // В игре кнопка "вниз" в воздухе ускоряет падение, а на земле - приседает.
        if (t.jumping) {
          t.setSpeedDrop();
        } else if (!t.ducking) {
          t.setDuck(true);
        }
      }

      window.__dinoNow += FRAME;
      r.update();
      return snapshot();
    },

    snapshot: snapshot
  };

  return { ok: true };
})()
"""


# ---------------------------------------------------------------------------
# Нейросеть DQN на numpy (MLP + experience replay + target network + Adam)
# ---------------------------------------------------------------------------

class DQN:
    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS,
                 lr: float = LEARNING_RATE, seed: int = SEED):
        rng = np.random.default_rng(seed)
        sizes = [state_dim, 256, 256, n_actions]
        self.sizes = sizes
        self.lr = lr
        self.n_actions = n_actions
        self.train_step = 0

        self.params = {}
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            self.params[f"W{i}"] = (
                rng.normal(0.0, math.sqrt(2.0 / fan_in), (fan_in, sizes[i + 1]))
                .astype(np.float32)
            )
            self.params[f"b{i}"] = np.zeros(sizes[i + 1], dtype=np.float32)

        self.target_params = self._clone(self.params)
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8

    @staticmethod
    def _clone(params):
        return {k: v.copy() for k, v in params.items()}

    @staticmethod
    def _forward(x, params):
        h = x
        caches = []
        last = len(params) // 2 - 1
        for i in range(last + 1):
            z = h @ params[f"W{i}"] + params[f"b{i}"]
            caches.append((h, z))
            h = np.maximum(z, 0.0) if i < last else z
        return h, caches

    def q_values(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return self._forward(x, self.params)[0][0]

    def act(self, state: np.ndarray, epsilon: float,
            rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def sync_target(self):
        self.target_params = self._clone(self.params)

    def _adam_step(self, grads):
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = 1.0 if norm <= 5.0 else 5.0 / max(norm, 1e-9)

        self.train_step += 1
        t = self.train_step
        for k, g in grads.items():
            g = np.clip(g * scale, -5.0, 5.0)
            self._m[k] = self._beta1 * self._m[k] + (1 - self._beta1) * g
            self._v[k] = self._beta2 * self._v[k] + (1 - self._beta2) * g * g
            mhat = self._m[k] / (1 - self._beta1 ** t)
            vhat = self._v[k] / (1 - self._beta2 ** t)
            self.params[k] += (-self.lr * mhat / (np.sqrt(vhat) + self._eps)).astype(np.float32)

    def imitate_batch(self, states, actions):
        """Тёплый старт: клонируем действия эвристики через softmax-кросс-энтропию."""
        q, caches = self._forward(states, self.params)
        q = q - q.max(axis=1, keepdims=True)
        exp_q = np.exp(q)
        probs = exp_q / np.sum(exp_q, axis=1, keepdims=True)
        probs[np.arange(len(actions)), actions] -= 1.0
        dq = probs / len(actions)

        grads = {}
        dout = dq
        for i in reversed(range(len(self.sizes) - 1)):
            h, z = caches[i]
            grads[f"W{i}"] = h.T @ dout
            grads[f"b{i}"] = dout.sum(axis=0)
            dh = dout @ self.params[f"W{i}"].T
            dout = dh * (caches[i - 1][1] > 0) if i > 0 else dh

        self._adam_step(grads)
        return float(np.mean(np.abs(dq)))

    def train_batch(self, states, actions, rewards, next_states, terminals):
        q, caches = self._forward(states, self.params)
        q_next = self._forward(next_states, self.target_params)[0]

        targets = q.copy()
        best_next = np.max(q_next, axis=1)
        targets[np.arange(len(actions)), actions] = (
            rewards + GAMMA * best_next * (1.0 - terminals.astype(np.float32))
        )

        # Huber loss по выбранным действиям
        diff = q - targets
        dq = np.clip(diff, -1.0, 1.0) / len(actions)

        grads = {}
        dout = dq
        for i in reversed(range(len(self.sizes) - 1)):
            h, z = caches[i]
            grads[f"W{i}"] = h.T @ dout
            grads[f"b{i}"] = dout.sum(axis=0)
            dh = dout @ self.params[f"W{i}"].T
            if i > 0:
                prev_z = caches[i - 1][1]
                dout = dh * (prev_z > 0)
            else:
                dout = dh

        self._adam_step(grads)
        return float(np.mean(np.abs(diff)))

    def save(self, path: Path = MODEL_FILE):
        data = {"state_dim": np.array(self.sizes[0])}
        for k, v in self.params.items():
            data[f"online_{k}"] = v
            data[f"target_{k}"] = self.target_params[k]
        data["train_step"] = np.array(self.train_step)
        np.savez_compressed(path, **data)
        print(f"Модель сохранена: {path}")

    @classmethod
    def load(cls, path: Path = MODEL_FILE):
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                state_dim = int(data["state_dim"])
                agent = cls(state_dim)
                for i in range(len(agent.sizes) - 1):
                    agent.params[f"W{i}"] = data[f"online_W{i}"].astype(np.float32)
                    agent.params[f"b{i}"] = data[f"online_b{i}"].astype(np.float32)
                    agent.target_params[f"W{i}"] = data[f"target_W{i}"].astype(np.float32)
                    agent.target_params[f"b{i}"] = data[f"target_b{i}"].astype(np.float32)
                if "train_step" in data:
                    agent.train_step = int(data["train_step"])
            print(f"Модель загружена: {path}")
            return agent
        except Exception as exc:  # битый/старый файл лучше пересоздать
            print(f"Не удалось загрузить модель ({exc}); начнём с нуля.")
            return None


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = SEED):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed + 1)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, terminal):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminals[i] = 1.0 if terminal else 0.0
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.terminals[idx])


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------

def _clip(x, lo=-2.0, hi=2.0):
    return float(np.clip(x, lo, hi))


def encode_state(snap: dict) -> np.ndarray:
    """Один кадр -> вектор признаков для сети."""
    d = snap["dino"]
    parts = [
        _clip(d["heightAbove"] / 60.0, -0.2, 1.5),
        _clip(d["vy"] / 12.0, -1.2, 1.0),
        1.0 if d["jumping"] else 0.0,
        1.0 if d["ducking"] else 0.0,
        _clip(snap["speed"] / 13.0, 0.0, 1.2),
    ]

    # 3 "сырых" препятствия + текущая цель (первое ещё не пройденное).
    for o in snap["obstacles"] + [snap["blocker"]]:
        kind = o["kind"]
        parts.extend([
            1.0 if kind == 1 else 0.0,
            1.0 if kind == 2 else 0.0,
            1.0 if kind == 3 else 0.0,
            _clip(o["dx"] / 300.0, -1.0, 2.0) if kind else 2.0,
            _clip(o["dt"] / 60.0, 0.0, 2.0) if kind else 2.0,
            _clip(o["width"] / 60.0, 0.0, 1.5),
            _clip(o["y"] / 150.0, 0.0, 1.2),
            _clip(o["height"] / 50.0, 0.0, 1.5),
            _clip(o["speedOffset"], -1.0, 1.0),
        ])

    # Явные признаки, на которых основана надёжная эвристика. Они делают
    # задачу клонирования почти линейной и убирают случайные ошибки сети
    # на границах порогов (главная причина редких столкновений).
    o = snap["blocker"]
    dx = o["dx"]
    need_duck = bool(o["kind"] == 3 and o["y"] <= 80.0)
    jump_window = bool(45.0 < dx < 95.0)
    parts.extend([
        1.0 if need_duck else 0.0,
        1.0 if (o["kind"] and not need_duck) else 0.0,
        1.0 if d["vy"] > 0.0 else 0.0,
        1.0 if dx < 105.0 else 0.0,
        1.0 if dx > 180.0 else 0.0,
        1.0 if jump_window else 0.0,
        1.0 if (jump_window and not d["jumping"]) else 0.0,
        1.0 if o["kind"] else 0.0,
    ])
    return np.asarray(parts, dtype=np.float32)


def compute_reward(prev: dict, cur: dict, action: int) -> float:
    if cur["dead"]:
        return -15.0

    reward = 0.01 + max(0.0, cur["dist"] - prev["dist"]) / 100.0
    reward -= (0.0, 0.008, 0.004)[action]

    # Формирующая награда: правильная поза относительно ближайшего
    # ещё не пройденного препятствия.
    o = cur["blocker"]
    d = cur["dino"]
    if o["kind"] and o["dx"] > 0.0 and o["dx"] < 150.0:
        need_duck = o["kind"] == 3 and o["y"] <= 80.0
        needed_height = 136.0 - o["y"] + 6.0   # низ динозавра выше верха кактуса

        if need_duck:
            posture_ok = d["ducking"]
        else:
            posture_ok = d["jumping"] and d["heightAbove"] >= needed_height

        if posture_ok:
            reward += 0.06
        elif action == (2 if need_duck else 1):
            reward += 0.04
        elif o["dx"] < 95.0:
            reward -= 0.08

        # Явно наказываем paranoid-приседание перед кактусом и прыжок под птеродактилем.
        if o["dx"] < 100.0:
            if not need_duck and d["ducking"]:
                reward -= 0.06
            if need_duck and d["jumping"]:
                reward -= 0.06

    return float(reward)


def expert_action(snap: dict, rng: np.random.Generator | None = None) -> int:
    """Простая эвристика для тёплого старта DQN (не используется в play)."""
    # Эвристика детерминированная: случайный шум портил бы метки для клонирования.
    o = snap["blocker"]
    d = snap["dino"]
    if not o["kind"]:
        return 0

    # Высокий птеродактиль (y=50/75): под ним нужно приседать.
    if o["kind"] == 3 and o["y"] <= 80.0:
        if d["jumping"] and d["vy"] > 0.0 and o["dx"] > 180.0:
            return 2          # быстрее приземлиться к СЛЕДУЮЩЕМУ птеродактилю
        return 2 if o["dx"] < 105.0 else 0

    # Кактус или низкий птеродактиль (y=100): нужен прыжок.
    if d["jumping"]:
        # Ускоряем приземление, только когда это уже следующее препятствие,
        # а то, под которое мы прыгали, позади. Иначе прыжок оборвётся слишком рано.
        if d["vy"] > 0.0 and o["dx"] > 180.0:
            return 2
        return 0

    if 45.0 < o["dx"] < 95.0:
        return 1
    return 0


class DinoEnv:
    def __init__(self, page):
        self.page = page
        self.frames = deque(maxlen=FRAME_STACK)
        self.prev = None

    def reset(self):
        snap = self.page.evaluate("() => window.__dino.reset()")
        self.prev = snap
        frame = encode_state(snap)
        self.frames.clear()
        for _ in range(FRAME_STACK):
            self.frames.append(frame)
        return self._state(), snap

    def _state(self):
        return np.concatenate(list(self.frames)).astype(np.float32)

    def step(self, action: int):
        snap = self.page.evaluate("(a) => window.__dino.step(a)", action)
        reward = compute_reward(self.prev, snap, action)
        self.prev = snap
        self.frames.append(encode_state(snap))
        return self._state(), reward, bool(snap["dead"]), snap


# ---------------------------------------------------------------------------
# Запуск браузера / страницы
# ---------------------------------------------------------------------------

def open_browser(headless: bool):
    if not LOCAL_JS.exists():
        raise FileNotFoundError(f"Не найден {LOCAL_JS}. Положите index.js рядом с dino.py")

    pw = sync_playwright().start()
    args = ["--mute-audio"]
    if headless:
        args.append("--disable-gpu")
    else:
        # Обычное окно браузера: как при переходе по ссылке, без маленького viewport.
        args.extend(["--start-maximized", "--window-size=1366,768"])
    launch_kwargs = {"headless": headless, "args": args}
    browser = None
    for channel in ("chrome", None):
        try:
            kwargs = dict(launch_kwargs)
            if channel:
                kwargs["channel"] = channel
            browser = pw.chromium.launch(**kwargs)
            break
        except Exception:
            continue
    if browser is None:
        browser = pw.chromium.launch(**launch_kwargs)

    if headless:
        page = browser.new_page(viewport={"width": 1280, "height": 800})
    else:
        page = browser.new_page(no_viewport=True)
    page.add_init_script(INIT_SCRIPT)

    js_source = LOCAL_JS.read_text(encoding="utf-8")

    def route_handler(route):
        route.fulfill(status=200, content_type="application/javascript", body=js_source)

    page.route("**/index.js", route_handler)
    page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_function("() => !!window.Runner && !!window.Runner.instance_", timeout=30_000)

    result = page.evaluate(BRIDGE_SCRIPT)
    if not result or not result.get("ok"):
        raise RuntimeError(f"Не удалось подключиться к игре: {result}")

    return pw, browser, page


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train(page, steps: int = TRAIN_STEPS, load_existing: bool = True):
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    first = page.evaluate("() => window.__dino.reset()")
    state_dim = len(encode_state(first)) * FRAME_STACK
    agent = DQN.load() if load_existing else None
    if agent is None or agent.sizes[0] != state_dim:
        agent = DQN(state_dim)
    agent.sync_target()

    buffer = ReplayBuffer(REPLAY_CAPACITY, state_dim)
    env = DinoEnv(page)
    state, _ = env.reset()

    ep_steps = 0
    episode = 0
    best_score = 0
    losses = []

    print(f"Обучение DQN: {steps} шагов, размер состояния {state_dim}")
    print("Действия: 0=ничего, 1=прыжок, 2=приседание")

    start = time.time()
    for step in range(1, steps + 1):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-step / EPSILON_DECAY)
        if step == EXPERT_STEPS + 1:
            agent.lr = LEARNING_RATE / 4.0   # аккуратный DQN-дообучатель после клонирования
            agent.sync_target()

        if step <= EXPERT_STEPS:
            action = expert_action(env.prev, rng)
        else:
            action = agent.act(state, epsilon, rng)
        next_state, reward, terminal, snap = env.step(action)

        # terminal=True только при реальном столкновении. Ограничение длины
        # эпизода — это truncating, поэтому target сеть всё равно бутстрепится.
        buffer.add(state, action, reward, next_state, terminal)
        state = next_state
        ep_steps += 1
        best_score = max(best_score, snap["score"])

        if buffer.size >= MIN_REPLAY:
            batch = buffer.sample(BATCH_SIZE)
            if step <= EXPERT_STEPS:
                losses.append(agent.imitate_batch(batch[0], batch[1]))
            else:
                losses.append(agent.train_batch(*batch))

        if step % TARGET_SYNC_EVERY == 0:
            agent.sync_target()

        if step % SAVE_EVERY == 0:
            agent.save()

        if step % 1000 == 0:
            avg_loss = float(np.mean(losses[-1000:])) if losses else 0.0
            elapsed = time.time() - start
            phase = "эксперт" if step <= EXPERT_STEPS else "DQN"
            print(
                f"step={step:6d} | {phase:7s} | eps={epsilon:.3f} | loss={avg_loss:.4f} | "
                f"эпизод={episode} | лучший счёт={best_score:5d} | "
                f"память={buffer.size:6d} | {elapsed:.1f}s"
            )

        if terminal or ep_steps >= MAX_EPISODE_STEPS:
            state, _ = env.reset()
            ep_steps = 0
            episode += 1

    agent.sync_target()
    agent.save()
    print(f"Обучение завершено за {time.time() - start:.1f}s. Лучший счёт: {best_score}")
    return agent


# ---------------------------------------------------------------------------
# Бесконечная игра
# ---------------------------------------------------------------------------

def play_forever(page, speed: float = PLAY_SPEED):
    env = DinoEnv(page)
    first = page.evaluate("() => window.__dino.reset()")
    state_dim = len(encode_state(first)) * FRAME_STACK

    agent = DQN.load()
    if agent is None or agent.sizes[0] != state_dim:
        print("Обученной модели нет или она не подходит. Сначала запустите: python dino.py train")
        return

    state, snap = env.reset()
    frame_interval = (FRAME_MS / 1000.0) / speed if speed > 0 else 0.0
    next_tick = time.perf_counter()
    total_frames = 0
    lives = 0
    best = 0

    print("Бесконечная игра запущена. Ctrl+C — остановить.")
    try:
        while True:
            action = int(np.argmax(agent.q_values(state)))
            # Штатный CLEAR_TIME = 3 c: пока препятствий нет, динозавр просто бежит
            # (как на исходном сайте), а не дёргается из-за случайных Q-значений.
            if not any(o["kind"] for o in snap["obstacles"]):
                action = 0

            next_state, _, terminal, snap = env.step(action)
            state = next_state
            total_frames += 1
            best = max(best, snap["score"])

            # Держим ровно 60 кадров/с с учётом времени самого шага, иначе
            # игра шла бы медленнее оригинала на стоимость вызова Playwright.
            if frame_interval:
                next_tick += frame_interval
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()

            if terminal:
                lives += 1
                print(f"Столкновение #{lives}: счёт {snap['score']}. Продолжаем...")
                state, snap = env.reset()
                next_tick = time.perf_counter()

            if total_frames % 6000 == 0:
                print(f"кадров: {total_frames} | Lifetime Best: {best} | смертей: {lives}")
    except KeyboardInterrupt:
        print(f"\nОстановлено. Всего кадров: {total_frames}, лучший счёт: {best}")


# ---------------------------------------------------------------------------

def can_show_window() -> bool:
    """Можно ли показать окно браузера (нужно для play на Linux с графикой)."""
    if os.environ.get("DINO_HEADLESS") == "1":
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def main():
    parser = argparse.ArgumentParser(description="DQN для T-Rex Runner")
    parser.add_argument("mode", nargs="?", default="auto",
                        choices=["train", "play", "auto"],
                        help="train - обучение, play - бесконечная игра, auto - оба режима")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--speed", type=float, default=PLAY_SPEED,
                        help="скорость отображения в режиме play (1 = реальное время)")
    parser.add_argument("--fresh", action="store_true",
                        help="не загружать существующую модель при обучении")
    args = parser.parse_args()

    show_play = args.mode in ("play", "auto") and can_show_window()
    headless = args.mode == "train" or not show_play
    if args.mode in ("play", "auto") and not show_play:
        print("Окно браузера недоступно — запускаем игру в headless-режиме.")

    pw, browser, page = open_browser(headless=headless)
    try:
        if args.mode in ("train", "auto"):
            train(page, steps=args.steps, load_existing=not args.fresh)
        if args.mode in ("play", "auto"):
            # После обучения переоткрываем браузер уже с видимым окном.
            if headless and show_play:
                page.close()
                browser.close()
                pw.stop()
                pw, browser, page = open_browser(headless=False)
            play_forever(page, speed=args.speed)
    finally:
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
