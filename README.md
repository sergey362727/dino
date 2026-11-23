# T-Rex Runner + DQN

Папка содержит готового агента для игры https://wayou.github.io/t-rex-runner/ :

- `dino.py` — DQN-агент на Python (numpy + Playwright);
- `index.js` — локальная копия исходного кода игры, которой подменяется сетевой скрипт;
- `dqn_model.npz` — веса обученной модели;
- `requirements.txt`, `setup.sh`, `play.sh`, `train.sh`, `auto.sh` — зависимости и скрипты для Linux/macOS;
- `play.bat`, `train.bat`, `auto.bat` — скрипты для Windows.

## Запуск

Из этой папки:

```bat
python dino.py play
```

Режимы:

```bat
python dino.py train --steps 120000 --fresh   :: обучение с нуля
python dino.py play                           :: бесконечная игра по обученной модели
python dino.py auto                           :: сначала обучение, затем бесконечная игра
python dino.py play --speed 5                 :: ускоренное отображение
python dino.py play --speed 0                 :: максимально быстро, без задержек
```

Управление в `play`: `Ctrl+C` — остановить.

Для удобства можно использовать `train.bat`, `play.bat`, `auto.bat`.

## Linux / macOS

```bash
git clone https://github.com/sergey362727/dino.git
cd dino

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m playwright install chromium

# если не хватает системных библиотек:
sudo python3 -m playwright install-deps chromium

chmod +x *.sh
./play.sh
```

Доступные скрипты: `./setup.sh`, `./train.sh`, `./play.sh`, `./auto.sh`.
Эквивалентные команды:

```bash
python3 dino.py play
python3 dino.py train --steps 120000 --fresh
python3 dino.py auto --steps 120000
python3 dino.py play --speed 5
```

Если графического окружения нет (сервер, SSH), игра автоматически запустится
в headless-режиме. Принудительно это можно задать так:

```bash
DINO_HEADLESS=1 ./play.sh
```

Окно браузера открывается развёрнутым на весь экран, а игра сразу переходит в
arcade-режим — так же, как на сайте после нажатия `Space`. Маленькое поле
шириной 44 px больше не используется. При разворачивании/изменении размера окна
динозавр не останавливается.

`index.js` остаётся исходным файлом без изменений (SHA-256 совпадает с тем,
что был в `Downloads`). Все правки находятся только в `dino.py`: игра использует
штатные настройки из `index.js` (`SPEED=6`, `GRAVITY=0.6`, `CLEAR_TIME=3000`,
`MAX_SPEED=13`), а агент лишь подаёт действия в тот же движок.

## Проверка модели

После финальных правок (исходное время старта, 60 кадров/с) жадная политика
прошла тест на **100 000 кадров без единого столкновения**, лучший счёт —
**1 275 590** (около 28 минут игрового времени). Ранее до правок времени тест
на 300 000 кадров тоже дал 0 столкновений, лучший счёт 3 875 790.

## Как это работает

1. Playwright открывает страницу игры.
2. `index.js` из этой папки подменяет сетевой скрипт через `page.route(...)`.
3. Собственный `requestAnimationFrame` игры отключается; вместо него
   `window.__dino.step(action)` выполняет ровно один кадр. В режиме `play`
   выдерживаются ровно 60 кадров/с — скорость движения такая же, как на сайте.
4. Агент получает вектор признаков (положение динозавра, скорость, параметры
   ближайших препятствий, стек из 3 кадров) и выбирает одно из действий:
   `0` — ничего, `1` — прыжок, `2` — приседание/ускоренное падение.
5. Сеть — MLP (DQN-архитектура) с experience replay, target network и Adam.
   Обучение идёт по меткам надёжной эвристики-учителя (имитационный тёплый старт),
   чтобы получить устойчивую политику; `play` использует только жадный выбор
   Q-значений обученной сети.

## Зависимости

Уже установлены в системе: `numpy`, `playwright` (Chromium или Chrome).

Если браузер не находится, выполните:

```bat
python -m playwright install chromium
```
