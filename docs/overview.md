# Overview: BUY / SELL / HOLD TA model (Binance 4h)

This project trains and serves a **multiclass** model that outputs exactly what you need:

- **signal**: BUY / SELL / HOLD
- **confidence**: \(0..1\)
- **probs**: \(P(BUY), P(SELL), P(HOLD)\)
- **as_of**: candle timestamp

## Data

- Source: Binance Spot klines (4h).
- Storage: local parquet per symbol in `data/<SYMBOL>_4h.parquet`.
- Updates: incremental pagination, newest candles appended by timestamp.

## Features (leakage-safe)

Features are computed **only from historical candles up to time \(t\)**, then the model uses the last row.

Current v1 feature set (see `src/model_tech/features/indicators.py`):

- log returns: `r_1`, `r_6`
- `RSI(14)`
- `MACD_hist`
- `ADX(14)`
- mean reversion: `(Close - EMA50) / EMA50`
- volatility: `ATR14/Close`, realized vol (rolling std of log returns)
- Bollinger: `percentB`, `width`
- volume: `log_volume`, `vol_rel = volume / SMA(volume,30)`
- optional: `CMF(20)`, `OBV`

Additionally, `symbol` is passed as a **categorical** feature to allow training a single multi-asset model.

## Labels (target)

Horizon is fixed:

- \(H = 6\) bars (1 day for 4h bars)
- \(r_{t,H} = Close_{t+H} / Close_t - 1\)

Class rule:

- BUY if \(r_{t,H} > +\\theta\)
- SELL if \(r_{t,H} < -\\theta\)
- HOLD otherwise

## Training

Model: `CatBoostClassifier` with multiclass objective.

Validation: **walk-forward** by time bars (no shuffling).

Parameter selection:

- `theta` is selected by grid search within `[theta_min, theta_max]` and filtered by HOLD share bounds.
- quality metric: mean **macro-F1** on validation windows.

Artifacts are written into `artifacts/` and used by inference.

## Inference rule

The model outputs class probabilities. Decision rule:

- if `max_prob < min_conf` → **HOLD**
- else `argmax(prob)`

`min_conf` is tuned on the most recent validation window and saved to artifacts.


### 1. Как работает сервис (Core Logic)

Сервис реализует **ML-подход (Machine Learning)** к техническому анализу для прогнозирования движения цены на **24 часа вперед** (горизонт 6 свечей по 4 часа).

- **Модель:** Градиентный бустинг **CatBoost** (Classifier).
- **Тип задачи:** Мультиклассовая классификация (3 класса: `BUY`, `SELL`, `HOLD`).
- **Признаки (Features):** Классические технические индикаторы (RSI, MACD, ADX, Bollinger Bands, ATR), но **нормализованные** (относительные значения, а не абсолютные цены), что позволяет модели работать стабильно при любых ценах актива.
- **Валидация:** Строгий **Walk-Forward** (тестирование на будущем) с **защитным гэпом** (gap), чтобы исключить подглядывание в будущее (data leakage).

### 2. Структура хранения (Data & Artifacts)

Сервис работает локально с файловой системой, не требуя внешней БД.

- **`data/{SYMBOL}_4h.parquet`**:
  - Здесь хранятся "сырые" свечи (OHLCV).
  - Формат Parquet (быстрый, сжатый).
  - Обновляется инкрементально (дописываются только новые свечи).
- **`artifacts/models/{SYMBOL}/`**:
  - Здесь лежит результат обучения ("мозги" модели).
  - `model.cbm`: Бинарный файл обученной модели CatBoost.
  - `inference.json`: Правила для принятия решений (пороги уверенности `min_conf`).
  - `feature_schema.json`: Список используемых индикаторов (чтобы при прогнозе считать то же самое, что при обучении).
  - `metrics.json`: Отчеты о качестве модели (F1-score, точность).

### 3. API и CLI Интерфейс

Взаимодействие идет через консольную утилиту `model-tech` или HTTP API.

#### Основные команды:

1.  **`download`**: Скачивает исторические данные с Binance.
2.  **`train`**: Запускает полный цикл обучения и подбора параметров.
3.  **`predict`**: Выдает торговый сигнал для текущего момента.
4.  **`serve`**: Поднимает HTTP-сервер (FastAPI) для интеграции с другими ботами.

### 4. Различия параметров (Refresh & Train)

#### В команде `predict`: `refresh=True` vs `refresh=False`

Этот флаг отвечает за **актуальность данных** перед прогнозом.

- **`refresh=True` (Рекомендуется для Live):**
  1.  Сначала сервис стучится в Binance API.
  2.  Скачивает свежие свечи, которых нет в локальном файле.
  3.  Обновляет `.parquet`.
  4.  Считает индикаторы и выдает прогноз.
  - _Плюс:_ Прогноз всегда на последней цене. _Минус:_ Занимает время (сетевой запрос).
- **`refresh=False` (Default):**
  1.  Сервис берет **только** то, что уже лежит в папке `data/`.
  2.  Если вы не делали `download` неделю, прогноз будет недельной давности.
  - _Плюс:_ Мгновенно. _Минус:_ Данные могут быть устаревшими.

#### Логика обучения: `train` (команда) и режимы

В самом `predict` флага `train` нет. Обучение — это отдельный процесс.

- **Команда `train` (Mode: `Quality`):**
  - Глубокий перебор параметров порога входа (`theta`).
  - Используется для первоначального создания модели.
- **Команда `train` (Mode: `Fast`):**
  - Быстрый перебор (меньше кандидатов).
  - Подходит для автоматического регулярного переобучения (например, раз в неделю), чтобы модель адаптировалась к волатильности рынка.

### Итоговый пайплайн (Workflow)

1.  **Setup:** `model-tech download --symbols BTCUSDT` (создается `data/BTC...`)
2.  **Training:** `model-tech train --symbols BTCUSDT` (создается `artifacts/models/BTC...`)
3.  **Live Trading:** `model-tech predict --symbol BTCUSDT --refresh` (читает `data`, грузит `artifacts`, обновляет свечи, выдает `BUY/SELL`).

### Пример запроса прогноза по токену без обученной модели

```bash
/v1/прогноз?symbol=ETH&refresh=true&train=true&since_days_default=1095
```

### Сценарий

- **Состояние**: У вас есть модель, обученная на "BTC", расположенная в корневом каталоге "artifacts/" (или, строго говоря, если вы запустили "model-tech train" через CLI, она сохраняется как "глобальная" модель по умолчанию).

* **Запрос**: Вы хотите получить прогноз для `ETH`.

### Последовательность выполнения (пошагово)

1. **Обновление данных (`refresh=true`)**:

   - Сервис **синхронно** загружает с Binance историю ETH за 3 года ("1095" дней).
   - _Результат_: `data/ETHUSDT_4h.parquet` создан/обновлен. Это занимает несколько секунд.

2. **Проверка модели**:

   - Система проверяет, существует ли выделенная модель в `artifacts/models/ETHUSDT/`.
   - _Результат_: **Ложь** (пока не существует).

3. **Триггер обучения (`train=true`)**:

   - Поскольку отсутствует специальная модель ETH и вы попросили провести обучение, сервис отправляет **Фоновое задание**.
   - - Действие\*: Оно не требует завершения обучения. Оно помещает задачу в очередь и приступает к выполнению немедленно.
   - _Результат_: Вы получаете `job_id` в ответе JSON.

4. **Прогнозирование (резервный вариант)**:
   - Сервис пытается дать вам прогноз _ прямо сейчас_.
   - Поскольку модель ETH отсутствует, он возвращается к ** Глобальной модели** (той, которая была разработана для BTC).
   - - Логика\*: Программа берет новые данные о ETH, вычисляет индикаторы и вводит их в модель, обученную на основе BTC.
   - _Результат_: Вы получаете сигнал "ПОКУПАТЬ/ПРОДАВАТЬ/ УДЕРЖИВАТЬ", основанный на том, "как ETH выглядит глазами модели, обученной на BTC".

### Вывод (ответ в формате JSON)

Вы немедленно получите что-то вроде этого:

```json
{
  "symbol": "ETHUSDT",
  "signal": "HOLD",
  "confidence": 0.42,
  "model_id_used": "global",   <-- IMPORTANT: Used the BTC model
  "job_id": "job_12345..."     <-- IMPORTANT: Training started in background
}
```

### Как с этим справиться?

1. **Первый запрос**: Примите "глобальный" прогноз, но рассматривайте его как прокси-сервер. Сохраните `job_id`.
2. ** Подождите**: Опросите конечную точку состояния (или просто подождите ~1-2 минуты).
   - `ПОЛУЧИТЬ /v1/задания/{job_id}`
3. **Последующие запросы**:
   - Как только задание будет завершено, появится файл "артефакты/модели/ETHUSDT/model.cbm`.
   - При следующем вызове `/v1/predict?symbol=ETH` сервис увидит выделенную модель.
   - В ответе будет указано `"model_id_used": "ETHUSDT".
