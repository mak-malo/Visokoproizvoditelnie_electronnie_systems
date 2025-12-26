import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ==============================
# 1. ЗАГРУЗКА ДАННЫХ
# ==============================

FILE_PATH = "/content/drive/MyDrive/Google Диск/Магистратура/3 Семестр/Курсовая работа/2025-03-20_18-17-32_266_rcms.csv"
ROW_LIMIT = 25000

df = pd.read_csv(FILE_PATH)

if ROW_LIMIT is not None:
    df = df.iloc[:ROW_LIMIT]

required_columns = ["TIME_s", "BASE_1_Ω"]
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Нет столбца {col}")

time = df["TIME_s"].to_numpy(dtype=float)
base = df["BASE_1_Ω"].to_numpy(dtype=float)

# ==============================
# 1.1 СГЛАЖИВАНИЕ ПО ГАУССУ
# ==============================

# фильтрация высокочастотных помех
sigma = 5
base_smoothed = gaussian_filter1d(base, sigma=sigma)

# ==============================
# 2. ПОИСК И ФИЛЬТРАЦИЯ МИНИМУМОВ
# ==============================

def find_local_minima(signal):
    idx = []
    n = len(signal)

    for i in range(1, n - 1):
        if signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]:
            idx.append(i)

    return idx

def filter_minima_by_time(minima_idx, time, min_dt, max_dt):
    filtered = [minima_idx[0]]

    for idx in minima_idx[1:]:
        dt = time[idx] - time[filtered[-1]]

        if min_dt <= dt <= max_dt:
            filtered.append(idx)
        elif dt > max_dt:
            filtered.append(idx)

    return filtered

# минимальное и максимальное значение периода дыхания
MIN_PERIOD = 2.0
MAX_PERIOD = 6.0

minima_idx_raw = find_local_minima(base_smoothed)
minima_idx = filter_minima_by_time(
    minima_idx_raw, time, MIN_PERIOD, MAX_PERIOD
)

# ==============================
# 3. КУБИЧЕСКИЙ СПЛАЙН И BASELINE REMOVAL
# ==============================

if len(minima_idx) < 4:
    raise ValueError("Недостаточно минимумов для построения сплайна")

# отбрасываем участки до второго и после предпоследнего минимума
minima_idx_inner = minima_idx #[1:-1]

t_min = time[minima_idx_inner]
y_min = base[minima_idx_inner]

# кубический сплайн
cs = CubicSpline(t_min, y_min)

start_idx = minima_idx_inner[0]
end_idx = minima_idx_inner[-1]

baseline = cs(time[start_idx:end_idx + 1])

base_shifted = np.full_like(base, np.nan, dtype=float)
base_shifted[start_idx:end_idx + 1] = (base[start_idx:end_idx + 1] - baseline)

# ==============================
# 4. ВИЗУАЛИЗАЦИЯ
# ==============================

plt.figure(figsize=(18, 8))
plt.plot(time, base, label="BASE_1_Ω (оригинал)", alpha=0.6, color = 'blue')
plt.xlabel("TIME_s")
plt.ylabel("Импеданс")
plt.title("Оригинальный сигнал и базовая линия (cubic spline)")
plt.legend()
plt.grid(True)

plt.figure(figsize=(18, 8))
plt.plot(time, base_smoothed, label="BASE_1_Ω (сглаженный)", linewidth=2, color = 'orange')
plt.plot(time[start_idx:end_idx + 1], baseline, label="Baseline (кубический сплайн)", linewidth=2, color = 'green')
plt.scatter(time[minima_idx_inner], base[minima_idx_inner], color="red", s=30, label="используемые минимумы")
plt.xlabel("TIME_s")
plt.ylabel("Импеданс")
plt.title("Сглаженный сигнал и базовая линия (cubic spline)")
plt.legend()
plt.grid(True)

plt.figure(figsize=(18, 8))
plt.plot( time[start_idx:end_idx + 1], base_shifted[start_idx:end_idx + 1], label="BASE_1_Ω (после вычитания baseline)")
plt.xlabel("TIME_s")
plt.ylabel("Импеданс")
plt.title("Сигнал, приведённый к нулю (без краевых минимумов)")
plt.legend()
plt.grid(True)

plt.show()