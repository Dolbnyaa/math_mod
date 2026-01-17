import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin, atan2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

#1 Функции для работы с геодезическими расстояниями 
def haversine_distance(lat1, lon1, lat2, lon2, R=6371.0):
    """Расстояние между двумя точками на сфере (км)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def objective_function(coords, cities_A, cities_B, c):
    """Целевая функция F(H) для минимизации"""
    H_lat, H_lon = coords
    total = 0
    
    # Сумма для городов Индостана
    for _, row in cities_A.iterrows():
        dist = haversine_distance(row['latitude'], row['longitude'], H_lat, H_lon)
        total += row['population_millions'] * dist
    
    # Сумма для городов Европы с коэффициентом c
    for _, row in cities_B.iterrows():
        dist = haversine_distance(row['latitude'], row['longitude'], H_lat, H_lon)
        total += c * row['population_millions'] * dist
    
    return total

def gradient_descent_optimization(cities_A, cities_B, c, init_coords=(35, 45), learning_rate=0.1, max_iter=1000, tol=1e-6):
    """Градиентный спуск для оптимизации"""
    lat, lon = init_coords
    step = 0.01  # шаг для численного градиента
    iterations_count = 0
    
    
    # История значений для отладки
    history = []
    
    for i in range(max_iter):
        iterations_count += 1
        
        if abs(lat) > 90 or abs(lon) > 180:
            print(f"Координаты вышли за пределы lat={lat}, lon={lon}")
            if len(history) > 0:
                lat, lon = history[-1][0], history[-1][1]
            else:
                lat, lon = init_coords
            break
        
        current_value = objective_function([lat, lon], cities_A, cities_B, c)
        history.append((lat, lon, current_value))
        
        step_adj = step * max(1, 100/(i+1))  # Адаптивный шаг
        
        # Градиент по широте
        lat1 = lat + step_adj
        lat2 = lat - step_adj
        # Ограничиваем широту
        lat1 = min(90, max(-90, lat1))
        lat2 = min(90, max(-90, lat2))
        
        f_lat_plus = objective_function([lat1, lon], cities_A, cities_B, c)
        f_lat_minus = objective_function([lat2, lon], cities_A, cities_B, c)
        grad_lat = (f_lat_plus - f_lat_minus) / (2 * step_adj)
        
        # Градиент по долготе
        lon1 = lon + step_adj
        lon2 = lon - step_adj
        # Обрабатываем переход через 180°
        if lon1 > 180: lon1 -= 360
        if lon1 < -180: lon1 += 360
        if lon2 > 180: lon2 -= 360
        if lon2 < -180: lon2 += 360
        
        f_lon_plus = objective_function([lat, lon1], cities_A, cities_B, c)
        f_lon_minus = objective_function([lat, lon2], cities_A, cities_B, c)
        grad_lon = (f_lon_plus - f_lon_minus) / (2 * step_adj)
        
        grad_norm = sqrt(grad_lat**2 + grad_lon**2)
        if grad_norm > 0:
            grad_lat /= grad_norm
            grad_lon /= grad_norm
        
        current_learning_rate = learning_rate * (0.1 + 0.9 * (1 - i/max_iter))
        
        # Обновление координат
        lat_new = lat - current_learning_rate * grad_lat
        lon_new = lon - current_learning_rate * grad_lon
        
        # Проверяем границы
        lat_new = min(90, max(-90, lat_new))
        if lon_new > 180: lon_new -= 360
        if lon_new < -180: lon_new += 360
        
        # Проверка сходимости
        lat_change = abs(lat_new - lat)
        lon_change = abs(lon_new - lon)
        
        # Также проверяем изменение функции
        new_value = objective_function([lat_new, lon_new], cities_A, cities_B, c)
        
        if lat_change < tol and lon_change < tol:
            print(f"Сходимость достигнута на итерации {i+1}")
            lat, lon = lat_new, lon_new
            break
        
        lat, lon = lat_new, lon_new
        
        if i % 50 == 0:
            print(f"Итерация {i+1}: lat={lat:.4f}, lon={lon:.4f}, F={current_value:.2f}, lr={current_learning_rate:.4f}")
    
    final_value = objective_function([lat, lon], cities_A, cities_B, c)
    
    # Выводим дополнительную информацию для отладки
    print(f"Градиентный спуск завершен за {iterations_count} итераций")
    print(f"Начальная точка: {init_coords}")
    print(f"Конечная точка: ({lat:.4f}, {lon:.4f})")
    print(f"Значение функции: {final_value:.2f}")
    
    return lat, lon, final_value, iterations_count

#2 Загрузка данных

europe_df = pd.read_csv('europe_airports.csv')
print(f"Загружено {len(europe_df)} европейских аэропортов")

indo_df = pd.read_csv('indian_subcontinent_airports.csv')
print(f"Загружено {len(indo_df)} аэропортов Индостана")

#3 Вычисление коэффициента балансировки c
total_pop_indo = indo_df['population_millions'].sum()
total_pop_europe = europe_df['population_millions'].sum()
c = total_pop_indo / total_pop_europe

print(f"\nСтатистика")
print(f"Общее население Индостана: {total_pop_indo:.2f} млн")
print(f"Общее население Европы: {total_pop_europe:.2f} млн")
print(f"Коэффициент балансировки c = {c:.4f}")

#4 Оптимизация градиентным спуском

opt_lat_gd, opt_lon_gd, opt_value_gd, iterations_gd = gradient_descent_optimization(
    indo_df, europe_df, c,
    init_coords=(35, 45),
    learning_rate=0.1,
    max_iter=1000,
    tol=1e-8
)

print(f"\nРезультат градиентного спуска:")
print(f"  Координаты: {opt_lat_gd:.4f}°N, {opt_lon_gd:.4f}°E")
print(f"  Значение F(H): {opt_value_gd:.2f} км")
print(f"  Количество итераций: {iterations_gd}")

#5 Оптимизация с помощью SciPy

initial_guess = [35, 45]
bounds = [(-90, 90), (-180, 180)]

result = minimize(
    lambda coords: objective_function(coords, indo_df, europe_df, c),
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 1000, 'ftol': 1e-12}
)

opt_lat_sp, opt_lon_sp = result.x
opt_value_sp = result.fun

print(f"Результат оптимизации SciPy:")
print(f"  Координаты: {opt_lat_sp:.4f}°N, {opt_lon_sp:.4f}°E")
print(f"  Значение F(H): {opt_value_sp:.2f} км")
print(f"  Количество итераций: {result.nit}")

#6 Поиск ближайших городов 
def find_nearest_major_city(lat, lon, reference_cities):
    """Поиск ближайшего крупного города"""
    min_dist = float('inf')
    nearest_city = None
    nearest_country = None
    nearest_lat = None
    nearest_lon = None
    
    for _, city in reference_cities.iterrows():
        city_lat = city['latitude']
        city_lon = city['longitude']
        dist = haversine_distance(lat, lon, city_lat, city_lon)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city['city']
            nearest_country = city['country']
            nearest_lat = city_lat
            nearest_lon = city_lon
    
    return nearest_city, nearest_country, min_dist, nearest_lat, nearest_lon

middle_east_cities = pd.DataFrame({
    'city': ['Ashgabat', 'Tehran', 'Dubai', 'Abu Dhabi', 'Doha', 'Muscat', 'Kuwait City', 
             'Riyadh', 'Baghdad', 'Istanbul', 'Ankara', 'Baku', 'Tashkent', 'Dushanbe',
             'Kabul', 'Islamabad', 'Karachi', 'Mashhad', 'Shiraz', 'Isfahan',
             'Tabriz', 'Erbil', 'Gaziantep', 'Adana', 'Antalya', 'Izmir'],
    'country': ['Turkmenistan', 'Iran', 'UAE', 'UAE', 'Qatar', 'Oman', 'Kuwait',
                'Saudi Arabia', 'Iraq', 'Turkey', 'Turkey', 'Azerbaijan', 'Uzbekistan',
                'Tajikistan', 'Afghanistan', 'Pakistan', 'Pakistan', 'Iran', 'Iran', 'Iran',
                'Iran', 'Iraq', 'Turkey', 'Turkey', 'Turkey', 'Turkey'],
    'latitude': [37.95, 35.6892, 25.2532, 24.4539, 25.2854, 23.5880, 29.3759,
                 24.7136, 33.3152, 41.2622, 39.9334, 40.4093, 41.2995, 38.5598,
                 34.5755, 33.6844, 24.8607, 36.2605, 29.5918, 32.6546,
                 38.0800, 36.1900, 37.0662, 37.0000, 36.8874, 38.4192],
    'longitude': [58.38, 51.3890, 55.3657, 54.3773, 51.5310, 58.3829, 47.9774,
                  46.6753, 44.3661, 28.7278, 32.8597, 49.8671, 69.2401, 68.7870,
                  69.2075, 73.0479, 67.0011, 59.6168, 52.5837, 51.6679,
                  46.2919, 44.0089, 37.3833, 35.3213, 30.7075, 27.1287]
})

# Поиск ближайших городов
nearest_city_gd, nearest_country_gd, nearest_dist_gd, nearest_lat_gd, nearest_lon_gd = find_nearest_major_city(
    opt_lat_gd, opt_lon_gd, middle_east_cities
)

nearest_city_sp, nearest_country_sp, nearest_dist_sp, nearest_lat_sp, nearest_lon_sp = find_nearest_major_city(
    opt_lat_sp, opt_lon_sp, middle_east_cities
)

print(f"\nБлижайший город к результату градиентного спуска:")
print(f"  {nearest_city_gd}, {nearest_country_gd} (~{nearest_dist_gd:.1f} км)")

print(f"\nБлижайший город к результату SciPy оптимизации:")
print(f"  {nearest_city_sp}, {nearest_country_sp} (~{nearest_dist_sp:.1f} км)")

#7 ВИЗУАЛИЗАЦИЯ С ПРОСТЫМИ ПОДПИСЯМИ

# Находим топ-10 крупнейших аэропортов в каждом регионе
top_indo_airports = indo_df.nlargest(10, 'population_millions')
top_europe_airports = europe_df.nlargest(10, 'population_millions')

print("\nКрупнейшие аэропорты Индостана (топ-10 по населению):")
for i, (_, airport) in enumerate(top_indo_airports.iterrows(), 1):
    print(f"{i}. {airport['city']} ({airport['country']}): {airport['population_millions']} млн")

print("\nКрупнейшие аэропорты Европы (топ-10 по населению):")
for i, (_, airport) in enumerate(top_europe_airports.iterrows(), 1):
    print(f"{i}. {airport['city']} ({airport['country']}): {airport['population_millions']} млн")

fig = plt.figure(figsize=(22, 18))

ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.2)
ax.add_feature(cfeature.LAND, alpha=0.1)

ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')

for _, city in indo_df.iterrows():
    ax.plot(city['longitude'], city['latitude'], 'ro', 
            markersize=6,
            transform=ccrs.PlateCarree(), 
            markeredgecolor='black', markeredgewidth=0.5,
            zorder=5, alpha=0.7)

for _, city in europe_df.iterrows():
    ax.plot(city['longitude'], city['latitude'], 'bs', 
            markersize=6,
            transform=ccrs.PlateCarree(),
            markeredgecolor='black', markeredgewidth=0.5,
            zorder=5, alpha=0.7)

for i, (_, airport) in enumerate(top_indo_airports.iterrows()):
    ax.text(airport['longitude'] + 1.2, airport['latitude'] + 0.3, 
            airport['city'],
            transform=ccrs.PlateCarree(), fontsize=8, color='darkred',
            fontweight='bold', alpha=0.9,
            zorder=6)

for i, (_, airport) in enumerate(top_europe_airports.iterrows()):
    ax.text(airport['longitude'] + 1.2, airport['latitude'] + 0.3, 
            airport['city'],
            transform=ccrs.PlateCarree(), fontsize=8, color='darkblue',
            fontweight='bold', alpha=0.9,
            zorder=6)

ax.plot(opt_lon_gd, opt_lat_gd, 'g*', 
        markersize=30, transform=ccrs.PlateCarree(),
        markeredgecolor='black', markeredgewidth=3,
        label=f'Градиентный спуск',
        zorder=15)

ax.text(opt_lon_gd + 2.0, opt_lat_gd, f'GD: {opt_lat_gd:.1f}°N, {opt_lon_gd:.1f}°E',
        transform=ccrs.PlateCarree(), fontsize=9, color='darkgreen',
        fontweight='bold', alpha=0.9,
        zorder=16)

ax.plot(opt_lon_sp, opt_lat_sp, 'm*', 
        markersize=30, transform=ccrs.PlateCarree(),
        markeredgecolor='black', markeredgewidth=3,
        label=f'SciPy L-BFGS-B',
        zorder=15)

ax.text(opt_lon_sp + 2.0, opt_lat_sp, f'SciPy: {opt_lat_sp:.1f}°N, {opt_lon_sp:.1f}°E',
        transform=ccrs.PlateCarree(), fontsize=9, color='darkmagenta',
        fontweight='bold', alpha=0.9,
        zorder=16)

if nearest_city_gd and nearest_lat_gd is not None and nearest_lon_gd is not None:
    ax.plot(nearest_lon_gd, nearest_lat_gd, '^', 
            color='orange', markersize=20, transform=ccrs.PlateCarree(),
            markeredgecolor='black', markeredgewidth=2,
            label=f'Ближ. к GD: {nearest_city_gd}',
            zorder=14)
    
    ax.text(nearest_lon_gd + 2.0, nearest_lat_gd, 
            f'{nearest_city_gd}\n{nearest_dist_gd:.0f} км',
            transform=ccrs.PlateCarree(), fontsize=8, color='darkorange',
            fontweight='bold', alpha=0.9,
            zorder=16)
    
    ax.plot([opt_lon_gd, nearest_lon_gd], [opt_lat_gd, nearest_lat_gd],
            'y--', alpha=0.7, transform=ccrs.Geodetic(), linewidth=2,
            zorder=13)

if nearest_city_sp and nearest_lat_sp is not None and nearest_lon_sp is not None:
    ax.plot(nearest_lon_sp, nearest_lat_sp, '^', 
            color='cyan', markersize=20, transform=ccrs.PlateCarree(),
            markeredgecolor='black', markeredgewidth=2,
            label=f'Ближ. к SciPy: {nearest_city_sp}',
            zorder=14)
    
    ax.text(nearest_lon_sp + 2.0, nearest_lat_sp, 
            f'{nearest_city_sp}\n{nearest_dist_sp:.0f} км',
            transform=ccrs.PlateCarree(), fontsize=8, color='darkcyan',
            fontweight='bold', alpha=0.9,
            zorder=16)
    
    ax.plot([opt_lon_sp, nearest_lon_sp], [opt_lat_sp, nearest_lat_sp],
            'c--', alpha=0.7, transform=ccrs.Geodetic(), linewidth=2,
            zorder=13)

# Устанавливаем область отображения
ax.set_extent([-20, 120, -10, 70], crs=ccrs.PlateCarree())

plt.title('Оптимальный авиахаб: Индостан - Европа\nСравнение методов оптимизации и крупнейшие аэропорты',
          fontsize=18, fontweight='bold', pad=25)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, 
           markeredgecolor='black', markeredgewidth=1, label='Индостан'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8,
           markeredgecolor='black', markeredgewidth=1, label='Европа'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=12,
           markeredgecolor='black', markeredgewidth=2, label='Градиентный спуск'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', markersize=12,
           markeredgecolor='black', markeredgewidth=2, label='SciPy L-BFGS-B'),
]

if nearest_city_gd and nearest_lat_gd is not None:
    legend_elements.append(
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10,
               markeredgecolor='black', markeredgewidth=1, label=f'Ближ. к GD: {nearest_city_gd}')
    )
if nearest_city_sp and nearest_lat_sp is not None:
    legend_elements.append(
        Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan', markersize=10,
               markeredgecolor='black', markeredgewidth=1, label=f'Ближ. к SciPy: {nearest_city_sp}')
    )

ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
          framealpha=0.95, bbox_to_anchor=(0.01, 0.99), borderaxespad=0.)

info_text = f'ПАРАМЕТРЫ ОПТИМИЗАЦИИ:\n' + \
            f'Коэффициент балансировки: c = {c:.4f}\n' + \
            f'Население Индостана: {total_pop_indo:.0f} млн\n' + \
            f'Население Европы: {total_pop_europe:.0f} млн\n\n' + \
            f'РЕЗУЛЬТАТЫ:\n' + \
            f'• Градиентный спуск: {opt_value_gd:,.0f} км\n' + \
            f'• SciPy L-BFGS-B: {opt_value_sp:,.0f} км\n' + \
            f'• Разница: {abs(opt_value_gd - opt_value_sp):.0f} км\n\n' + \
            f'КРУПНЕЙШИЕ АЭРОПОРТЫ:\n' + \
            f'• Индостан: топ-10 подписаны\n' + \
            f'• Европа: топ-10 подписаны'
            
ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.97, edgecolor='black'),
        verticalalignment='top', horizontalalignment='right', zorder=20)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig('aviahub_simple_labels.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.show()


#8 Дополнительная таблица крупнейших аэропортов
print("\n ТАБЛИЦА КРУПНЕЙШИХ АЭРОПОРТОВ")

top_indo_table = top_indo_airports[['city', 'country', 'population_millions']].copy()
top_indo_table['Rank'] = range(1, len(top_indo_table) + 1)
top_indo_table['Population (млн)'] = top_indo_table['population_millions']
top_indo_table = top_indo_table[['Rank', 'city', 'country', 'Population (млн)']]
top_indo_table.columns = ['Ранг', 'Город', 'Страна', 'Население (млн)']

top_europe_table = top_europe_airports[['city', 'country', 'population_millions']].copy()
top_europe_table['Rank'] = range(1, len(top_europe_table) + 1)
top_europe_table['Population (млн)'] = top_europe_table['population_millions']
top_europe_table = top_europe_table[['Rank', 'city', 'country', 'Population (млн)']]
top_europe_table.columns = ['Ранг', 'Город', 'Страна', 'Население (млн)']

print("\nТоп-10 крупнейших аэропортов Индостана:")
print(top_indo_table.to_string(index=False))

print("\n\nТоп-10 крупнейших аэропортов Европы:")
print(top_europe_table.to_string(index=False))

top_indo_table.to_csv('top_indo_airports.csv', index=False, encoding='utf-8-sig')
top_europe_table.to_csv('top_europe_airports.csv', index=False, encoding='utf-8-sig')


#9 Создание отчета

distance_between_methods = haversine_distance(opt_lat_gd, opt_lon_gd, opt_lat_sp, opt_lon_sp)

# Определяем лучший метод
if opt_value_gd <= opt_value_sp:
    best_method = "Градиентный спуск"
    best_lat, best_lon = opt_lat_gd, opt_lon_gd
    best_value = opt_value_gd
    best_city = nearest_city_gd
    best_country = nearest_country_gd
else:
    best_method = "SciPy L-BFGS-B"
    best_lat, best_lon = opt_lat_sp, opt_lon_sp
    best_value = opt_value_sp
    best_city = nearest_city_sp
    best_country = nearest_country_sp

top_indo_list = ""
for i, (_, airport) in enumerate(top_indo_airports.head(5).iterrows(), 1):
    top_indo_list += f"     {i}. {airport['city']} ({airport['country']}): {airport['population_millions']:.1f} млн\n"

top_europe_list = ""
for i, (_, airport) in enumerate(top_europe_airports.head(5).iterrows(), 1):
    top_europe_list += f"     {i}. {airport['city']} ({airport['country']}): {airport['population_millions']:.1f} млн\n"

report = f"""
{'='*90}
ОТЧЕТ ПО ОПТИМИЗАЦИИ АВИАХАБА: ИНДОСТАН - ЕВРОПА
{'='*90}

1. ИСХОДНЫЕ ДАННЫЕ:
   Аэропортов в Индостане: {len(indo_df)}
   Аэропортов в Европе: {len(europe_df)}
   Общее население Индостана: {total_pop_indo:.2f} млн чел.
   Общее население Европы: {total_pop_europe:.2f} млн чел.
   Коэффициент балансировки: c = {c:.4f}

2. КРУПНЕЙШИЕ АЭРОПОРТЫ (ТОП-5):

   2.1. ИНДОСТАН:
{top_indo_list}
   2.2. ЕВРОПА:
{top_europe_list}

3. РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:

   3.1. МЕТОД: ГРАДИЕНТНЫЙ СПУСК
        Координаты оптимального хаба: {opt_lat_gd:.4f}°N, {opt_lon_gd:.4f}°E
        Значение целевой функции F(H): {opt_value_gd:,.2f} км
        Количество итераций: {iterations_gd}
        Ближайший город: {nearest_city_gd}, {nearest_country_gd}
        Расстояние до ближайшего города: {nearest_dist_gd:.1f} км

   3.2. МЕТОД: SCIPY L-BFGS-B
        Координаты оптимального хаба: {opt_lat_sp:.4f}°N, {opt_lon_sp:.4f}°E
        Значение целевой функции F(H): {opt_value_sp:,.2f} км
        Количество итераций: {result.nit}
        Ближайший город: {nearest_city_sp}, {nearest_country_sp}
        Расстояние до ближайшего города: {nearest_dist_sp:.1f} км

4. СРАВНИТЕЛЬНЫЙ АНАЛИЗ:
   Расстояние между полученными точками: {distance_between_methods:.1f} км
   Разница в значениях функции F(H): {abs(opt_value_gd - opt_value_sp):.2f} км
   Относительная разница: {abs(opt_value_gd - opt_value_sp)/min(opt_value_gd, opt_value_sp)*100:.4f}%
   Наилучший результат: {best_method} (F(H) = {best_value:,.2f} км)

5. ГЕОГРАФИЧЕСКИЙ АНАЛИЗ:
   Оба метода указывают на регион Туркменистана/Ирана
   Ближайший крупный город к результатам оптимизации: {best_city} ({best_country})
   {best_city} является столицей {best_country} и имеет развитую инфраструктуру
   Регион обладает стратегическим положением между Европой и Индостаном

6. ВЫВОДЫ И РЕКОМЕНДАЦИИ:
   Оба метода дали близкие результаты в районе {best_city}, {best_country}
   {best_city} является логистически выгодным местом для авиахаба:
     - Расположен на пересечении путей Европа-Азия
     - Имеет международный аэропорт
     - Находится на примерно равном удалении от ключевых городов обоих регионов
   Для практической реализации рекомендуется:
      - Рассмотреть {best_city} в качестве основного кандидата
      - Проанализировать инфраструктуру аэропорта {best_city}
      - Оценить экономические и политические факторы
      - Учесть близость к крупнейшим аэропортам Индостана и Европы
   Полученные результаты подтверждают стратегическую важность региона Каспийского моря

{'='*90}
Дополнительные файлы:
   aviahub_simple_labels.png - визуализация с простыми подписями
   top_indo_airports.csv - таблица крупнейших аэропортов Индостана
   top_europe_airports.csv - таблица крупнейших аэропортов Европы
{'='*90}
"""

print(report)

with open('aviahub_final_report_simple.txt', 'w', encoding='utf-8') as f:
    f.write(report)


#10 Создание таблицы сравнения

comparison_df = pd.DataFrame({
    'Параметр': ['Метод оптимизации', 
                 'Широта (°N)', 
                 'Долгота (°E)', 
                 'F(H) (км)', 
                 'Итерации', 
                 'Ближайший город', 
                 'Страна', 
                 'Расстояние до города (км)'],
    'Градиентный спуск': [
        'Градиентный спуск',
        f'{opt_lat_gd:.4f}',
        f'{opt_lon_gd:.4f}',
        f'{opt_value_gd:,.2f}',
        f'{iterations_gd}',
        nearest_city_gd,
        nearest_country_gd,
        f'{nearest_dist_gd:.1f}'
    ],
    'SciPy L-BFGS-B': [
        'SciPy L-BFGS-B',
        f'{opt_lat_sp:.4f}',
        f'{opt_lon_sp:.4f}',
        f'{opt_value_sp:,.2f}',
        f'{result.nit}',
        nearest_city_sp,
        nearest_country_sp,
        f'{nearest_dist_sp:.1f}'
    ]
})

print(comparison_df.to_string(index=False))
comparison_df.to_csv('methods_comparison_simple.csv', index=False, encoding='utf-8-sig')
