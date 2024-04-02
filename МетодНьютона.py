import numpy as np
import matplotlib.pyplot as plt

# Определяем целевую функцию
def f(x):
    x1, x2 = x
    return (x1 - 3)**2 + (x2 - 8)**2 + x1 * x2

# Определяем градиент целевой функции
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2 * (x1 - 3) + x2
    df_dx2 = 2 * (x2 - 8) + x1
    return np.array([df_dx1, df_dx2])

# Определяем гессиан целевой функции
def hessian_f(x):
    x1, x2 = x
    return np.array([[2, 1], [1, 2]])

# Реализуем метод Ньютона для поиска оптимума
def newton_method(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100):
    x_history = [x0]  # Храним историю точек для визуализации
    x = x0
    damping_factor = 1.0
    for iteration in range(max_iter):
        grad = grad_f(x)
        hessian = hessian_f(x)
        delta_x = -np.linalg.inv(hessian) @ grad * damping_factor
        new_x = x + delta_x

        # Проверка условия достижения оптимума
        if np.linalg.norm(grad) < tol:
            break

        # Проверка условия, при котором необходимо увеличить коэффициент
        if f(new_x) >= f(x):
            damping_factor *= 0.5  # Уменьшаем коэффициент демпфирования вдвое

        x = new_x
        x_history.append(x)  # Добавляем текущую точку в историю

    return x_history

# Начальное значение
x0 = np.array([-10.0, -10.0])

# Запуск метода Ньютона для нахождения оптимума с точностью 0.01
optimum_path = newton_method(f, grad_f, hessian_f, x0, tol=0.01)
optimum = optimum_path[-1]  # Получаем оптимальное значение как последнюю точку в истории

# Генерируем сетку точек для построения линий уровня
x1_values = np.linspace(-15, 15, 400)
x2_values = np.linspace(-14, 20, 400)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = f([X1, X2])

# Построение графика
plt.figure(figsize=(10, 8))
plt.contour(X1, X2, Z, levels=[10, 20, 30, 70, 100], colors='gray', linestyles='dashed')  # Линии уровня функции
plt.plot(optimum[0], optimum[1], 'ro')  # Точка оптимума
plt.plot(x0[0], x0[1], 'ro')  # Начальная точка
for i in range(1, len(optimum_path)):
    plt.plot([optimum_path[i-1][0], optimum_path[i][0]], [optimum_path[i-1][1], optimum_path[i][1]], color='red')  # Линии между точками
plt.axhline(0, color='black')  # Горизонтальная линия по центру
plt.axvline(0, color='black')  # Вертикальная линия по центру
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
print("Значение оптимума =", optimum)
