# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Чашкин Никита Андреевич
- РИ210941

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Цель работы
Ознакомиться с основными операторами языка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello, World! на Python и Unity
![alt text](https://github.com/codEnjoyer/DA-in-GameDev-lab1/blob/main/Screenshots/SayHelloProgram.png?raw=true)
![alt text](https://github.com/codEnjoyer/DA-in-GameDev-lab1/blob/main/Screenshots/SayHelloOnGDrive.png?raw=true)
![alt text](https://github.com/codEnjoyer/DA-in-GameDev-lab1/blob/main/Screenshots/HelloWorldInUnity.png?raw=true)


## Задание 2
Подготовил данные, перенёс функции модели, потерь и оптимизации. Далее построил график по заданным значениям, используемый код: 

```py

In [ ]:
import numpy as np
from matplotlib import pyplot as plt


def model(a, b, x):
    return a * x + b


def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5 / num) * (np.square(prediction - y)).sum()


def optimize(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0 / num) * ((prediction - y) * x).sum()
    db = (1.0 / num) * (prediction - y).sum()
    a = a - Lr * da
    b = b - Lr * db
    return a, b


def iterate(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y)
    return a, b


x = [3, 21, 22, 34, 54, 34, 55, 67, 89, 99]
x = np.array(x)
y = [2, 22, 24, 65, 79, 82, 55, 130, 150, 199]
y = np.array(y)

a = np.random.rand(1)
b = np.random.rand(1)
print(f"a: {a}, b: {b}")
Lr = 0.0000001

a, b = iterate(a, b, x, y, 10_000)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(f"a: {a}, b: {b}, loss: {loss}")
plt.scatter(x, y)
plt.plot(x, prediction)
plt.show()

```

## Задание 3
### - Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.
### - Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.
Величина loss должна стремиться к нулю при увеличении числа итераций.

![alt text](https://github.com/codEnjoyer/DA-in-GameDev-lab1/blob/main/Screenshots/Linear_regression_5_iterations.png?raw=true)
![alt text](https://github.com/codEnjoyer/DA-in-GameDev-lab1/blob/main/Screenshots/Linear_regression_500_iterations.png?raw=true)
![alt text](https://github.com/codEnjoyer/DA-in-GameDev-lab1/blob/main/Screenshots/Linear_regression_10000_iterations.png?raw=true)

Параметр Lr отражает точность линейной регрессии(Lr - Linear regression). Так, при его уменьшении в 10 раз величина loss увеличивается в несколько раз.

![alt text](https://github.com/codEnjoyer/DA-in-GameDev-lab1/blob/main/Screenshots/Linear_regression_low_Lr.png?raw=true)

## Выводы
Получен опыт работы с библиотекой matplotlib, изучен алгоритм реализации линейной регрессии. В процессе написания кода также было выучено несколько полезных комбинаций клавиш для работы в PyCharm.

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
