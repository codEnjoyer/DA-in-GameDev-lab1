# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #4 выполнил(а):
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

## Цель работы
Ознакомиться с принципами работы перцептрона.

## Задание 1
### В проекте Unity реализовать перцептрон

Перенёс предоставленный код перцептрона в Unity

```C#
using System;
using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;
using Random = UnityEngine.Random;

[Serializable]
public class TrainingSet
{
    public double[] input;
    public double output;
}

public class Perceptron : MonoBehaviour
{
    // public TextMeshProUGUI text;
    public TrainingSet[] ts;
    private readonly double[] weights = { 0, 0 };
    private double bias;
    private double totalError;

    private double DotProductBias(double[] v1, double[] v2)
    {
        if (v1 == null || v2 == null)
            return -1;

        if (v1.Length != v2.Length)
            return -1;

        var d = v1.Select((t, x) => t * v2[x]).Sum();

        d += bias;

        return d;
    }

    public int CalcOutput(int i1, int i2)
    {
        double[] inp = { i1, i2 };
        var dp = DotProductBias(weights, inp);
        return dp > 0 ? 1 : 0;
    }

    private int CalcOutput(int i)
    {
        var dp = DotProductBias(weights, ts[i].input);
        return dp > 0 ? 1 : 0;
    }

    private void InitialiseWeights()
    {
        for (var i = 0; i < weights.Length; i++)
            weights[i] = Random.Range(-1.0f, 1.0f);
        
        bias = Random.Range(-1.0f, 1.0f);
    }


    private void UpdateWeights(int j)
    {
        var error = ts[j].output - CalcOutput(j);
        totalError += Mathf.Abs((float)error);
        
        for (var i = 0; i < weights.Length; i++)
            weights[i] += error * ts[j].input[i];
        
        bias += error;
    }

    private void Train(int epochs)
    {
        InitialiseWeights();

        for (var e = 0; e < epochs; e++)
        {
            totalError = 0;
            for (var t = 0; t < ts.Length; t++)
                UpdateWeights(t);
            /*
                Debug.Log("W1: " + (weights[0]) + " W2: " + (weights[1]) + " B: " + bias);
            */
            Debug.Log($"Total error: {totalError}, epoch {e}.");
        }
    }

    private void Awake()
    {
        Train(8);
    }
```

Реализовал логические операции:
OR - 

![image](https://user-images.githubusercontent.com/87475288/204602696-e4a77135-109e-4e7f-8635-e46002d5431c.png)

AND - 

![image](https://user-images.githubusercontent.com/87475288/204602785-dcf410b4-3dfb-40c8-8c50-91ca4d80a6c6.png)

NAND - 

![image](https://user-images.githubusercontent.com/87475288/204602835-6b4d6bcf-6fef-4258-89ba-c056bb25b10b.png)

XOR - 

![image](https://user-images.githubusercontent.com/87475288/204602891-327ce755-ef0a-4a01-a7e0-4b763bb62869.png)

Как видите, перцептрон не осилил XOR. Наглядная причина этому:

![image](https://user-images.githubusercontent.com/87475288/204603203-2f2db54a-d894-4704-a30f-bd97edb1598f.png)

## Задание 2: Построить графики зависимости количества эпох от ошибки.

Перенёс данные со скриншотов в Google Sheets в формате - "номер эпохи : total error".
Вот результаты:

![image](https://user-images.githubusercontent.com/87475288/204604583-cf048753-ae90-4d48-a38a-b979c2ac15db.png)

![image](https://user-images.githubusercontent.com/87475288/204604635-f2c05aa1-44f3-4e4e-8b43-fd7bd115fcb7.png)

![image](https://user-images.githubusercontent.com/87475288/204604646-2772467e-f194-44ae-a2db-a01f252a22c4.png)

![image](https://user-images.githubusercontent.com/87475288/204604658-239043b6-a1b4-4926-b6be-60727e0dc5d8.png)

На графиках прослеживается зависимость успешности обучения от количества эпох, что вполне очевидно. Можно с уверенностью сказать, что количество эпох, в течение которых будет обучаться перцептрон, напрямую зависит от сложности ожидаемой операции. Так, например, на OR у перцептрона уходило 2 или 3 эпохи, на AND от 2 до 5, а с XOR он так и не справился.

## Задание 3
### Визулазировать работу перцептрона в Unity.

Решил визуализировать так: в случае, если результат логической операции равен нулю, куб отправляется в полёт.
Создал канвас, на который добавил текст с вводимыми данными. Добавил куб, на который в своб очеред добавил скрипт, контролирующий результат работы перцептрона на этих данных и решающий, что сделать с кубом, в зависимости от полученных данных.

Сам скрипт:
```C#
using UnityEngine;

public class Perceptronnoe : MonoBehaviour
{
    private Rigidbody _rb;
    [SerializeField] private int firstValue;
    [SerializeField] private int secondValue;
    [SerializeField] private float force;
    [SerializeField] private Perceptron perceptron;
    private bool _goesFar;

    void Start()
    {
        _rb = GetComponent<Rigidbody>();
        _goesFar = perceptron.CalcOutput(firstValue, secondValue) != 0;
    }

    private void FixedUpdate()
    {
        if (_goesFar)
        {
            _rb.AddForce(_rb.transform.TransformDirection(Vector3.forward) * force, ForceMode.Impulse);
        }
    }
}

```

Результат работы, после обучения перцептрона операции NOR:

![NOR_gif](https://user-images.githubusercontent.com/87475288/204607536-ef86d6c1-b680-4c8c-af3d-24a6ad49e71e.gif)

## Выводы

Перцептрон это один из переходных инструментов в создании нейронных сетей, со своими слабыми местами, которые были исправлены в нейронах. Мне понравилось работать с перцептроном, это очень хорошо помогает разобраться с тем, как работают нейронные сети, и заинтересовывает на изучении данной темы на более углубленном уровне.

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
