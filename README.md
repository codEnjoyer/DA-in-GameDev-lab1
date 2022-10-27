# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
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
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.
## Задание 1
### Реализовать систему машинного обучения в связке Python -Google-Sheets – Unity.

Создал пустой 3D проект:

![Empty_project](https://user-images.githubusercontent.com/87475288/198113172-dd6fcf44-71bb-4de0-bc2a-4ee0ce127cc7.png)

И добавил необходимые пакеты:

![Packages](https://user-images.githubusercontent.com/87475288/198113512-0f468974-1d7e-4f0c-b384-ddec53c53527.png)

При помощи Anaconda установил mlagents 0.28.0 и torch 1.8.2 (последняя релизная версия, команду для скачивания получил на оф. сайте, совместима с остальными пакетами):

![Conda_packages](https://user-images.githubusercontent.com/87475288/198113886-9f7a8a21-e665-49d6-bea9-4d8384e13540.png)

Добавил необходимые объекты на сцену и накинул скрипт на шар, будущий RollerAgent:

![first_step](https://user-images.githubusercontent.com/87475288/198114033-40c5c6f5-48ed-4b8d-80de-27c1b216dc02.png)

Добавил в этот скрипт следующий код (отрефакторенная версия кода из материалов):

```C#
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    private Rigidbody _rBody;
    [SerializeField] private Transform _target;
    private const float ForceMultiplier = 10;

    public void Start()
    {
        _rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        if (transform.localPosition.y < 0)
        {
            _rBody.angularVelocity = Vector3.zero;
            _rBody.velocity = Vector3.zero;
            transform.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
        }

        _target.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(_target.localPosition);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(_rBody.velocity.x);
        sensor.AddObservation(_rBody.velocity.z);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        _rBody.AddForce(controlSignal * ForceMultiplier);

        if (transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
        
    }
    
    private void OnTriggerEnter(Component other)
    {
        if (!other.TryGetComponent<Goal>(out var goal)) return;
        SetReward(1.0f);
        EndEpisode();
    }
}

```

Добавил компоненты шару для того, чтобы он стал агентом (прям Джеймс Бонд какой-то), и настроил соответствующим образом:

![second_step](https://user-images.githubusercontent.com/87475288/198114600-031e76f6-d167-4bd8-8daf-ef1963aac017.png)

Вставил в корень проекта конфиг нейронной сети из материалов и запустил обучение:

![third_step](https://user-images.githubusercontent.com/87475288/198114982-349849ca-4302-4412-b643-bccab62462de.png)

Далее поступил аналогичным образом для 3, 9 и 27 копий модели:

![3_areas](https://user-images.githubusercontent.com/87475288/198115143-06e21081-31a1-445d-b841-4c6170f787ca.png)

![9_areas](https://user-images.githubusercontent.com/87475288/198115163-2a15230c-3d08-4784-a56d-4cafbad70a3e.png)

![27_areas](https://user-images.githubusercontent.com/87475288/198115172-d105d8c6-4180-4885-b53c-1bef63ac8722.png)

По итогам обучения шар стал двигаться намного менее хаотично, реже падать за границу плоскости и практически всегда достигать цели в виде куба.

![start_work](https://user-images.githubusercontent.com/87475288/198115543-4c0e5367-b961-41f3-b40b-cbd035e42fae.png)

![end_work](https://user-images.githubusercontent.com/87475288/198115555-559116ea-aab9-45cb-9d98-310f99abefb1.png)


## Задание 2: Подробно опишите каждую строку файла конфигурации нейронной сети. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters

```
behaviors:
  RollerBall:                        #Имя агента
    trainer_type: ppo                #Устанавливаем режим обучения.
    hyperparameters:                 
      batch_size: 10                 #Количество опытов на каждой итерации.
      buffer_size: 100               #Количество опыта, которое нужно набрать для обновления
      learning_rate: 3.0e-4          #Шаг обучения.
      beta: 5.0e-4                   #Случайность действия; повышает разнообразие и иследованность пространства обучения.
      epsilon: 0.2                   #Порог расхождений между старой и новой политиками при обновлении.
      lambd: 0.99                    #Авторитетность оценок значений во времени. Чем выше значение, тем более авторитетен набор предыдущих оценок.
      num_epoch: 3                   #Количество проходов через буфер опыта при выполнении оптимизации.
      learning_rate_schedule: linear #Темп скорости обучения со временем.
    network_settings:                
      normalize: false               #Нормализация входных данных.
      hidden_units: 128              #Количество нейронов в скрытых слоях сети.
      num_layers: 2                  #Количество скрытых слоев для размещения нейронов.
    reward_signals:                  #Задает сигналы о вознаграждении.
      extrinsic:
        gamma: 0.99                  #Коэффициент скидки для будущих вознаграждений.
        strength: 1.0                #Шаг для learning_rate.
    max_steps: 500000                #Количество шагов, которые должны быть выполнены в среде для завершения обучения.
    time_horizon: 64                 #Количество циклов ML-агента, хранящихся в буфере до ввода в модель.
    summary_freq: 10000              #Количество необходимого опыта перед созданием и отображением статистики.
```

## Задание 3: Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

Задание я прочитал после того, как сделал его, так что кубы одного цвета :)

Сначала дописал логику награждения за достижение кубов (чем дальше он продвигается, тем большая награда даётся, а за полное выполнение задания я давал удвоенную награду). К тому же добавил поле, которое отвечает за то, сколько раз шар должен задеть кубы. Код выглядит следующим образом:

```C#
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    [SerializeField] private Transform _target;
    [SerializeField] private Transform _secondTarget;
    [SerializeField] private float _forceMultiplier = 10;
    [SerializeField] private int _targetTouches;
    private Rigidbody _rBody;
    private bool _firstTargetAchieved;
    private int _targetTouchCount;
    
    public void Start()
    {
        _rBody = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        if (transform.localPosition.y < 0)
        {
            _targetTouchCount = 0;
            _firstTargetAchieved = false;
            _rBody.angularVelocity = Vector3.zero;
            _rBody.velocity = Vector3.zero;
            transform.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
        }

        _target.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
        _secondTarget.localPosition = new Vector3(Random.Range(-4f, 4f), 0.5f, Random.Range(-4f, 4f));
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(_target.localPosition);
        sensor.AddObservation(_secondTarget.localPosition);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(_rBody.velocity.x);
        sensor.AddObservation(_rBody.velocity.z);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        _rBody.AddForce(controlSignal * _forceMultiplier);

        if (transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
        
    }
    
    private void OnTriggerEnter(Component other)
    {
        if (_targetTouches == _targetTouchCount)
        {
            AddReward(_targetTouchCount * 2);
            _targetTouchCount = 0;
            _firstTargetAchieved = false;
            EndEpisode();
        }
        if (other.TryGetComponent<FirstGoal>(out var g) && !_firstTargetAchieved)
        {
            _targetTouchCount++;
            _firstTargetAchieved = true;
            AddReward(_targetTouchCount);
        }
        if (other.TryGetComponent<SecondGoal>(out var goal) && _firstTargetAchieved)
        {
            _targetTouchCount++;
            _firstTargetAchieved = false;
            AddReward(_targetTouchCount);
        }
    }
}
```
Добавил побольше зон для обучения и запустил ml-агента:

![Learning_between_cubes](https://user-images.githubusercontent.com/87475288/198261073-20cf6a49-bc40-4cad-9c3e-220833749f96.png)

Результат обучения: шар катается по кругу на платформе, попутно стараясь зацепить кубы:

![Start_between_cubes](https://user-images.githubusercontent.com/87475288/198261239-1040256d-9140-4f6c-bd0f-bd53eb5d26b9.png)

![Goes_between_cubes](https://user-images.githubusercontent.com/87475288/198261252-286c8e24-8bad-46df-9fad-900d8f1ce3f8.png)

![Finish_between_cubes](https://user-images.githubusercontent.com/87475288/198261268-527327fc-a509-450b-ab6b-fe22d845da0a.png)

Ну а после того, как заденет кубы установленное перед запуском количество раз, зона обновляется, и всё начинается заново:

![Next_between_cubes](https://user-images.githubusercontent.com/87475288/198261411-3f6f7ae0-d6f1-4e2e-8668-38f1e997beec.png)


## Выводы
Игровой баланс в играх - это субъективное «равновесие» между персонажами, командами, тактиками игры и другими игровыми объектами. Игровой баланс является одним из требований к «честности» правил.

Относительно простой нейронной сети достаточно, чтобы достичь высокой эффективности игры против игроков и традиционного игрового ИИ. Таких агентов можно использовать различными способами, например, для тренировки новых игроков или для выявления неожиданных стратегий. Так же системы машинного обучения могут использоваться для того, чтобы выявить дисбаланс в игре.

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
