# Multi-Dataset Keypoint Detection using Pretrained Models


Задача — найти предобученную и оценить её работу на 7 предоставленных датасетах. 
Каждый датасет содержит набор изображений с объектами, центры которых нужно детектировать.

### Датасеты
Изображения в основном содержат один объект, однако есть примеры, на которых
присутствует более одного объекта (обозначены как дополнительные). Не все изображения содержат аннотации к объектам — неаннотированные примеры игнорируются.

Характеристика датасетов:

| Название <br/>датасета |  Исходный<br/>размер   | Пустых <br/>json файлов | Дополнительных <br/>объектов в сумме | 
|------------------------|:----------------------:|:-----------------------:|:------------------------------------:|
| Squirrel's **Head**    |          9998          |          8486           |                  0                   |
| Squirrel's **Tail**    |          1123          |            0            |                  17                  |
| **Gemstone**           |          4188          |            0            |                  40                  |
| Koala's **Nose**       |          3992          |            9            |                  5                   |
| Owls **Head**          |          1911          |            1            |                  4                   |
| Seahorse's **Head**    |          4240          |            6            |                  24                  |
| Teddy Bear's **Nose**  |          4320          |           10            |                  3                   |

### Репозиторий содержит
- Код с замером метрик `main.py`
- Код для загрузки модели и обработки ее выходов `src/model.py`
- Код для загрузки датасета `src/data_loading.py`

### Решение

- Детектирование ключевых точек реализовано с помощью мультимодальной модели
[OWLv2](https://huggingface.co/docs/transformers/main/model_doc/owlv2).
Модель предназначена для определения ограничивающих прямоугольников объекта по
заданному текстовому описанию. Поэтому в качестве выхода модели берется центр предсказанного 
прямоугольника.
- В качестве текстового описания берется по одному слову для каждого датасета,
описывающему объект который нужно найти, например head.

### Упрощения
- Предсказания модели можно упорядочить по степени уверенности, порог которой следует контролировать.
Однако, согласно анализу, каждое изображение в основном содержит ровно один объект. Поэтому, чтобы не подбирать трешхолд для каждого датасета,
принято решение брать в качестве ответа прямуогольник с наибольшей степенью уверенности. 
- Если фильтровать выходы модели по трешхолду, может появиться большое число False Positives. Тогда
следует также учитывать метрики precision и recall.
- Для примеров, имеющих несколько точек в качестве ответа, берется первая по порядку, остальные считаются
дополнительными. Они учтены в посчитанной метрике.
- Если модель не определила объектов, то в качестве ответа берется координата верхнего левого угла.

### Инструкция по установке
1. ``` git clone https://github.com/DmitryKuznetsov1/keypoint_detection.git ```
2. ```cd keypoint_detection```
2. ```make setup```
3. Положить датасеты в папку tasks

### Запуск
Параметры:
1. ```DATASETS_PATH``` — путь до датасетов, по умолчанию ```tasks```.
2. ```BATCH_SIZE``` — размер батча, умолчанию ```2```.
3. ```LOAD_IMAGES``` — флаг, определяющий, загружать ли изображения сразу
в оперативную память при считывании датасета. По умолчанию ```0```.

* ```make eval DATASETS_PATH=arg1 BATCH_SIZE=arg2 LOAD_IMAGES=arg3```

### Что можно улучшить
- Подобрать трешхолд индивидуально для каждой задачи, чтобы в целом сделать модель более устойчивой
- Попробовать более легковесные модели, в том числе первую версию используемой
- Выгрузить и использовать веса модели в другом формате для более эффективного использования,
например в onnx.
- Хотя полученное решение универсально и не требует ничего кроме python и библиотек, установленных
с помощью pip, его можно обернуть в докер. 

### Результаты

Для получения результатов использовалась видеокарта GeForce 3090 TI 24Gb. Размер батча — 8–10 изображений.
Размер указан после вычитания примеров
с пустыми json и прибавления к размеру неучтенных точек.
Точность рассчитана как среднее по всем примерам, а не как среднее по датасетам, поскольку они не сбалансированы.

| Название датасета       |  Размер   |  Точность   |  Среднее относительное расстояние   |    Время     |
|-------------------------|:---------:|:-----------:|:-----------------------------------:|:------------:|
| Голова белки            |   1512    |    0.89     |                0.061                |    312.85    |
| Хвост белки             |   1140    |    0.23     |                0.207                |    237.17    |
| Драгоценный камень      |   4228    |    0.81     |                0.071                |    874.93    |
| Нос коалы               |   3998    |    0.94     |                0.034                |    848.03    |
| Голова совы             |   1914    |    0.89     |                0.062                |    384.70    |
| Голова морского конька  |   4258    |    0.85     |                0.073                |    864.01    |
| Нос медвежонка          |   4313    |    0.99     |                0.009                |    874.06    |
| **Все датасеты**        | **21363** |  **0.86**   |              **0.057**              | **4395.75**  |
