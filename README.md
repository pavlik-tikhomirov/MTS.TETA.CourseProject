# MTS.TETA.CourseProject
Оглавление
1. Введение
2. Аннотация
3. Мой вклад в проект

## Введение
Летом 2021 (июль-август) была организована летняя школа МТС.ТЕТА по направлению "Машинное Обучение"

Перед вами репозиторий курсового проекта данной школы.

### Тема проекта – **построение рекомендаций на фондовом рынке**

## Аннотация проекта (Executive summary)

### Задача проекта:
Формирование стратегий инвестирования и рекомендаций на фондовом рынке с применением средств машинного обучения.

### В рамках проекта были сформулированы следующие стратегии: 

**Принцип “купи и держи”** подразумевает идею купить акции и ожидать длительный период в расчете получить доход от роста стоимости ценной бумаги и от выплачиваемых дивидендов.

**“Колебания стоимости акций в периоды дивидендных выплат”** – идея строиться на предположении, что за несколько дней до выплаты дивидендов происходит рост стоимости ценной бумаги. Задача приобрести ценную бумагу до роста и продавать максимально близко к экс дивидендной дате (продажа по повышенной цене).

**Стратегия “Корреляция различных финансовых инструментов в общей отрасли”** – идея строиться на предположении, что компании одной отрасли могут быть связаны общими событиями и менять тренды движения котировок в общую сторону. 

**Стратегия “Прогнозирование тренда на основе предыдущих данных”** – прогнозирование направления тренда на основе предыдущих цен. Идея заключается в том что при построении модели машинного обучения, моделью будут выявлены закономерности по которым будут строиться рекомендации.

В качестве входных данных выбраны ценные бумаги нефтегазовой отрасли с 2017 по 2021 года.

Основным параметром измерения экономическая оценки стратегии является **среднедневная доходность** стратегии. Дополнительными параметрами являются общее кол-во сделок за рассматриваемый период, кол-во прибыльных сделок, максимальные потери от худшей сделки, средняя длительность сделки, максимальная просадка и другие.

### Реализация
Метрикой измерения точности моделей машинного обучения выбрана метрика accuracy. 

Построенные модели по стратегиям “Корреляция различных финансовых инструментов в общей отрасли” и “Прогнозирование тренда на основе предыдущих данных” выведены в демонстрационный телеграм-канал: “https://t.me/MTSProject_StackMarket”.

Публикация осуществляется с помощью отправки рекомендаций в телеграм-канал алгоритмом на Google Colab, который использует построенные модели и данные о котировках ценных бумаг за текущие сутки.
	
  **Пример рекомендаций телеграм-канала:**

Рекомендация на 31.08.2021:  
  По стратегии "Корреляция в отрасли":      
     - Покупка LKOH (уверенность модели Conv1D: 55%)    
     - Покупка SIBN (уверенность модели LSTM: 60%)

## Участие в проекте

Проект был разработан в команде.

Моя задача заключалась в  реализации стратегии "Прогнозирование тренда на основе предыдущих данных", расчёте просадок по стратегия.

Для расчёта просадок были использованы библиотека Pandas, Numpy и стандартные средства языка Python.

Для разработки стратегии были опробованы различные специализированные библиотеки (Facebook prophet, Stocker), а также Sklearn (Модуль классификация).

В ходе проекта были получены навыки по работе с Sklearn (в частности классификация), ансамблевыми методами и в целом, задачей классификации.
Был улучшен навык ООП программирования при проведении взаимного кодового ревью с более опытными коллегами.
Также при моём участии быд написан чат-бот для отправки рекомендаций (но данное решение не попало в итоговую версию проекта, т.к. необходимо было перенетсти бота на удалённый сервер с целью его беспрерывной работы, что оказалось нерациональным)
