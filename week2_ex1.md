### COURSERA / специализация МАШИННОЕ ОБУЧЕНИЕ И АНАЛИЗ ДАННЫХ (МФТИ)
### Курс 1: Математика и Python для анализа данных.
### Задание по программированию. Линейная алгебра: сходство текстов и аппроксимация функций
Данное задание основано на материалах секции, посвященной введению в линейную алгебру. Вам понадобится компьютер с установленным интерпретатором Python и подключенными библиотеками NumPy и SciPy.

Вы научитесь:
- читать тексты из файла с помощью Python и разбивать их на слова;
- переводить тексты в векторные пространства, вычислять расстояния в этих пространствах;
- решать системы линейных уравнений;
- приближать любые функции с помощью многочленов.

Введение

В этом задании вы познакомитесь с некоторыми базовыми методами из линейной алгебры, реализованными в пакете SciPy — в частности, с методами подсчета косинусного расстояния и решения систем линейных уравнений. Обе эти задачи еще много раз встретятся нам в специализации. Так, на решении систем линейных уравнений основана настройка линейных моделей — очень большого и важного класса алгоритмов машинного обучения. Косинусное расстояние же часто используется в анализе текстов для измерения сходства между ними.

Материалы

Справка по функциям пакета scipy.linalg: http://docs.scipy.org/doc/scipy/reference/linalg.html

Справка по работе с файлами в Python: https://docs.python.org/2/tutorial/inputoutput.html#reading-and-writing-files

Справка по регулярным выражениям в Python (если вы захотите узнать про них чуть больше): https://docs.python.org/2/library/re.html

Инструкция по выполнению

Данное задание состоит из двух частей. В каждой ответом будет набор чисел, который вам нужно будет ввести в соответствующее поле через пробел.

### Задача 1: сравнение предложений

Дан набор предложений, скопированных с Википедии. Каждое из них имеет "кошачью тему" в одном из трех смыслов:

- кошки (животные);
- UNIX-утилита cat для вывода содержимого файлов;
- версии операционной системы OS X, названные в честь семейства кошачьих.

Ваша задача — найти два предложения, которые ближе всего по смыслу к расположенному в самой первой строке. В качестве меры близости по смыслу мы будем использовать косинусное расстояние.

Выполните следующие шаги:

1. Скачайте файл с предложениями (sentences.txt).
2. Каждая строка в файле соответствует одному предложению. Считайте их, приведите каждую к нижнему регистру с помощью строковой функции lower().


```python
#file_object = open ('sentences.txt', 'r')
#text_lower = file_object.read().lower()
#print(text_lower)
```

3. Произведите токенизацию, то есть разбиение текстов на слова. Для этого можно воспользоваться регулярным выражением, которое считает разделителем любой символ, не являющийся буквой: re.split('[^a-z]', t). Не забудьте удалить пустые слова после разделения.


```python
#import re
#for line in open('sentences.txt', 'r').readlines():
#    words = re.split('[^a-z]', line.lower())
#    filtered_words = list(filter(None, words))
#    print(filtered_words)
```

4. Составьте список всех слов, встречающихся в предложениях. Сопоставьте каждому слову индекс от нуля до (d - 1), где d — число различных слов в предложениях. Для этого удобно воспользоваться структурой dict.



```python
import re
words_dict = re.split('[^a-z]', open('sentences.txt', 'r').read().lower())
key_lst = set(list(filter(None, words_dict))) # set убираем повторяющиеся слова, итого уникальных 254
value_list = [i for i in range(0,254)]
my_dict = dict(zip(key_lst, value_list))
print(my_dict)
```

    {'later': 0, 'ancestor': 1, 'with': 2, 'lion': 3, 'often': 4, 'firmware': 5, 'was': 6, 'closest': 7, 'since': 8, 'use': 9, 'entirely': 10, 'received': 11, 'release': 12, 'create': 13, 'however': 14, 'they': 15, 'just': 16, 'both': 17, 'domestication': 18, 'commands': 19, 'left': 20, 'no': 21, 'major': 22, 'based': 23, 'small': 24, 'pipes': 25, 'type': 26, 'by': 27, 'longer': 28, 'simply': 29, 'are': 30, 'possess': 31, 'high': 32, 'versions': 33, 'rather': 34, 'installs': 35, 'won': 36, 'run': 37, 'released': 38, 'domestic': 39, 'typically': 40, 'safer': 41, 'you': 42, 'processors': 43, 'app': 44, 'intel': 45, 'from': 46, 'tiger': 47, 'external': 48, 'started': 49, 'editions': 50, 'predecessor': 51, 'mac': 52, 'kg': 53, 'running': 54, 'frequency': 55, 'allow': 56, 'binary': 57, 'linux': 58, 't': 59, 'dogs': 60, 'online': 61, 'disk': 62, 'selection': 63, 'can': 64, 'organisms': 65, 'installation': 66, 'three': 67, 'offered': 68, 'purchase': 69, 'mid': 70, 'permanently': 71, 'wild': 72, 'upgrade': 73, 'bytes': 74, 'unix': 75, 'default': 76, 'instead': 77, 'be': 78, 'successor': 79, 'roughly': 80, 'connected': 81, 'symbol': 82, 'redirection': 83, 'content': 84, 'right': 85, 'cats': 86, 'now': 87, 'domesticated': 88, 'drive': 89, 'not': 90, 'according': 91, 'those': 92, 'named': 93, 'os': 94, 'have': 95, 'undergone': 96, 'clear': 97, 'need': 98, 'piped': 99, 'apple': 100, 'arguments': 101, 'such': 102, 'common': 103, 'switch': 104, 'genes': 105, 'command': 106, 'interactive': 107, 'standard': 108, 'its': 109, 'when': 110, 'animals': 111, 'members': 112, 'new': 113, 'safari': 114, 'were': 115, 'leopard': 116, 'hear': 117, 'more': 118, 'faint': 119, 'most': 120, 'in': 121, 'using': 122, 'moved': 123, 'comparison': 124, 'error': 125, 'may': 126, 'has': 127, 'output': 128, 'that': 129, 'human': 130, 'time': 131, 'lines': 132, 'is': 133, 'any': 134, 'mavericks': 135, 'place': 136, 'sequence': 137, 'fifth': 138, 'cat': 139, 'receives': 140, 'without': 141, 'some': 142, 'between': 143, 'than': 144, 'where': 145, 'and': 146, 'stdout': 147, 'process': 148, 'used': 149, 'starting': 150, 'mice': 151, 'which': 152, 'patch': 153, 'basic': 154, 'useful': 155, 'july': 156, 'ears': 157, 'genus': 158, 'so': 159, 'installed': 160, 'predators': 161, 'as': 162, 'chromosomes': 163, 's': 164, 'october': 165, 'vermin': 166, 'an': 167, 'the': 168, 'displays': 169, 'second': 170, 'osx': 171, 'off': 172, 'concatenate': 173, 'unnecessary': 174, 'to': 175, 'part': 176, 'streams': 177, 'deliberately': 178, 'releasing': 179, 'diploid': 180, 'flow': 181, 'wrong': 182, 'factory': 183, 'world': 184, 'or': 185, 'if': 186, 'download': 187, 'allows': 188, 'changes': 189, 'redirected': 190, 'of': 191, 'a': 192, 'will': 193, 'felis': 194, 'for': 195, 'information': 196, 'every': 197, 'yosemite': 198, 'enhancements': 199, 'files': 200, 'incremental': 201, 'needing': 202, 'version': 203, 'one': 204, 'terms': 205, 'over': 206, 'non': 207, 'file': 208, 'keyboards': 209, 'adjacent': 210, 'symbols': 211, 'marks': 212, 'two': 213, 'artificial': 214, 'year': 215, 'size': 216, 'weighing': 217, 'people': 218, 'x': 219, 'releases': 220, 'contains': 221, 'also': 222, 'read': 223, 'mountain': 224, 'separate': 225, 'features': 226, 'community': 227, 'legibility': 228, 'concern': 229, 'learned': 230, 'count': 231, 'during': 232, 'update': 233, 'lb': 234, 'catenates': 235, 'single': 236, 'stdin': 237, 'computers': 238, 'store': 239, 'similar': 240, 'developed': 241, 'available': 242, 'recent': 243, 'it': 244, 'their': 245, 'sounds': 246, 'through': 247, 'delete': 248, 'tamed': 249, 'made': 250, 'other': 251, 'too': 252, 'on': 253}


5. Создайте матрицу размера n * d, где n — число предложений. Заполните ее: элемент с индексом (i, j) в этой матрице должен быть равен количеству вхождений j-го слова в i-е предложение. У вас должна получиться матрица размера 22 * 254.


```python
# Проводим токенизацию текста, переводим строки к нижнему регистру, фильтруем текст - убираем все знаки
# препинания и пустые значения. В итоге получаем слова из каждого предложения filtered_words.
import re
import numpy as np
big_list = []
# Работаем по каждому предложению отдельно
for line in open('sentences.txt', 'r').readlines():
    words = re.split('[^a-z]', line.lower())
    filtered_words = list(filter(None, words))
#    print(filtered_words)
    total_vector = np.zeros(254) # делаю нулевой вектор, понадобится ниже
# Для слов в предложении:    
    for word in filtered_words:
        s = [] # делаю пустой список, понадобится ниже
# Перебираем ключи-слова в словаре
        for key in my_dict:
            if key == word: # если ключ-слово из словаря my_dict совпадает со словом из предложения
                s.append(1) # в созданный пустой список добавляем 1
            else:
                s.append(0) # в другом случае добавляем 0
        vector = np.array(s) # полученный список преобразуем в вектор
        total_vector = total_vector + vector # складываем полученные векторы для получения итогового вектора-предложения
#    print(total_vector)
    total_list = list(total_vector) # вектор переделываем на список
#    print(total_list)
    big_list = big_list + total_list # складываем все списки простым суммированием
#print(big_list)
matrix = (np.array(big_list)).reshape((22, 254)) # из списка переделываем в вектор и reshape в матрицу
print(matrix)
        
```

    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]


6. Найдите косинусное расстояние от предложения в самой первой строке (In comparison to dogs, cats have not undergone...) до всех остальных с помощью функции scipy.spatial.distance.cosine. Какие номера у двух предложений, ближайших к нему по этому расстоянию (строки нумеруются с нуля)? Эти два числа и будут ответами на задание. Само предложение (In comparison to dogs, cats have not undergone... ) имеет индекс 0.


```python
#Чем более схожи векторы (объекты), тем меньше косинусное расстояние

import scipy
from scipy import spatial
# Выведем первую строку матрицы
for n in range(0, 22):
    result = scipy.spatial.distance.cosine(matrix[0, :], matrix[n, :])
    print(result)

```

    0.0
    0.9527544408738466
    0.8644738145642124
    0.8951715163278082
    0.7770887149698589
    0.9402385695332803
    0.7327387580875756
    0.9258750683338899
    0.8842724875284311
    0.9055088817476932
    0.8328165362273942
    0.8804771390665607
    0.8396432548525454
    0.8703592552895671
    0.8740118423302576
    0.9442721787424647
    0.8406361854220809
    0.956644501523794
    0.9442721787424647
    0.8885443574849294
    0.8427572744917122
    0.8250364469440588


7. Запишите полученные числа в файл, разделив пробелом. Обратите внимание, что файл должен состоять из одной строки, в конце которой не должно быть переноса. Пример файла с решением вы можете найти в конце задания (submission-1.txt).
8. Совпадают ли ближайшие два предложения по тематике с первым? Совпадают ли тематики у следующих по близости предложений?

### ответ: предложение №6 и №4
