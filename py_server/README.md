## Основные изменения

Не пугайтесь, я тут немного зарефакторил...

1. Там было как то странно с базой данных, она подключалось к `app` которое потом вообще нигде не использовалось и просто импортировался sql\_worker.
Ну я подключил все к одному `app`, и сервер и базу.

2. Вся работа происходит в локальном окружении, `Makefile` настроен на автоматическую генерацию этого окружения в папке `.venv` если ее нет
И дальше в это окружение устанавливается все зависимости, когда вы собираете проект как package.

3. Теперь у нас не модуль, а `package` mlcraft, чтобы его установить как раз нужен `pyproject.toml`.
 Это по идеи должен делать Makefile, но на всякий случай, в папке `py_server` c активированным окружением нужно сделать:
```
 pip install -e .
```
По идеи он должен сам установить все зависимости в окружение и пакет `mlcraft` вместе с ними. То есть должно быть что-то такое:
```
(.venv) $ pip list
Package            Version   Editable project location
------------------ --------- ----------------------------------------------------------
blinker            1.6.3
certifi            2023.7.22
click              8.1.7
Flask              3.0.0
Flask-SQLAlchemy   3.1.1
iniconfig          2.0.0
itsdangerous       2.1.2
Jinja2             3.1.2
mlcraft            1.0.0     /Users/goldenberg/Developer/GraphicalEditorForNN/py_server
mypy               1.6.0
pip                23.2.1
pytest             7.4.2
requests           2.31.0
SQLAlchemy         2.0.21
```
Еще раз скажу, это все должен делать `Makefile` когда вы пытаетесь сделать `make debug`, `make release` или `make test`

4. Я убрал `utils.py` из корня в эту папку. Это изначально было мое кривое решения потому, что я увидел что там какая-то функция на порт применяется
и подумал что так и надо, а оказывается она вообще бесполезна и не нужна. Так что теперь в корне `config.json` а все просто из него читают.

5. Я использовал форматер для питона `black`, поэтому в diff могут быть всякие странные изменения типа `'` изменилась на `"`. Предлагаю дальше всем использовать этот форматтер. 

## Тесты

`Pytest` - это конечно мощная штука~...~:
1. Все тесты находятся в папке `/tests`
2. для `pytest` все функции, которые начинаются с `test_` - это тесты
3. если имя аргумента у теста совпадает с именем какого-то `fixture`, то сначала вызовется этот `fixture` чтобы создать аргумент, а потом уже сам тест.
4. есть встроенные `fixture`, есть кастомные. Кастомные определены в `conftest.py`

## pyproject.toml

Не знаю что там написано и для чего, но это позволяет собрать всю папку mlcraft как пакет.
И потом можно установить этот пакет, и можно запускать из любой директории, так что да, это решение проблемы с импортом.  

**НО**
1. Я не знаю, у меня работает, у других, как всегда скорее всего с 1 раза ничего не заработает
2. Надо явно добавить каких-то зависимостей туда еще
3. Все это вообще надо делать в своей локальной env, и туда надо загружать пакет каждый раз

