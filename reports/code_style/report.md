## Инструменты
Были выбраны следующие инструменты:

*   pycodestyle

Этот стилистический линтер был выбран так как Python --- основной язык данного проекта. 
Для читаемости и простоты дальнейшей поддержки необходимо держать код в описанном стандартом PEP8 виде.
*   Bandit

Проверка кодовой базы на уязвимости --- важное дело не только для SE, но и для ML--инженера.
*   prospector

Захотелось использовать более мощный инструмент, чем pycodestyle и подвергнуть свой код более тщательному анализу.


### pycodestyle 

Был установлен с помощью пакетного менеджера `pip`.
 
Запускался со следующими настройками: 

```(shell)
pycodestyle --statistics -qq . > reports/code_style/pycodestyle.txt
```

Проверка выполняется на всех .py файлах
 в корневой директории проекте и проверяет их на соответствие PEP8.
Найденные ошибки аггрегируются и подсчитываются. 

Результат анализа:

```(shell)
69      E501 line too long (86 > 79 characters)
1       W391 blank line at end of file
``` 

Нашлось 69 строк, превышающих рекомендованную длину.
А так же один файл без пустой завершающей строки.

### bandit 

Был установлен с помощью пакетного менеджера `pip`.
 
Запускался со следующими настройками: 

```(shell)
bandit -r -l -i -v -x .git . > reports/code_style/bandit.txt
```
Настроил вывод всех ошибок, не смотря на степерь уверенности и уровень угрозы. 
Указывает все успешно обойденные и пропущенные файлы.

Результаты анализа:

```(shell)
Test results:
>> Issue: [B605:start_process_with_a_shell] Starting a process with a shell, possible injection detected, security issue.
   Severity: High   Confidence: High
   Location: ./src/data/download.py:15
   More Info: https://bandit.readthedocs.io/en/latest/plugins/b605_start_process_with_a_shell.html
14	    if rewrite or not path_train.exists():
15	        os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-train.csv -P {path_raw_data}")
16	        LOGGER.debug("Train dataframe successfully loaded.")

--------------------------------------------------
>> Issue: [B605:start_process_with_a_shell] Starting a process with a shell, possible injection detected, security issue.
   Severity: High   Confidence: High
   Location: ./src/data/download.py:18
   More Info: https://bandit.readthedocs.io/en/latest/plugins/b605_start_process_with_a_shell.html
17	    if rewrite or not path_test.exists():
18	        os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-test.csv -P {path_raw_data}")
19	        LOGGER.debug("Test dataframe successfully loaded.")

--------------------------------------------------
>> Issue: [B605:start_process_with_a_shell] Starting a process with a shell, possible injection detected, security issue.
   Severity: High   Confidence: High
   Location: ./src/data/download.py:21
   More Info: https://bandit.readthedocs.io/en/latest/plugins/b605_start_process_with_a_shell.html
20	    if rewrite or not path_desc.exists():
21	        os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-desc.csv -P {path_raw_data}")
22	        LOGGER.debug("Description dataframe successfully loaded.")

--------------------------------------------------
>> Issue: [B101:assert_used] Use of assert detected. The enclosed code will be removed when compiling to optimised byte code.
   Severity: Low   Confidence: High
   Location: ./src/models/threshold_tuning.py:13
   More Info: https://bandit.readthedocs.io/en/latest/plugins/b101_assert_used.html
12	def _get_optimal_threshold_by_accuracy(probs, labels):
13	    assert len(probs) == len(labels)
14	    n = len(probs)

--------------------------------------------------

Code scanned:
	Total lines of code: 426
	Total lines skipped (#nosec): 0

Run metrics:
	Total issues (by severity):
		Undefined: 0.0
		Low: 1.0
		Medium: 0.0
		High: 3.0
	Total issues (by confidence):
		Undefined: 0.0
		Low: 0.0
		Medium: 0.0
		High: 4.0
Files skipped (0):
```

Ругается на запуск подпроцесса в shell и на использование assert.



### prospector 

Был установлен с помощью пакетного менеджера `pip`.
 
Запускался со следующими настройками: 

```(shell)
prospector --strictness medium > reports/code_style/prospector.txt
``` 

Результат анализа:

```
Messages
========

definitions.py
  Line: 36
    pylint: trailing-newlines / Trailing newlines

src/features/feature_extraction.py
  Line: 9
    pylint: too-many-locals / Too many local variables (21/15)
    pylint: too-many-statements / Too many statements (73/60)

src/main_pipeline.py
  Line: 90
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)
  Line: 99
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)
  Line: 109
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)
  Line: 116
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)
  Line: 146
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)
  Line: 150
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)
  Line: 161
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)

src/models/parameter_selection.py
  Line: 13
    pylint: no-else-return / Unnecessary "else" after "return" (col 8)
  Line: 47
    pylint: too-many-arguments / Too many arguments (6/5)

src/models/threshold_tuning.py
  Line: 40
    pylint: too-many-locals / Too many local variables (20/15)
  Line: 66
    pylint: logging-fstring-interpolation / Use lazy % formatting in logging functions (col 4)

```

Много замечаний использовать вместо fstring % formatting, но я считаю, что это менее читаемо.
Обилие локальных переменных и параметров в функции.
Дельное замечание про ненужный блок else, исправил следующим коммитом.
