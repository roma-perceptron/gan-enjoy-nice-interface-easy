# coding=utf-8

class GENIE_Texts():
	def __init__(self, lang='rus'):
		self.lang = lang

	def hello(self):
		text = {'rus': rus_hello, 'eng': eng_hello}
		return text[self.lang]

	def example(self):
		text = {'rus': rus_example, 'eng': eng_example}
		return text[self.lang]

	def help(self):
		text = {'rus': rus_help, 'eng': eng_help}
		return text[self.lang]


rus_example = '''
!git clone -s -q https://github.com/roma-perceptron/gan-interface-ready-to-labor.git genie_lib
from genie_lib import genie_lib
import threading

metrics_for_monitoring = ['loss', 'accuracy', 'your_custom_complex_metric']

genie = GAN_Enjoy_Nice_Interface_Easy('/content/monitoring/', metrics_for_monitoring)

epochs = 100

def training_func(epochs):
# как часто дозаписывать данные, в эпохах
save_period = 1

"""
код для подготовки, batch_size, labels и прочее
"""

# создание словаря для хранения данных в памяти
history = {field_name: [] for field_name in metrics_for_monitoring}

# цикл для эпох
for epoch in range(epochs):

  """
  ваш код обучения
  """

  # запись данных по метрикам в словарь истории
  history['loss'].append(your_var_with_this_metric)
  history['accuracy'].append(your_var_with_this_metric)
  # пример ниже для демонстрации того что вы можете накапливать не только
  # встроенные метрики, но и свои собственные, если это необходимо
  history['your_custom_complex_metric'].append(your_var_with_this_metric)

  # условие для очередной дозаписи данных
  if epoch % save_period == 0:  
    
    # этот код преобразует последние save_period записей в словарях в строки для дозаписи
    data_for_write = zip(*[history[field][-save_period:] for field in metrics_for_monitoring])
    string_lines = [';'.join([str(e) for e in line])+'\n' for line in data_for_write]
    
    # для дозаписи используется встроенный метод из экземпляра класса genie
    genie.write_to_history_file(string_lines)

    # для генерации используется так же встроенный метод; ему необходимо передать ваш генератор и константный шум
    genie.show_gen_cat(generator, noise=your_noise_for_generate_previews, epoch_number=epoch)
    
  # в конце эпохи нужно сообщить экземпляру класса номер последней эпохи
  genie.epoch = epoch

  # интерфейс имеет кнопку досрочного прекращения обучения, этот код для нее
  # не пренебрегайте этим, т.к. после запуска обучения в потоке, прекратить его через остановку ячейки будет невозможно
  if genie.control_command_code == 'stop_training':
    break
    
thread = threading.Thread(target=training_func, args=(epochs,))
thread.start()

genie.show_interface(epochs=epochs)
'''

eng_example = rus_example

rus_help = '''
GAN: Enjoy Nice Interface Easy или просто GENIE это небольшая библиотека предназанченная для облегчения процесса
мониторинга за обучением генеративно-состязательных сетей (GAN). А в будущем, возможно появятся и по управлению
гиперпараметрами.

Мониторинг - это отображение данных накопленных во время обучения, и существенная часть GENIE это построение интерфейса
на базе виджетов (ipywidgets) чтобы можно было следить за данными метрик в табличном и графическом виде, а так же
за примером генерации изображения. Для этих целей экземпляр GENIE создаст на локальном диске .csv файл для данных
и папку под сгенерированные изображения.

Но чтобы данные отображать из в первую очередь необходимо собрать. И делать это нужно в процессе обучения, который
кроме того необходимо запустить асинхронно в отдельном потоке. Звучит сложнее чем на самом деле и ниже будет подробный
шаблон кода.

Как использовать:

1. Для начала необходимо подгрузить саму библиотеку genie и threading для создания потоков

	!git clone -s -q https://github.com/roma-perceptron/gan-interface-ready-to-labor.git genie_lib
	from genie_lib import genie_lib
	import threading

2. Теперь необходимо определиться с набором метрик которые будут сохраняться в .csv файле. Этот момент, возможно, самый
сложный в использовании GENIE, т.к. необходимо заранее указать то что будет окончательно оформлено только на этапе
доведения до ума функции обучения. В реальной жизни придется возвращаться к этому пункту и редактировать его несколько
раз. На старте не стоит слишком заморачиваться. Сейчас набор метрик это просто список строковых имен в том виде как вы
хотите их видеть в финальной таблице и на графиках.

	metrics_for_monitoring = ['loss', 'accuracy', 'your_custom_complex_metric']

3. Следующим шагом нужно создать экземпляр класса указав два обязательных параметра. Первый это корневая папка для
хранения данных, а второй это список метрик из предыдущего пункта.

	genie = GAN_Enjoy_Nice_Interface_Easy('/content/monitoring/', metrics_for_monitoring)
Имейте в виду, что при каждом пересоздании экземпляра класса все данные в указанной папке будут перезатерты. Поэтому
удобно располагать этот код в отдельной ячейке, чтобы при повторных обучениях данные накапливались.
	
4. Теперь необходимо написать код для самого процесса обучения и обернуть его в функцию. Сделайте число эпох для
обучения параметром данной функции. Эпоха, а точнее ее конец - это момент очередной записи данных в .cvs файл.
Организовать дозапись в файл можно различными способами. В примере ниже будет использоваться подход с накоплением всех
данных в памяти специального словаря и порционной дозаписи последних N-записей из этого словаря. Способ избыточный если
готовы писать каждую эпоху.

  epochs = 100

  def training_func(epochs):
  	# как часто дозаписывать данные, в эпохах
    save_period = 1

	"""
	код для подготовки, batch_size, labels и прочее
	"""
	
	# создание словаря для хранения данных в памяти
    history = {field_name: [] for field_name in metrics_for_monitoring}

	# цикл для эпох
    for epoch in range(epochs):

	  """
	  ваш код обучения
	  """

      # запись данных по метрикам в словарь истории
      history['loss'].append(your_var_with_this_metric)
      history['accuracy'].append(your_var_with_this_metric)
	  # пример ниже для демонстрации того что вы можете накапливать не только
	  # встроенные метрики, но и свои собственные, если это необходимо
      history['your_custom_complex_metric'].append(your_var_with_this_metric)

      # условие для очередной дозаписи данных
      if epoch % save_period == 0:  
	  	
		# этот код преобразует последние save_period записей в словарях в строки для дозаписи
        data_for_write = zip(*[history[field][-save_period:] for field in metrics_for_monitoring])
        string_lines = [';'.join([str(e) for e in line])+'\\n' for line in data_for_write]
		
		# для дозаписи используется встроенный метод из экземпляра класса genie
        genie.write_to_history_file(string_lines)

		# для генерации используется так же встроенный метод; ему необходимо передать ваш генератор и константный шум
        genie.show_gen_cat(generator, noise=your_noise_for_generate_previews, epoch_number=epoch)
		
      # в конце эпохи нужно сообщить экземпляру класса номер последней эпохи
      genie.epoch = epoch

	  # интерфейс имеет кнопку досрочного прекращения обучения, этот код для нее
	  # не пренебрегайте этим, т.к. после запуска обучения в потоке, прекратить его через остановку ячейки будет невозможно
      if genie.control_command_code == 'stop_training':
        break
		
      
5. Чтобы запустить обучение и иметь возможность работать с интерфейсом (а бонусом запускать код в других ячейках)
необходимо запустить обучение в отдельном потоке. Я использую библиотеку threading. Имя целевой функции указывается
в параметре target, а ее аргументы в виде кортежа параметра args.

  thread = threading.Thread(target=training_func, args=(epochs,))
  thread.start()


6. И последний шаг - запуск самого интерфейса. После запуска он автоматически начнет с определенной переодичностью
собирать и обновлять данные. Переодичностью можно будет управлять в диапазоне 1-60 секунд.

  genie.show_interface(epochs=epochs)
'''

eng_help = '''
Sorry, english version in process, try russian.
'''

rus_hello = '''
Привет! Используй метод help() если нужно больше инофмации и примеры!
'''

eng_hello = '''
Hi! Try help() for more information and examples!
'''

