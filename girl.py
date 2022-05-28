# coding=utf-8

"""
WOW!
"""

# imports
import os
import csv
import time
import shutil
import imageio
import datetime

import numpy as np
import pandas as pd

# from tqdm import tqdm
# from google import colab
import ipywidgets as widgets

import matplotlib.pyplot as plt
# from IPython.core.display import clear_output

# self.df = pd.DataFrame({f: sorted([1 if abs(n) >=1 else abs(n) for n in np.random.normal(-0.5, 0.5, size=100)], reverse=True if 'loss' in f else False) for f in history_fields})


class GAN_Interface_Ready_to_Labor():
  # DeepLearningRealTimeMonitoring
  # GAN-fan GAN-van
  # GIRL: gan interface ready to labor
  # 
  def __init__(self, endpoint, history_fields, separator=';', preview_size=250, clear_generated=True, image_shape=(64, 64, 3), plt_style='default'):
    # class constants
    self.ENDPOINT = endpoint if endpoint.endswith('/') else endpoint + '/'
    self.HISTORY_FILE = self.ENDPOINT + 'history.csv'
    self.GENERATED = self.ENDPOINT + 'generated/'
    self._GENERATED = self.ENDPOINT + '_generated/'
    self.SEPARATOR = separator
    self.PREVIEW_SIZE = preview_size
    self.FIELDNAMES = history_fields
    self.CLEAR_GENERATED = clear_generated
    self.IMAGE_SHAPE = image_shape

    # class variables
    self.allowed_refresh_graph = True
    self.allowed_refresh_image = True
    self.images_volume = 0
    self.df = pd.DataFrame({f: [None] for f in self.FIELDNAMES})
    self.control_command_code = 0
    self.epoch = 0
    self.epochs = 0


    # cheking path for endpoint
    if not os.path.exists(self.ENDPOINT):
      os.mkdir(self.ENDPOINT)

    # initializing
    self._prepare_history_file()
    self._create_generated_path()
    self.images_volume = len(os.listdir(self.GENERATED))
    plt.style.use(plt_style)
    
    # create interface
    self.interface = self.get_interface()


  def display_interface(self, epochs):
    self.control_command_code = 0
    self.epochs = epochs
    display(self.interface)


  def _prepare_history_file(self):
    if os.path.exists(self.HISTORY_FILE):
      os.remove(self.HISTORY_FILE)

    if not os.path.exists(self.HISTORY_FILE):
      with open(self.HISTORY_FILE, mode='w') as f:
        f.write(';'.join(self.FIELDNAMES)+'\n')

  def _create_zero_epoch_image(self):
    generated = np.zeros(self.IMAGE_SHAPE)
    pic = Image.fromarray(generated.astype('uint8'), mode='RGB')
    pic.format = 'png'

    next_num = len(os.listdir(self.GENERATED)) + 1
    pic.save('{}/0_e0.png'.format(self.GENERATED), format='png')

  def _create_generated_path(self):
    if not os.path.exists(self.GENERATED):
      os.mkdir(self.GENERATED)
      self._create_zero_epoch_image()
    elif self.CLEAR_GENERATED:
      shutil.rmtree(self.GENERATED)
      os.mkdir(self.GENERATED)
      self._create_zero_epoch_image()

  # sort by number
  def _get_filelist_sorted(self, pathname):
    return sorted(os.listdir(pathname), key=lambda x: int(x.split('.')[0].split('_e')[0]))

  # read image in bytes
  def _get_genered_image(self, num=-1):
    if os.path.exists(self.GENERATED) and len(os.listdir(self.GENERATED)) > 0:
      path_for_generated = self.GENERATED
    else:
      self._create_pseudo_generated_path()
      path_for_generated = self._GENERATED
    # 
    files = self._get_filelist_sorted(path_for_generated)
    genenered_image_name = files[num]
    with open(path_for_generated+genenered_image_name, mode='rb') as f:
      genered_image_bytes = f.read()
    # 
    return genered_image_bytes, genenered_image_name.split('.')[0].split('_e')[-1]


  # draw plot for loss and metrics
  def _draw_history(self, history, only='', exclude='', final=None, show_trend=False, show_avg=False, start_x=0):
    # 
    labeling = {
        'accuracy': 'Доля верных ответов на обучающем наборе',
        'val_accuracy': 'Доля верных ответов на проверочном наборе',
        'loss': 'Ошибка на обучающем наборе',
        'val_loss': 'Ошибка на проверочном наборе',
        'final': 'Финальное значение метрики',
    } 

    fig, (m_plot, l_plot) = plt.subplots(nrows=1, ncols=2, figsize=(35, 6), facecolor='#f5f5f5')
    m_plot.set(title='График метрики (accuracy или иное)', xlabel='Эпоха обучения', ylabel='Значение')
    l_plot.set(title='График ошибки (loss)', xlabel='Эпоха обучения', ylabel='Значение')
    
    # отрисовка всех данных которые есть в history
    for param in history:
      ax = l_plot if 'loss' in param else m_plot
      if param not in exclude:
        # показываю примерный тренд
        if show_trend and 'val' in param:
          # xdots, ydots это набор икс-координат, и y-координат по отдельности!
          xdots = len(history[param])//2, len(history[param])-1
          ydots = history[param][len(history[param])//2], history[param][-1]
          ax.plot(xdots, ydots, color='r', linestyle='--', linewidth=3)
          
        # # рисую скользящую среднюю
        # if show_avg:
        #   xLen = len(history[param]) // 10
        #   ma = lambda x: np.asarray(x).mean() # маленькая обертка для расчета среднего и возврата целого числа
        #   xyma = [ma(history[param][i:i+xLen+1]) if i > xLen else ma(history[param][:i+1]) for i in range(len(history[param]))]
        #   ax.plot(xyma, color='y', linewidth=6)
        # print(*history[param].items())
        if 'loss' in param:
          ax.plot(range(start_x, start_x+len(history[param])), history[param], label=labeling.get(param, param))
        else: #? это что? хм
          ax.plot(range(start_x, start_x+len(history[param])), history[param], label=labeling.get(param, param))

      # лениво узнаю сколько эпох было..
      epochs = len(history[param])

    if final:
      m_plot.plot([final for i in range(epochs)], label=labeling.get('final', final))

    m_plot.legend()
    l_plot.legend()

    # 
    plt.show()



  def get_interface(self):
    """
    """

    # iternal functions and widget-callbacks (with "self_widget" argument instead global "self" for class instance)
    def change_image_preview(self_widget):
      img_bytes, epoch_num = self._get_genered_image(slider_image.value)
      img_preview.value = img_bytes
      epoch_label.value = f'from epoch: {epoch_num}'
      # 
      if slider_image.value == -1:
        self.allowed_refresh_image = True
      else:
        self.allowed_refresh_image = False
      

    def update_image_preview(self_widget):
      if self.allowed_refresh_image:
        img_bytes, epoch_num = self._get_genered_image(-1)
        img_preview.value = img_bytes
        epoch_label.value = f'from epoch: {epoch_num}'
      

    def reset_epochs_range_now(self_widget):
      epochs_range.options = range(self.df.shape[0])
      epochs_range.index = (0, self.df.shape[0]-1)
      self.allowed_refresh_graph = True

    def update_graph(self_widget):
      # draw_history
      if self_widget.new != (0, 0):
        metrics_to_draw = [m.description for m in metrics if m.value]
        epochs_range_to_draw = epochs_range.index
        with output_for_graph:
          clear_output(wait=True)
          self._draw_history(self.df[metrics_to_draw][epochs_range_to_draw[0]: epochs_range_to_draw[1]].to_dict(orient='list'), start_x=epochs_range_to_draw[0])


    def update_all(self):
      # print('try to update!!!')
      # обновление базы со строками
      if epochs_range.index[0] != 0 or epochs_range.index[1] != self.df.shape[0]-1:
        self.allowed_refresh_graph = False

      if os.path.getsize(self.HISTORY_FILE) > 100:
        self.df = pd.read_csv(self.HISTORY_FILE, sep=';')

        # обновление графиков
        if self.allowed_refresh_graph:
          epochs_range.options = range(self.df.shape[0])
          epochs_range.index = (0, self.df.shape[0]-1)
      else:
        with output_for_graph:
          clear_output(wait=True)
          self._draw_history(df)

      # обновление картинки
      self.images_volume = len(os.listdir(self.GENERATED))
      slider_image.min = -self.images_volume
      update_image_preview(None)
      
      # обновление таблицы
      with output_for_table:
        clear_output(wait=True)
        pd.set_option('display.precision', 3)
        display(self.df.tail(10))




    # global updater by timer
    def update_data(self_widget):
      training_progress = round(100 * (self.epoch / self.epochs), 2)
      training_progress_bar.value = training_progress
      training_progress_lbl.value = f'Прогресс обучения: {training_progress}%'

      updating_progress = round(100 * slider_update.value / slider_update.max, 2)
      updating_progress_bar.value = updating_progress
      updating_progress_lbl.value = f'Обновление: {updating_progress}%'

      with output_for_props:
        clear_output(wait=True)
        if gan_monitor.epoch > 0:
          print(f'Эпоха #{gan_monitor.epoch} завершена')
          # print(slider_update.value, '/', slider_update.max)
        if slider_update.value == slider_update.max:
          print('Обновляюсь..')
          update_all(self)
          slider_update.value = 1

    def start_updating(self):
      if self.old == False and self.new == True:
        update_all(self)

    def update_button_click(self_widget):
      self.control_command_code = 99
      with output_for_props:
        print('Останавливаю обучение..')

    def change_delay_time(self_widget):
      slider_update.max = 2 * slider_update_delay.value
      update_buttons.max = 2 * slider_update_delay.value

    # end of iternal functions

    # --- --- --- --- --- --- ---
    # create outputs
    output_for_graph = widgets.Output()
    output_for_table = widgets.Output()
    output_for_props = widgets.Output()
    output_for_prevs = widgets.Output()

    # управление обновлением
    update_buttons = widgets.Play(
        value=15,
        min=1,
        max=16,
        step=1,
        interval=500,
        description="Press play",
        _repeat = True
    )
    slider_update = widgets.IntSlider(15, min=1, max=16, readout=True)
    update_widget = widgets.jslink((update_buttons, 'value'), (slider_update, 'value'))
    update_box = widgets.HBox([update_buttons, slider_update], layout=widgets.Layout(width='95%'))

    update_buttons.observe(start_updating, names='_playing')
    slider_update.observe(update_data, names='value')


    # превью сгенерированных картинок
    img_preview = widgets.Image(
        value=self._get_genered_image(-1)[0],
        format='png',
        width=self.PREVIEW_SIZE,
        layout=widgets.Layout(border='3px outset #e0e0e0')
    )

    slider_image = widgets.IntSlider(-1, min=-self.images_volume, max=-1, continuous_update=False, readout=True, layout=widgets.Layout())
    slider_image.observe(change_image_preview, names='value')

    # play_buttons = widgets.Play(
    #     value=-1,
    #     min=-self.images_volume,
    #     max=-1,
    #     step=1,
    #     interval=100,
    #     description="Press play",
    #     disabled=False
    # )
    # play_widget = widgets.jslink((play_buttons, 'value'), (slider_image, 'value'))
    # play_box = widgets.HBox([play_buttons, slider_image])

    epoch_label = widgets.Label(value='before training')

    # manual button for some controls
    update_button = widgets.Button(description='завершить обучение')
    update_button.on_click(update_button_click)

    slider_update_delay = widgets.IntSlider(8, min=1, max=60, continuous_update=False, readout=True, layout=widgets.Layout())
    slider_update_delay.observe(change_delay_time)

    #progress bars
    training_progress_lbl = widgets.Label('Прогресс обучения: 0%')
    training_progress_bar = widgets.FloatProgress(value=0, style={'bar_color': '#00bbff'}, layout=widgets.Layout(margin='-10px 0px 0px 0px'))
    training_progress_box = widgets.VBox([training_progress_lbl, training_progress_bar], layout=widgets.Layout(margin='0px 0px 10px 0px'))

    updating_progress_lbl = widgets.Label('Обновление: 0%')
    updating_progress_bar = widgets.FloatProgress(value=0, style={'bar_color': '#1cd3a2'}, layout=widgets.Layout(margin='-10px 0px 0px 0px')) #1cd3a2
    updating_progress_box = widgets.VBox([updating_progress_lbl, updating_progress_bar])

    progress_bars = widgets.VBox([training_progress_box, updating_progress_box])
    

    # 
    table_box = widgets.Box([output_for_table], layout=widgets.Layout(border='2px solid #e0e0e0', width='40%'))
    props_box = widgets.VBox([update_box, progress_bars, slider_update_delay, output_for_props, update_button], layout=widgets.Layout(border='2px solid #e0e0e0', width='25%', align_items='center'))
    graph_box = widgets.Box([output_for_graph], layout=widgets.Layout(border='2px solid #e0e0e0'))

    # панель с кнопками включения/выключения метрик на графике
    metrics = [
      widgets.ToggleButton(
          value=True,
          description=metric,
          button_style='',
          tooltip=f'Включить/исключить метрику {metric}',
      )
      for metric in sorted(self.FIELDNAMES, key=lambda x: 'loss' in x)
    ]
    for m in metrics:
      m.observe(update_graph, names='value')
    metrics_box = widgets.HBox(metrics, layout=widgets.Layout(justify_content='space-around', border='2px solid #e0e0e0', margin='25px 0px 0px 0px'))


    # панелька с выбором диапазона эпох для графика
    epochs_range = widgets.SelectionRangeSlider(
        options=range(self.df.shape[0]),
        index=(0, self.df.shape[0]-1),
        description='Диапазон эпох:',
        continuous_update=False,
        layout=widgets.Layout(width='90%'),
        style={'description_width': 'initial'}
    )
    epochs_range.observe(update_graph, names='value')
    
    reset_epochs_range = widgets.Button(description='Весь диапазон')
    reset_epochs_range.on_click(reset_epochs_range_now)
    epochs_range_box = widgets.HBox([epochs_range, reset_epochs_range])
    

    # сборка интерфейса из частей
    preview_box = widgets.VBox([img_preview, epoch_label, slider_image], layout=widgets.Layout(border='2px solid #e0e0e0', width='35%', align_items='center'))
    upper_box = widgets.HBox([preview_box, table_box, props_box])
    lower_box = widgets.VBox([metrics_box, graph_box, epochs_range_box])
    main_box = widgets.VBox([upper_box, lower_box, output_for_props], layout=widgets.Layout(align_content='space-around', border='10px solid transparent', width='100%'))

    upper_box.add_class('upper_box')
    lower_box.add_class('lower_box')

    # /content/drive/MyDrive/lesson 17 GANs/styles.css
    with open('/content/drive/MyDrive/lesson 17 GANs/styles.css', mode='r') as f:
      data_input_style = f.read()

    interface = widgets.Box([widgets.HTML(data_input_style), main_box])
    interface.add_class('main')

    #INTER
    
    # 
    return interface


  # отрисовква генерации
  def show_gen_cat(self, generator, noise, epoch_number=0, verbose=0, path_for_generated=''):
  # 
    mode = None
    generated = generator.predict(np.asarray([noise]))[0]
    if generated.ndim == 3 and generated.shape[-1] == 3:
      mode = 'RGB'
    else:
      generated = generated.reshape(generated.shape[:2])
    # 
    pic = Image.fromarray((255 * generated).astype('uint8'), mode=mode)
    pic.format = 'png'

    if not path_for_generated:
      path_for_generated = self.GENERATED
    next_num = len(os.listdir(path_for_generated)) + 1
    pic.save('{}/{}_e{}.png'.format(path_for_generated, next_num, epoch_number), format='png')

    if verbose:
      display(pic)


  def get_animation(self, first=0, last=-1, each=1, easy_count=100, easy_in=True, easy_out=True, silent=False, path_for_generated=''):
    if not path_for_generated:
      path_for_generated = self.GENERATED

    files = self._get_filelist_sorted(path_for_generated)
    # 
    easy_in_count = easy_count if easy_in else 0
    easy_out_count = -easy_count if easy_out else 0
    # 
    first_part = files[first:first+easy_in_count]
    mid_part = files[first+easy_in_count:easy_out_count+last:each]
    last_part = files[easy_out_count+last:last]
    selected_files = first_part + mid_part + last_part
    # 
    generated_images = [imageio.imread('{}/{}'.format(path_for_generated, file)) for file in selected_files]
    next_num = len([f for f in os.listdir(self.ENDPOINT) if 'GAN' in f and f.endswith('.gif')]) + 1
    imageio.mimsave(self.ENDPOINT + f'GAN_{next_num}.gif', generated_images)
    # 
    if not silent:
      print(f'Анимация GAN_{next_num}.gif из {len(generated_images)}-ти кадров создана!')


  def write_to_history_file(self, lines):
    with open(self.HISTORY_FILE, mode='a') as f:
      f.writelines(lines)

  def help(self, lang='ru'):
    help_text = {
        'eng': """Hi there!

DLRTM is tool for real time monitoring traing process of GANs primarily.
Also included features for easily making and collection preview of generated
images, and turn in to gif-animation.


Prepare monitoring in three steps:
1. Define list of metrics which you will write to CSV file. You must use same names
history_fields = ['accuracy', 'val accuracy', 'loss', 'val loss']

2. Create instance of DLRTN class
dlrtm_instance = DeepLearningRealTimeMonitoring(endpoint='/content', history_fields = history_fields)

3. Render and run interface
display(dlrtm_instance.get_interface())

now you should start training process. Don't remember to implement writing
history data to csv-file and making preview of generated images (if GAN)
use special tools included in DLRTN.

  data_for_write = zip(*[history[field][-save_period:] for field in history_fields])
  string_lines = [';'.join([str(e) for e in line])+'\\n' for line in data_for_write]
  write_to_history_file(string_lines)

Go ahead!

Roma Perceprton, 2022
roma.perceptron@gmail.com | telegram: @roma_perceptron
        """,
        'ru': """ Всем привет!
И пока!
        """
    }

    return print(help_text[lang])

    
# # --- prepare monitoring in three steps:
# # 1. Define list of metrics which you will write to CSV file. You must use same names
# history_fields = ['accuracy', 'val accuracy', 'loss', 'val loss']

# # 2. Create instance of DLRTN class
# dlrtm_instance = DeepLearningRealTimeMonitoring(endpoint='/content', history_fields = history_fields)

# # 3. Render interface
# display(dlrtm_instance.get_interface())

# # now you should start training process. Don't remember to implement writing
# # history data to csv-file and making preview of generated images (if GAN)
# # use special tools included in DLRTN
