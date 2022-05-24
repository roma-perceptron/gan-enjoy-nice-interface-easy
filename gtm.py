# coding=utf-8

"""
Tool for monitoring and manage(in future) process of trainig GANs
"""

# imports
import os
import csv
import time
import shutil
import imageio
import datetime

# import pickle
import numpy as np
import pandas as pd

# from tqdm import tqdm
# from google import colab
import ipywidgets as widgets

from IPython.core.display import clear_output
import matplotlib.pyplot as plt


class DeepLearningRealTimeMonitoring():
  # 
  def __init__(self, endpoint, history_fields, separator=';', reload_time=10, preview_size=200, clear_generated=True, plt_style='default'):
  #     
    # cheking path for endpoint
    if not os.path.exists(ENDPOINT):
      # try to create path if parent folder exist, else raised FileNotFoundError
      os.mkdir(ENDPOINT)

    # class constants
    self.ENDPOINT = endpoint if endpoint.endswith('/') else endpoint + '/'
    self.HISTORY_FILE = self.ENDPOINT + 'history.csv'
    self.GENERATED = self.ENDPOINT + 'generated/'
    self._GENERATED = self.ENDPOINT + '_generated/'
    self.SEPARATOR = separator
    self.PREVIEW_SIZE = preview_size
    self.FIELDNAMES = history_fields
    self.CLEAR_GENERATED = clear_generated

    # class variables
    self.reload_time = reload_time
    self.images_volume = 0
    self.df = pd.DataFrame({f: sorted([1 if abs(n) >=1 else abs(n) for n in np.random.normal(-0.5, 0.5, size=100)], reverse=True if 'loss' in f else False) for f in history_fields})

    # initializing
    self._prepare_history_file()
    self._create_generated_path()
    self._create_pseudo_generated_path()
    self.images_volume = len(os.listdir(self.GENERATED)) if len(os.listdir(self.GENERATED)) > 0 else len(os.listdir(self._GENERATED))
    plt.style.use(plt_style)
    
    # create interface
    self.interface = self.get_interface()

  def _prepare_history_file(self):
    if os.path.exists(self.HISTORY_FILE):
      os.remove(self.HISTORY_FILE)

    if not os.path.exists(self.HISTORY_FILE):
      with open(self.HISTORY_FILE, mode='w') as f:
        f.write(';'.join(self.FIELDNAMES)+'\n')


  def _create_generated_path(self):
    if not os.path.exists(self.GENERATED):
      os.mkdir(self.GENERATED)
    elif self.CLEAR_GENERATED:
      shutil.rmtree(self.GENERATED)
      os.mkdir(self.GENERATED)

  # 
  def _create_pseudo_generated_path(self):
    if not os.path.exists(self._GENERATED):
      os.mkdir(self._GENERATED)
    if len(os.listdir(self._GENERATED)) == 0:
      for i in range(1, 6):
        genered = np.random.randint(0, 255, (64,64,3))
        genered = genered / 255
        plt.imshow(genered, cmap='gray')
        plt.axis('off')
        plt.title('placeholder', y=-0.1, loc='left')
        plt.title(i, y=-0.1, loc='right')
        plt.savefig('{}/{}.png'.format(self._GENERATED, i), format='png', bbox_inches='tight')
        plt.close()


  # read image in bytes
  def _get_genered_image(self, num=-1):
    if os.path.exists(self.GENERATED) and len(os.listdir(self.GENERATED)) > 0:
      path_for_generated = self.GENERATED
    else:
      self._create_pseudo_generated_path()
      path_for_generated = self._GENERATED
    # 
    files = sorted(os.listdir(path_for_generated), key=lambda x: int(x.split('.')[0]))
    genenered_image_name = files[num]
    with open(path_for_generated+genenered_image_name, mode='rb') as f:
      genered_image_bytes = f.read()
    # 
    return genered_image_bytes


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

    fig, (m_plot, l_plot) = plt.subplots(nrows=1, ncols=2, figsize=(25, 6))
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
      img_preview.value = self._get_genered_image(slider_image.value)

    def update_image_preview(self_widget):
      if slider_image.value == -1:
        img_preview.value = self._get_genered_image(-1)
      else:
        with output_for_prevs:
          slider_image.min = -self.images_volume
          clear_output()
          print('Есть новые изображения')

    def reset_epochs_range_now(self_widget):
      epochs_range.index = (0, self.df.shape[0]-1)

    def update_graph(self_widget):
      # draw_history
      if self_widget.name != '_property_lock':
        metrics_to_draw = [m.description for m in metrics if m.value]
        epochs_range_to_draw = epochs_range.index
        with output_for_graph:
          clear_output(wait=True)
          self._draw_history(self.df[metrics_to_draw][epochs_range_to_draw[0]: epochs_range_to_draw[1]].to_dict(orient='list'), start_x=epochs_range_to_draw[0])

    # global updater by timer
    def update_data(self_widget):      
      with output_for_props:
        print('До обновления..', 9 - slider_update.value, 'секунд.')
        if slider_update.value == 0:
          if self_widget != None and self_widget.name == '_property_lock':
            print('breaking!')
            return None
          print('Обновляюсь!!!')
          # 
          # обновление картинки
          self.images_volume = len(os.listdir(self.GENERATED)) if len(os.listdir(self.GENERATED)) > 0 else len(os.listdir(self._GENERATED))
          update_image_preview(None)
          
          # обновление таблицы
          with output_for_table:
            clear_output(wait=True)
            pd.set_option('display.precision', 3)
            display(self.df.tail(10))

          # обновление графиков
          with output_for_graph:
            clear_output(wait=True)
            metrics_to_draw = [m.description for m in metrics if m.value]
            epochs_range_to_draw = epochs_range.index
            self._draw_history(self.df[metrics_to_draw][epochs_range_to_draw[0]: epochs_range_to_draw[1]].to_dict(orient='list'), start_x=epochs_range_to_draw[0])
            # self._draw_history(self.df.to_dict(orient='list')) #[['D loss', 'D acc']]
        clear_output(wait=True)
    # end of iternal functions

    # create outputs
    output_for_graph = widgets.Output()
    output_for_table = widgets.Output()
    output_for_props = widgets.Output()
    output_for_prevs = widgets.Output()

    # управление обновлением
    uptdate_buttons = widgets.Play(
        value=1,
        min=0,
        max=9,
        step=1,
        interval=1000,
        description="Press play"
    )
    slider_update = widgets.IntSlider(0, min=0, max=9)
    update_widget = widgets.jslink((uptdate_buttons, 'value'), (slider_update, 'value'))
    update_box = widgets.HBox([uptdate_buttons, slider_update])
    slider_update.observe(update_data)
    uptdate_buttons._repeat = True


    # превью сгенерированных картинок
    img_preview = widgets.Image(
        value=self._get_genered_image(-1),
        format='jpg',
        # width=self.PREVIEW_SIZE,
        layout=widgets.Layout(border='3px outset #e0e0e0')
    )

    slider_image = widgets.IntSlider(-1, min=-self.images_volume, max=-1, continuous_update=False, readout=False, layout=widgets.Layout())
    slider_image.observe(change_image_preview)

    play_buttons = widgets.Play(
        value=-1,
        min=-self.images_volume,
        max=-1,
        step=1,
        interval=100,
        description="Press play",
        disabled=False
    )
    play_widget = widgets.jslink((play_buttons, 'value'), (slider_image, 'value'))
    play_box = widgets.HBox([play_buttons, slider_image])

    # 
    table_box = widgets.Box([output_for_table], layout=widgets.Layout(border='2px solid #e0e0e0', height='350px', width='40%'))
    props_box = widgets.VBox([update_box, output_for_props], layout=widgets.Layout(border='2px solid #e0e0e0', height='350px'))
    graph_box = widgets.Box([output_for_graph], layout=widgets.Layout(border='2px solid #e0e0e0', height='400px'))

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
      m.observe(update_graph)
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
    epochs_range.observe(update_graph)
    reset_epochs_range = widgets.Button(description='Весь диапазон')
    reset_epochs_range.on_click(reset_epochs_range_now)
    epochs_range_box = widgets.HBox([epochs_range, reset_epochs_range])
    

    # сборка интерфейса из частей
    preview_box = widgets.VBox([img_preview, play_box], layout=widgets.Layout(border='2px solid #e0e0e0', width='35%', height='350px', align_items='center'))
    upper_box = widgets.HBox(children=[preview_box, table_box, props_box])
    lower_box = widgets.VBox([metrics_box, graph_box, epochs_range_box])
    main_box = widgets.VBox([upper_box, lower_box], layout=widgets.Layout(align_content='space-around', border='10px solid white'))

    interface = widgets.Box([main_box], layoyt=widgets.Layout(border='50px solid green'))
    #
    update_data(None)
    return interface


  # отрисовква генерации
  def show_gen_cat(self, generator, noise, epoch_number=0, verbose=1, path_for_generated=''):
    cmap = None
    generated = generator.predict(np.asarray([noise]))[0]
    if generated.ndim == 3 and generated.shape[-1] == 3:
      cmap = 'gray'
      generated = generated.reshape(generated.shape[:2])
    # 
    plt.imshow(generated, cmap=cmap)
    plt.axis('off')
    plt.title(generator.name, y=-0.1, loc='left')
    plt.title(epoch_number, y=-0.1, loc='right')

    if not path_for_generated:
      path_for_generated = self.GENERATED
    next_num = len(os.listdir(path_for_generated)) + 1
    plt.savefig('{}/{}.png'.format(path_for_generated, next_num), format='png')
    
    if verbose:
      plt.show()
    else:
      plt.close()


  def get_animation(self, first=0, last=-1, each=1, easy_count=100, easy_in=True, easy_out=True, silent=False, path_for_generated=''):
    if not path_for_generated:
      path_for_generated = self.GENERATED

    files = sorted(os.listdir(path_for_generated), key=lambda x: int(x.split('.')[0]))
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

  def help(self, lang='eng'):
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

    
# --- prepare monitoring in three steps:
# 1. Define list of metrics which you will write to CSV file. You must use same names
# history_fields = ['accuracy', 'val accuracy', 'loss', 'val loss']

# 2. Create instance of DLRTN class
# dlrtm_instance = DeepLearningRealTimeMonitoring(endpoint='/content', history_fields = history_fields)

# 3. Render interface
# display(dlrtm_instance.get_interface())

# now you should start training process. Don't remember to implement writing
# history data to csv-file and making preview of generated images (if GAN)
# use special tools included in DLRTN
