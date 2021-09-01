import tensorflow
from tensorflow.keras.models import Sequential #, Model # Подлючаем класс создания модели Sequential
import pandas as pd
import numpy as np # Подключаем библиотеку numpy

import joblib
import os
import glob
from datetime import datetime, date


class PredictRecommendation(object):
	def set_error(self,e_text):
		self.errors += f"\n{e_text}"
		if self.is_log:
			print(e_text)

	def __init__(self, model_path, is_log = False):
		self.is_log = is_log
		self.errors = ""
		if self.is_log:
			print(model_path)
		try:
			self.model = tensorflow.keras.models.load_model(model_path)
		except:
			self.set_error("Error. Не удалось загрузить модель")
			return None

		# В имени модели указаны наименование модели и для какого компании она делает предикт (важно для стартегии корреляции)
		split_file_model = model_path.split("/")[-1].split("\\")[-1].split("_")
		self.type_model = split_file_model[5] 
		self.symbol = split_file_model[0]
		self.symbols_df = {}
		self.symbols_df_normalize = {}
		# Для корреляционной модели используются только опредленыне символы
		self.available_symbols = ["LKOH", "ROSN", "GAZP", "NVTK", "SIBN", "SNGSP"]
		if self.is_log:
			print(self.type_model)
			print(self.symbol)



	def load_DataSet(self, path = "DataSet/GazNeft_D1/"):
		# Опреденляем текущий день
		today = datetime.now().strftime("%d_%m_%Y")
		# Получаем список файлов
		print(os.getcwd())
		dataset_path = path + today + "/"
		dirlist = os.listdir(dataset_path)
		print(dirlist)		
		# Получаем список файлов
# 		try:
# 			dataset_path = path + today + "/"
# 			dirlist = os.listdir(dataset_path)
# 			if self.is_log:
# 				print(dirlist)
# 		except:
# 			self.set_error(f"Error. Не удалось найти каталог с датасетами за сегодня: {today}")
# 			return None
		if len(dirlist) == 0:
			self.set_error(f"Error. Отсутсвтуют данные в каталоге с сегодняшнями данными : {today}")
			return None
		#массив символов 
		symbols = []
		
		# Парсер дат для индексов для D1
		mydateparser = lambda x: pd.datetime.strptime(x, "%Y.%m.%d %H:%M").date()
		for csv_file in dirlist:
			symbol = csv_file.split("_")[0]
			# Часть символов пропускается 
			if not symbol in self.available_symbols: 
			    continue
			symbols.append(symbol)
			df = pd.read_csv(dataset_path + csv_file, sep=";", parse_dates=['TIMEDATE'], date_parser=mydateparser, index_col='TIMEDATE')
			self.symbols_df[symbol] = df
		# Важным условием является одинаковый размер векторов по всем символам, для обучеиня все вектоа приводились к единой длине за счет пропуска везде отсутсвующих данных хотя бы где-то
		# В бою же будем только проверять и если длина данных хоть у одного символа отличается, то не делаем прогноз и выводим сообщение об ошибке 
		# Берем список индексов от первого датасета, в дальнейшем будем делать персение с ним всех остальных
		cnt_indexes = len(self.symbols_df[list(self.symbols_df.keys())[0]].index.values) 
		for symbol in self.symbols_df.keys():
			if len(self.symbols_df[symbol].index.values) != cnt_indexes:
				self.set_error(f"Есть пропуски в данных. Кол-во данных {symbol} не соответстввеут длине {list(self.symbols_df.keys())[0]}")
				return None
		if self.is_log:
			print("Кол-во уникальных индексов общих для всех датасетов: ", cnt_indexes)
		return self.symbols_df

	def normalize_Dataset(self, path_normailzer = "Normalizer"):
		if len(self.symbols_df) == 0:
			self.set_error("Error. Отсутсвтуют данные по символам ")
			return None
		# self.symbols_df
		# symbols_df[symbol]["Close_Normalize"] = x_transform
		for symbol in self.symbols_df.keys():
			x = self.symbols_df[symbol]["Close"].values.reshape(1,-1)
			transformer = joblib.load(f"{path_normailzer}/{symbol}_dataset_Normalizer.pkl")
			x_transform = transformer.transform(x)[0]
			# symbols_df[symbol]["Close_Normalize"] = x_transform
			self.symbols_df_normalize[symbol] = x_transform
			self.symbols_df_normalize["TIMEDATE"] = self.symbols_df[symbol].index.values
		self.symbols_df_normalize = pd.DataFrame.from_dict(self.symbols_df_normalize, orient='columns')
		self.symbols_df_normalize = self.symbols_df_normalize.set_index(['TIMEDATE'])
		return self.symbols_df_normalize

	def create_x(self):
		if len(self.symbols_df_normalize) == 0:
			self.set_error("Error. Отсутсвтуют нормализованные данные по символам ")
			return []
		# Формируем таргет
		for symbol in self.symbols_df_normalize.keys():
			self.symbols_df_normalize[f"{symbol}_prc_change"] = self.symbols_df_normalize[symbol].pct_change()
			self.symbols_df_normalize[f"{symbol}_trend"] = np.where(self.symbols_df_normalize[f"{symbol}_prc_change"] >= 0, 1,0).copy()
		x = self.symbols_df_normalize[["LKOH_trend", "NVTK_trend",  "SIBN_trend", "ROSN_trend", "SNGSP_trend", "GAZP_trend"]].values
		return x

	def predict_y(self, x_input):
		if len(x_input) == 0:
			self.set_error("Error. Отсутсвтуют данные для предикта")
			self.predict = []
			return []
		self.predict = self.model.predict(x_input).flatten()
		if self.is_log: 
			print("x_input => ", x_input)
			print("predict => ", self.predict)
		return self.predict

	def get_recomend(self):
		print(f" type_model => {self.type_model}, symbol => {self.symbol}, predict => {self.predict}")
		if len(self.predict) == 0:
			self.set_error("Error. Отсутсвтуют предикты")
			return ""
		if self.predict[0] >= 0.5 and self.predict[1] >= 0.5: # and self.predict[1] > self.predict[0]:
			return f"<i> - Покупка {self.symbol} (уверенность модели {self.type_model}: {int(self.predict[1]*100)}%)</i>"
		elif self.predict[0] < 0.5 and self.predict[1] < 0.5: # and self.predict[1] < self.predict[0]:
			return f"<i> - Продажа {self.symbol} (уверенность модели {self.type_model}: {int((1- self.predict[1])*100)}%)</i>"
		else:
			return ""


class PredictRecommendationTrend(object):
	
	def __init__(self, model_strength_path, model_direction_path, is_log = False):
		self.is_log = is_log
		if self.is_log:
			print(model_strength_path)
			print(model_direction_path)
		try:
			self.model_strength = joblib.load(model_strength_path)
			self.model_direction = joblib.load(model_direction_path)
		except ValueError:
			print("Error. Не удалось загрузить модель")
			print(ValueError)
			return None

		# В имени модели указаны наименование модели и для какого компании она делает предикт (важно для стартегии корреляции)
		split_file_model_strength = model_strength_path.split("/")[-1].split("_")
		split_file_model_direction = model_direction_path.split("/")[-1].split("_")

		self.type_model_strength = split_file_model_strength[0] 
		self.type_model_direction = split_file_model_direction[0] 
		self.symbol_strength = split_file_model_strength[3]
		self.symbol_direction  = split_file_model_direction[3]
		self.symbols_df = {}
		self.symbols_df_normalize = {}
		# Для корреляционной модели используются только опредленыне символы
		self.available_symbols = ["LKOH", "ROSN", "GAZP", "NVTK", "SIBN", "SNGSP"]
		if self.is_log:
			print(self.type_mtype_model_strengthodel)
			print(self.symbol_strength)
			print(self.type_model_direction)
			print(self.symbol_direction)

	def load_DataSet(self, path = "DataSet/GazNeft_D1/"):
		# Опреденляем текущий день
		today = datetime.now().strftime("%d_%m_%Y")
		# Получаем список файлов
		#print(path + "GazNeft_D1/" + today )
		try:
			dataset_path = path + today + "/"
			dirlist = os.listdir(dataset_path)
			if self.is_log:
				print(dirlist)
		except:
			print("Error. Не удалось найти каталог с датасетами за сегодня: ", today)
			return None
		if len(dirlist) == 0:	
			print("Error. Отсутсвтуют данные в каталоге с сегодняшнями данными : ", today)
			return None
		#массив символов 
		symbols = []
		
		# Парсер дат для индексов для D1
		mydateparser = lambda x: pd.datetime.strptime(x, "%Y.%m.%d %H:%M").date()
		for csv_file in dirlist:
			symbol = csv_file.split("_")[0]
			# Часть символов пропускается 
			if not symbol in self.available_symbols: 
			    continue
			symbols.append(symbol)
			df = pd.read_csv(dataset_path + csv_file, sep=";", parse_dates=['TIMEDATE'], date_parser=mydateparser, index_col='TIMEDATE')
			self.symbols_df[symbol] = df
		# Важным условием является одинаковый размер векторов по всем символам, для обучеиня все вектоа приводились к единой длине за счет пропуска везде отсутсвующих данных хотя бы где-то
		# В бою же будем только проверять и если длина данных хоть у одного символа отличается, то не делаем прогноз и выводим сообщение об ошибке 
		# Берем список индексов от первого датасета, в дальнейшем будем делать персение с ним всех остальных
		cnt_indexes = len(self.symbols_df[list(self.symbols_df.keys())[0]].index.values) 
		for symbol in self.symbols_df.keys():
			if len(self.symbols_df[symbol].index.values) != cnt_indexes:
				print("Есть пропуски в данных. Кол-во данных ", symbol, " не соответстввеут длине ", list(self.symbols_df.keys())[0])
				return None
		if self.is_log:
			print("Кол-во уникальных индексов общих для всех датасетов: ", cnt_indexes)
		return self.symbols_df
	
	def normalize_Dataset(self, path_normailzer = "Trend_normalizer"):
		# self.symbols_df
		# symbols_df[symbol]["Close_Normalize"] = x_transform
		for symbol in self.symbols_df.keys():
			#df = data.diff()[1:]['Close']
			x = self.symbols_df[symbol]['Close'].diff().tail(11)[1:]
			#print(x)
			transformer = joblib.load(f"{path_normailzer}/trend_classification_normalizer_{symbol}.pkl")
			#print(transformer, 'GOTCHA!_transformer')
			x_transform = transformer.transform([x])
			#print(x_transform, 'GOTCHA_X_transform!')
			# symbols_df[symbol]["Close_Normalize"] = x_transform
			self.symbols_df_normalize[symbol] = x_transform[0]
		#self.symbols_df_normalize["TIMEDATE"] = self.symbols_df[symbol].index.values
		
		#self.symbols_df_normalize = pd.DataFrame.from_dict(self.symbols_df_normalize, orient='columns')
		
		#self.symbols_df_normalize = self.symbols_df_normalize.set_index(['TIMEDATE'])
		#print(self.symbols_df_normalize)
		return self.symbols_df_normalize

	def create_x(self):
	# 	# Формируем таргет
	# 	# for symbol in self.symbols_df_normalize.keys():
	# 	# 	self.symbols_df_normalize[f"{symbol}_prc_change"] = self.symbols_df_normalize[symbol].pct_change()
	# 	# 	self.symbols_df_normalize[f"{symbol}_trend"] = np.where(self.symbols_df_normalize[f"{symbol}_prc_change"] >= 0, 1,0).copy()
		#x = self.symbols_df_normalize[["LKOH_trend", "NVTK_trend",  "SIBN_trend", "ROSN_trend", "SNGSP_trend", "GAZP_trend"]].values
		# print(x)
		 return self.symbols_df_normalize[self.symbol_direction.split('.')[0]]

	def predict(self, x):
		self.predict_strength = self.model_strength.predict([x]).flatten()
		self.predict_direction = self.model_direction.predict([x]).flatten()

		if self.is_log: 
			print("x => ", x)
			print("predict => ", self.predict_strength)
			print("predict => ", self.predict_direction)
		return [self.predict_strength, self.predict_direction]
	
	def get_recomend(self):
			
		symbol = self.symbol_direction.split('.')[0]	
		if self.predict_direction == 1:
			# Направление вверх
			text_rec_direct = 'Предположительное направление тренда – вверх'
		else:
			# Направление вниз
			text_rec_direct = 'Предположительное направление тренда – вниз'

		
		if self.predict_strength == 1:
			# Слабый тренд
			text_rec_strength = 'Вероятность дальнейшего продолжения тренда – средняя'
		elif self.predict_strength == 2:
			# Сильный тренд
			text_rec_strength = 'Вероятность дальнейшего продолжения тренда – большая'
		else:
			# Отсутствие тренда
			text_rec_strength = 'Вероятность дальнейшего продолжения тренда – маленькая'

		if self.predict_strength == 2:
			if self.predict_direction == 0:
				return  f'\n\t{symbol}: Продавать ({text_rec_direct}; {text_rec_strength})'
			else:
				return  f'\n\t{symbol}: Покупать ({text_rec_direct}; {text_rec_strength})'
		else:
			return ""

# # Формируем прогноз 
# path_model = "models/"
# rec = ""
# rec_array = []
# # dirlist = os.listdir(path_model)
# # Dense
# dirlist = glob.glob(path_model + f"*model_Dense*")
# for file_model in dirlist:
# 	print(file_model)
# 	# recommend = PredictRecommendation(model_path = path_model + file_model, is_log = False)
# 	recommend = PredictRecommendation(model_path = file_model, is_log = False)
# 	recommend.load_DataSet()
# 	symbols_df_normalize = recommend.normalize_Dataset(path_normailzer = "Normalizer")
# 	x = recommend.create_x()
# 	# Для модели dense необходимо передавать значения за текущий и предыдущий день. Решение о покупке или продаже принимается по двум предикатам 
# 	# last_day_x = x[-1][None]
# 	last_day_x = x[-2:]
# 	predict = recommend.predict_y(last_day_x)
# 	if recommend.errors.strip() == "":
# 		rec_array.append(recommend.get_recomend())
# 	else:
# 		print("Выявлены ошибки!")
# 		print(recommend.errors)

# # LSTM + Conv1D
# dirlist = glob.glob(path_model + f"*model_Conv1D*") + glob.glob(path_model + f"*model_LSTM*")
# for file_model in dirlist:
# 	print(file_model)
# 	# recommend = PredictRecommendation(model_path = path_model + file_model, is_log = False)
# 	recommend = PredictRecommendation(model_path = file_model, is_log = False)
# 	recommend.load_DataSet()
# 	symbols_df_normalize = recommend.normalize_Dataset(path_normailzer = "Normalizer")
# 	x = recommend.create_x()
# 	# Для модели dense необходимо передавать значения за текущий и предыдущий день. Решение о покупке или продаже принимается по двум предикатам 
# 	# last_day_x = x[-1][None]
# 	# last_day_x = x[-2:]

# 	# Для модели LSTM и Conv1D передаем вектор из 10 дней, но для предсказания нужны предикты за текущий и предыдузий дни
# 	last_day_x = [x[-8:]]
# 	last_day_x = np.array(last_day_x)
# 	last_day_predict = recommend.predict_y(last_day_x)
	
# 	pre_last_day_x = [x[-9:-1]]
# 	pre_last_day_x = np.array(pre_last_day_x)
# 	pre_last_day_predict = recommend.predict_y(pre_last_day_x)

# 	recommend.predict = np.array([pre_last_day_predict[0], last_day_predict[0]])
# 	# recommend.predict = [pre_last_day_predict[0], last_day_predict[0]]
	
# 	if recommend.errors.strip() == "":
# 		rec_str = recommend.get_recomend()
# 		if rec_str.strip() != "":
# 			rec_array.append(rec_str)
# 	else:
# 		print("Выявлены ошибки!")
# 		print(recommend.errors)





# show_rec = True 
# today = datetime.now().strftime("%d.%m.%Y")
# recommendation_text = f"Рекоменадция на <b>{today}</b>:\n\t\t По стратегии \"Корреляция в отрасли\" (среднедневная доходность стратегии: 0.49%):\n\t\t\t\t"

# if len(rec_array) == 0:
# 	show_rec = False
# else:
# 	if "".join(rec_array).strip() == "":
# 		rec = " Рекомендаций нет"
# 	else:
# 		rec = '\n\t\t\t\t'.join(rec_array)
# 	recommendation_text+=rec

# print("show_rec: ", show_rec)
# print("recommendation_text: ", recommendation_text)


# def bot_recommend():	#Функция для передачи в бота (будет вызывать её при отправке сообщений)
# 	path_model = "models/"
# 	dirlist = os.listdir(path_model)
# 	rec = ""

# 	#НОВЫЙ БЛОК ИНИЦИАЛИЗАЦИЙ
# 	path_model_direction = "trend_direction_models/"
# 	path_model_strength = "trend_strength_models/"	
# 	dirlist_trend_strength = os.listdir(path_model_strength)
# 	dirlist_trend_direction = os.listdir(path_model_direction)	
# 	rec_trend = "" 

# 	#СТАРЫЙ ЦИКЛ ДЛЯ ОТРАБОТКИ КОРРЕЛЯЦИЙ В ОТРАСЛИ
# 	# for file_model in dirlist:
# 	# 	recommend = PredictRecommendation(model_path = path_model + file_model, is_log = False)
# 	# 	recommend.load_DataSet()
# 	# 	symbols_df_normalize = recommend.normalize_Dataset(path_normailzer = "Normalizer")
# 	# 	x = recommend.create_x()
# 	# 	# Для модели dense необходимо передавать значения за текущий и предыдущий день. Решение о покупке или продаже принимается по двум предикатам
# 	#     # last_day_x = x[-1][None]
# 	# 	last_day_x = x[-2:]
# 	# 	predict = recommend.predict(last_day_x)
# 	# 	rec += recommend.get_recomend()
# 	# if rec.strip() == "":
# 	# 	rec = "\n\tРекомендаций нет"

# 	#НОВЫЙ ЦИКЛ ДЛЯ ПРОГНОЗИРОВАНИЯ ТРЕНДА
# 	for i in range(len(dirlist_trend_strength)):
# 		file_model_strength = dirlist_trend_strength[i]
# 		file_model_direction = dirlist_trend_direction[i]
# 		recommend = PredictRecommendationTrend(model_strength_path= path_model_strength + file_model_strength, model_direction_path=path_model_direction+file_model_direction, is_log = False)
# 		recommend.load_DataSet() #WORKING
# 		symbols_df_normalize = recommend.normalize_Dataset(path_normailzer="Trend_normalizer")
# 		x = recommend.create_x()
# 		predict = recommend.predict(x)
# 		rec_trend += recommend.get_recomend()
# 	if rec_trend.strip() == "":
# 		rec_trend = "\n\tРекомендаций нет"


# 	today = datetime.now().strftime("%d.%m.%Y")
# 	recommendation_text = f"Рекоменадция на {today}:\n- По стратегии \"Корреляция в отрасли\" (среднедневная доходность стратегии: 0.49%)"
# 	recommendation_text+=rec
	
# 	# ДОБАВЛЕНИЕ РЕКОМЕНДАЦИЙ В ВЫВОД
# 	recommendation_text_trend =  f"\n- По стратегии \"Предсказание тренда\" (среднедневная доходность стратегии: 0.10%)"
# 	recommendation_text_trend+=rec_trend

# 	return recommendation_text+recommendation_text_trend


# # MAIN – бот
# import telebot
# import schedule
# import time
# import random
# bot = telebot.TeleBot("1796683033:AAFufTxBTP7tpA6Vg_L6ITZ-u4sCKv2S-Dw") #Токен бота

# def job():
#     bot.send_message(-538207044, bot_recommend()) #Настройка чата для отправки
# schedule.every(5).seconds.do(job) #Настройка времени отправки
# print("Bot is working...")
# while True:
#     schedule.run_pending()
#     time.sleep(1)
# bot.polling()

# #id чата -449244173 (с минусом именно) 
# #token 1998964887:AAEfRPVITx5uSpAR9dggYn2z4m2MiNS_Eoo