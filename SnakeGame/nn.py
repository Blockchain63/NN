from snake_game import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected,dropout
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

class SnakeNN:
    #Конструктор
    def __init__(self, train_games = 25000, test_games = 2500, goal_steps = 2000, lr = 0.01, filename = 'snake_nn.tflearn', test_filename='test_observations'):
        self.train_games = train_games  #размер обучающей выборки (в играх)
        self.test_games = test_games        #размер тестовой выборки
        self.goal_steps = goal_steps        #количество дествий за игру
        self.lr = lr
        self.filename = filename
        self.test_filename = test_filename
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        #Вектор для перемещения от соответсвующих клавиш
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]

    def save_list(self, observations,filename):
        X = np.array([i[0] for i in observations])
        Y = np.array([i[1] for i in observations])
        np.save(file=filename + ' x', arr=X)
        np.save(file=filename + ' y', arr=Y)

    def generate_train_data(self,add_test=False):
        training_data = []
        for i in range(self.train_games):
            game = SnakeGame()
            _done, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for j in range(self.goal_steps):
                #Генерируем действие
                action, game_action = self.generate_action(snake)
                #Меняем состояние игры
                done, score, snake, food = game.step(game_action)
                if done:
                    #Если завершена то добавить в тренеровочные данные наблюдения с действием и -1
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    # Иначе если счет увеличился или расстояние до еды сократилось
                    if score > prev_score or food_distance < prev_food_distance:
                        # Добавить в тренеровочные данные наблюдения с действием и 1
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        # Иначе добавить в тренеровочные данные наблюдения с действием и 0
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
        if add_test:
            X = np.load(file=self.test_filename +' x.npy')
            Y = np.load(file=self.test_filename +' y.npy')
            X.reshape(-1, 5, 1)
            Y.reshape(-1, 1)

            for i in range(0,len(X)-1):
                training_data.append([X[i],Y[i]])
        self.save_list(observations=training_data, filename='train_set')
        #Возвращает массив наблюдений и оценок
        return training_data

    def generate_action(self, snake):
        #Генерируем число от -1 до 1
        action = randint(0,2) - 1
        #Возвращаем действие и нажатую клавишу
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        #Получаем направление движения змейки
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        #Определяем новое направление змейки
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        #Ищем соответствие
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi


    def model(self):
        #Инициализация входного слоя
        network = input_data(shape=[None, 5, 1], name='input')
        #Проход через скрытый слой с relu
        network = fully_connected(network, 40, activation='relu')
        network = dropout(network, keep_prob=0.4)
        network = fully_connected(network, 30, activation='softplus')
        network = dropout(network, keep_prob=0.6)
        network = fully_connected(network, 30, activation='softplus')
        network = dropout(network, keep_prob=0.3)
        network = fully_connected(network, 30, activation='relu')
        network = dropout(network, keep_prob=0.7)
        #Выходной слой с линейной активацией
        network = fully_connected(network, 1, activation='tanh')
        #Оптимизатор
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        #Сборка модели НН где network - тензор
        model = tflearn.DNN(network, tensorboard_dir='log',tensorboard_verbose=1)
        return model

    def train_model(self, training_data, model):
        # X - observation
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        # Y - оценка данного наблюдения -1 0 1
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        #Тренировка сети, где epoch - один пропуск input'а через сеть
        model.fit(X,y, n_epoch = 5, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model,print_stats,save_obs,print_avrg=True):
        steps_arr = []
        scores_arr = []
        game_memory = []
        #Цикл по играм
        for i in range(self.test_games):
            steps = 0

            game = SnakeGame()
            _done, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            #Цикл по шагам
            for j in range(self.goal_steps):
                predictions = []
                #Перебор предсказаний для разных действий
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                #Выбор действия с наибольшей ценностью
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                # Совершаем выбранное действие
                done, score, snake, food  = game.step(game_action)
                # Сохраняем текущее действие
                if save_obs:
                    game_memory.append([self.add_action_to_observation(prev_observation, action),float(predictions[action])])
                else:
                    game_memory.append([prev_observation, action])
                #если игра закончена то печатаем результаты
                if done:
                    if print_stats:
                        print('-----')
                        print(steps)
                        print(snake)
                        print(food)
                        print(prev_observation)
                        print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        if save_obs:
            self.save_list(observations=game_memory,filename=self.test_filename)

        #Печать средних значений по всем играм
        if print_avrg:
            print('Average steps:',mean(steps_arr))
            print(Counter(steps_arr))
            print('Average score:',mean(scores_arr))
            print(Counter(scores_arr))

    def visualise_game(self, model):
        game = SnakeGame(gui = True)
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)

    def train(self,add_test):
        training_data = self.generate_train_data(add_test=add_test)
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        #self.test_model(nn_model, print_stats= True, save_obs=True)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(model_file=self.filename)
        self.visualise_game(nn_model)

    def test(self,print_stats,save_obs):
        nn_model = self.model()
        nn_model.load(model_file=self.filename)
        self.test_model(nn_model,print_stats,save_obs)





if __name__ == "__main__":
    #SnakeNN().train(False)
    SnakeNN().test(print_stats=True,save_obs=True)
    #SnakeNN().train(True)
    #SnakeNN().test(print_stats=False,save_obs=True)
    #SnakeNN().test(print_stats=True, save_obs=False)
    #SnakeNN().visualise()
