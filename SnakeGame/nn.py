from snake_game import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter


def save_list(observations, filename):
    X = np.array([i[0] for i in observations])
    Y = np.array([i[1] for i in observations])
    np.save(file=filename + ' x', arr=X)
    np.save(file=filename + ' y', arr=Y)


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def get_angle(a, b):
    a = normalize_vector(a)
    b = normalize_vector(b)
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi


class SnakeNN:
    # Конструктор
    def __init__(self, trains=15000, tests=1000, max_steps=2000, leraning_rate=0.01, filename='nn.tflearn',
                 test_filename='test_observations'):
        self.trains = trains  # размер обучающей выборки (в играх)
        self.tests = tests  # размер тестовой выборки
        self.max_steps = max_steps  # количество дествий за игру
        self.leraning_rate = leraning_rate
        self.filename = filename
        self.test_filename = test_filename
        # 0 - Вверх
        # 1 - Вправо
        # 2 - Вниз
        # 3 - Влево
        # Вектор для перемещения для соответсвующих клавиш
        self.keys_to_vectors = [
            [[-1, 0], 0],
            [[0, 1], 1],
            [[1, 0], 2],
            [[0, -1], 3]
        ]

    def generate_train_data(self, add_test=False):
        training_data = []
        for i in range(self.trains):
            game = SnakeGame()
            _done, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for j in range(self.max_steps):
                # Генерируем действие
                action, game_action = self.generate_action(snake)
                # Меняем состояние игры после совершения действия
                done, score, snake, food = game.step(game_action)
                if done:
                    # Если завершена то добавить в тренеровочные данные наблюдения с действием и -1
                    training_data.append([self.merge_action_and_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    # Иначе если счет увеличился или расстояние до еды сократилось
                    if score > prev_score or food_distance < prev_food_distance:
                        # Добавить в тренеровочные данные наблюдения с действием и 1
                        training_data.append([self.merge_action_and_observation(prev_observation, action), 1])
                    else:
                        # Иначе добавить в тренеровочные данные наблюдения с действием и 0
                        training_data.append([self.merge_action_and_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
        if add_test:
            X = np.load(file=self.test_filename + ' x.npy')
            Y = np.load(file=self.test_filename + ' y.npy')
            X.reshape(-1, 5, 1)
            Y.reshape(-1, 1)

            for i in range(0, len(X) - 1):
                training_data.append([X[i], Y[i]])
        save_list(observations=training_data, filename='train_set')
        # Возвращает массив наблюдений и оценок
        return training_data

    def generate_action(self, snake):
        # Генерируем число от -1 до 1
        action = randint(0, 2) - 1
        # Возвращаем действие и нажатую клавишу
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        # Получаем направление движения змейки
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        # Определяем новое направление змейки
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        # Ищем соответствие
        for pair in self.keys_to_vectors:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    @staticmethod
    def merge_action_and_observation(observation, action):
        return np.append([action], observation)

    @staticmethod
    def get_snake_direction_vector(snake):
        return np.array(snake[0]) - np.array(snake[1])

    @staticmethod
    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    @staticmethod
    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    @staticmethod
    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    @staticmethod
    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    @staticmethod
    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def model(self):
        # Инициализация входного слоя
        network = input_data(shape=[None, 5, 1], name='input')
        # Проход через скрытый слой с ReLU
        network = fully_connected(network, 100, activation='relu')
        # Выходной слой с линейной активацией
        network = fully_connected(network, 1, activation='linear')
        # Оптимизатор
        network = regression(network, optimizer='adam', learning_rate=self.leraning_rate, loss='mean_square')
        # Сборка модели НН где network - тензор
        model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=1)
        return model

    def train_model(self, training_data, model):
        # X - observation
        x = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        # Y - оценка данного наблюдения -1 0 1
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        # Тренировка сети, где epoch - один пропуск input'а через сеть
        model.fit(x, y, n_epoch=5, shuffle=True, run_id=self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model, print_stats, save_obs, print_avrg=True):
        steps_arr = []
        scores_arr = []
        test_observations = []
        # Цикл по играм
        for i in range(self.tests):
            steps = 0

            game = SnakeGame()
            _isdone, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            # Цикл по шагам
            for j in range(self.max_steps):
                step_predictions = []
                # Перебор предсказаний для разных действий
                for action in range(-1, 2):
                    step_predictions.append(
                        model.predict(self.merge_action_and_observation(prev_observation, action).reshape(-1, 5, 1)))
                # Выбор действия с наибольшей ценностью
                action = np.argmax(np.array(step_predictions))
                game_action = self.get_game_action(snake, action - 1)
                # Совершаем выбранное действие
                isdone, score, snake, food = game.step(game_action)
                # Сохраняем текущее действие
                if save_obs:
                    test_observations.append([self.merge_action_and_observation(prev_observation, action), float(step_predictions[action])])
                # если игра закончена то печатаем результаты
                if isdone:
                    if print_stats:
                        print('#####################################')
                        print('steps:' + str(steps))
                        print('snake length:' + str(len(snake)))
                        # print('last step_predictions;' + str(step_predictions))
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        if save_obs:
            save_list(observations=test_observations, filename=self.test_filename)

        # Печать средних значений по всем играм
        if print_avrg:
            print('Average steps:', mean(steps_arr))
            print(Counter(steps_arr))
            print('Average score:', mean(scores_arr))
            print(Counter(scores_arr))

    def visual_test_model(self, model):
        game = SnakeGame(gui=True)
        isdone, score, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for i in range(self.max_steps):
            step_predictions = []
            for action in range(-1, 2):
                step_predictions.append(
                    model.predict(self.merge_action_and_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(step_predictions))
            game_action = self.get_game_action(snake, action - 1)
            done, score, snake, food = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)

    def train(self, add_test):
        training_data = self.generate_train_data(add_test=add_test)
        nn_model = self.model()
        self.train_model(training_data, nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(model_file=self.filename)
        self.visual_test_model(nn_model)

    def test(self, print_stats, save_obs):
        nn_model = self.model()
        nn_model.load(model_file=self.filename)
        self.test_model(nn_model, print_stats, save_obs)

    def multi_train(self, generations):
        training_data = self.generate_train_data(add_test=False)
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        for i in range(0, generations):
            print("generation:" + str(i + 1))
            self.test_model(nn_model, print_stats=False, save_obs=True, print_avrg=True)
            training_data = self.generate_train_data(add_test=True)
            nn_model = self.train_model(training_data, nn_model)


if __name__ == "__main__":
    print("1 - train without adding test observations")
    print("2 - train with adding saved test observations")
    print("3 - test without saving observations")
    print("4 - test with saving observations")
    print("5 - visualise")
    print("6 - multiple train-test-train")
    ans = input()
    if ans == '1':
        SnakeNN().train(False)
    elif ans == '2':
        SnakeNN().train(True)
    elif ans == '3':
        pr_st = input("print stats?(y/n)")
        if pr_st == 'y':
            SnakeNN().test(print_stats=True, save_obs=True)
        if pr_st == 'n':
            SnakeNN().test(print_stats=False, save_obs=True)
    elif ans == '4':
        pr_st = input("print stats?(y/n)")
        if pr_st == 'y':
            SnakeNN().test(print_stats=True, save_obs=False)
        if pr_st == 'n':
            SnakeNN().test(print_stats=False, save_obs=False)
    elif ans == '5':
        SnakeNN().visualise()
    elif ans == '6':
        gen = input("number of generations:")
        SnakeNN().multi_train(generations=int(gen))
