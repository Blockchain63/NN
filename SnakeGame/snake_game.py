import curses
from random import randint


class SnakeGame:
    # "Конструктор"
    def __init__(self, board_width=25, board_height=25, gui=False):
        self.score = 0
        self.is_done = False
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui

    def start(self):
        self.init_snake()
        self.generate_food()
        if self.gui:
            self.render_init()
        return self.generate_observations()

    def init_snake(self):
        x = randint(5, self.board["width"] - 5)
        y = randint(5, self.board["height"] - 5)
        # Массив из частей змейки
        self.snake = []
        for i in range(3):
            point = [x, y + i]
            self.snake.insert(0, point)

    def generate_food(self):
        food = []
        while food == []:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]
            if food in self.snake: food = []
        self.food = food

    def render_init(self):
        curses.initscr()  # инициализация библиотеки
        # возвращает окно с левым верхним углом в (0,0) с размерами width+2 height+2
        window = curses.newwindow(self.board["width"] + 2, self.board["height"] + 2, 0, 0)
        # делает курсор невидимым
        curses.curs_set(0)
        # getch() не блокирует
        window.nodelay(1)
        # getch() блокирует на 200 мс
        window.timeout(200)
        self.window = window
        self.render()

    def render(self):
        self.window.clear()
        self.window.border(0)
        self.window.addstr(0, 2, 'Score : ' + str(self.score))
        self.window.addch(self.food[0], self.food[1], 'Q')
        for i, point in enumerate(self.snake):
            if i == 0:
                # Голова змеи
                self.window.addch(point[0], point[1], '@')
            else:
                # Остальные куски змеи
                self.window.addch(point[0], point[1], 'o')
        # Получение кнопки
        self.window.getch()

    def step(self, key):
        # Клавиши
        # 0 - Вверх
        # 1 - Вправо
        # 2 - Вниз
        # 3 - Влево
        if self.is_done:
            self.end_game()
        self.add_point(key)
        if self.food_eaten():
            self.score += 1
            self.generate_food()
        else:
            self.remove_last_point()
        self.check_collisions()
        if self.gui: self.render()
        return self.generate_observations()

    def add_point(self, key):
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == 0:
            new_point[0] -= 1
        elif key == 1:
            new_point[1] += 1
        elif key == 2:
            new_point[0] += 1
        elif key == 3:
            new_point[1] -= 1
        self.snake.insert(0, new_point)

    def remove_last_point(self):
        self.snake.pop()

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_collisions(self):
        # Если змейка совершив ход, врежется в одну из границ
        if (self.snake[0][0] == 0 or
                self.snake[0][0] == self.board["width"] + 1 or
                self.snake[0][1] == 0 or
                self.snake[0][1] == self.board["height"] + 1 or
                # Не врежется в одну из своих точек(кроме хвоста) 
                self.snake[0] in self.snake[1:-1]):
            self.is_done = True

    def generate_observations(self):
        return self.is_done, self.score, self.snake, self.food

    def render_end(self):
        curses.endwindow()

    def end_game(self):
        if self.gui: self.render_end()
        raise Exception("Game over")
