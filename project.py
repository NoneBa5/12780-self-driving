import pygame
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Basic vehicle parameters
INITIAL_SPEED = 8
SPEED_STEP = 1
INITIAL_ANGLE = 0
ANGLE_STEP = 5

INITIAL_RADAR = [5 * [0., 0., 0.]]

# Basic setting
TIME_LIMIT = 1000

COLLISION = pygame.USEREVENT + 1

START_POS = (800, 900)
CAR_SIZE = 60

FPS = 60
WIDTH = 1920
HEIGHT = 1080
BORDER = (255, 255, 255)
FINISH_LINE = [(19, 76, 0), (163, 193, 160)]

# Setting for neural network
INPUT_SIZE = 15
OUTPUT_SIZE = 5

HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 128

POPULATION_SIZE = 100
GENERATION = 1000
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

"""
The class of vehicles
Including all the functions and information of the vehicle
"""


class Vehicles:
    def __init__(self):
        # Load the image
        self.image = pygame.image.load("car.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (CAR_SIZE, CAR_SIZE))

        # Create the sprite for vehicles and add it to the group
        self.sprite = pygame.sprite.Sprite()
        self.sprite.image = self.image
        self.sprite.rect = self.image.get_rect()
        self.sprite.rect.center = START_POS

        # Initialize the parameters of the vehicle
        self.speed_step = SPEED_STEP
        self.angle_step = ANGLE_STEP
        self.speed = INITIAL_SPEED
        self.angle = INITIAL_ANGLE
        self.position = [START_POS[0], START_POS[1]]
        self.center = [START_POS[0] + CAR_SIZE / 2, START_POS[1] + CAR_SIZE / 2]

        # Data for collision detection
        self.alive = True
        self.corner = []

        # Data for radar detection
        self.radars = INITIAL_RADAR.copy()

        # Reward data
        self.reward = 0
        self.distance = 0

    def reset(self):
        self.sprite = pygame.sprite.Sprite()
        self.sprite.image = self.image
        self.sprite.rect = self.image.get_rect()
        self.sprite.rect.center = START_POS

        # Initialize the parameters of the vehicle
        self.speed_step = SPEED_STEP
        self.angle_step = ANGLE_STEP
        self.speed = INITIAL_SPEED
        self.angle = INITIAL_ANGLE
        self.position = [START_POS[0], START_POS[1]]
        self.center = [START_POS[0] + CAR_SIZE / 2, START_POS[1] + CAR_SIZE / 2]

        # Data for collision detection
        self.alive = True
        self.corner = []

        # Data for radar detection
        self.radars = INITIAL_RADAR.copy()

        # Reward data
        self.reward = 0
        self.distance = 0

    def move_update_by_keys(self):
        # update by keys
        # unused because using neural network
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.speed += self.speed_step
        if keys[pygame.K_s]:
            self.speed -= self.speed_step
        if keys[pygame.K_a]:
            self.angle += self.angle_step
        if keys[pygame.K_d]:
            self.angle -= self.angle_step
        self.sprite.image = self.center_rotation()
        self.position[0] += self.speed * math.cos(math.radians(self.angle))
        self.position[1] -= self.speed * math.sin(math.radians(self.angle))

    def move_update_by_nn(self, action):
        # update by neural network
        if action == 0:
            self.speed += self.speed_step
        if action == 1:
            if self.speed > INITIAL_SPEED:
                self.speed -= self.speed_step
        if action == 2:
            self.angle += self.angle_step
        if action == 3:
            self.angle -= self.angle_step
        self.sprite.image = self.center_rotation()
        self.position[0] += self.speed * math.cos(math.radians(self.angle))
        self.position[1] -= self.speed * math.sin(math.radians(self.angle))
        self.distance += self.speed

    def update_corner(self):
        # Update the location of the four corners of the vehicle
        length = CAR_SIZE / 2
        self.center = [int(self.position[0]) + length, int(self.position[1]) + length]
        self.corner = [[self.center[0] + math.sin(math.radians(self.angle + 45)) * length,
                        self.center[1] + math.cos(math.radians(self.angle + 45)) * length],
                       [self.center[0] + math.sin(math.radians(self.angle + 135)) * length,
                        self.center[1] + math.cos(math.radians(self.angle + 135)) * length],
                       [self.center[0] + math.sin(math.radians(self.angle + 225)) * length,
                        self.center[1] + math.cos(math.radians(self.angle + 225)) * length],
                       [self.center[0] + math.sin(math.radians(self.angle + 315)) * length,
                        self.center[1] + math.cos(math.radians(self.angle + 315)) * length]]

    def collision_detection(self, background):
        # check if the vehicle collides with the border of the race track
        for point in self.corner:
            point = [int(point[0]), int(point[1])]
            if background.get_at(point) == BORDER:
                pygame.event.post(pygame.event.Event(COLLISION))
                self.collision_response()
                break

    def collision_response(self):
        self.alive = False
        self.speed = 0

    def radar_detection(self, background, degree):
        # Basic parameters for radar detection
        theta = math.radians(self.angle + degree)
        direction = (math.cos(theta), math.sin(theta))
        length = CAR_SIZE / 2
        distance = 0

        # The point in the front of the vehicle
        x = int(self.center[0] + direction[0] * length)
        y = int(self.center[1] - direction[1] * length)

        while background.get_at((x, y)) != BORDER and distance < 300:
            distance += 1
            x = int(self.center[0] + direction[0] * (length + distance))
            y = int(self.center[1] - direction[1] * (length + distance))

        distance = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([x, y, distance])

    def draw_radar(self, screen):
        # Draw the radar on the screen
        for point in self.radars:
            pygame.draw.line(screen, "green", self.center, [point[0], point[1]])
            pygame.draw.circle(screen, "green", [point[0], point[1]], 5)

    def update_reward(self):
        # Calculate the reward for the current state
        if not self.alive:
            self.reward -= 100
        else:
            self.reward = self.distance

    def center_rotation(self):
        # rotation focusing in the center of the vehicle
        rect = self.image.get_rect()
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        rotated_rect = rect.copy()
        rotated_rect.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rect).copy()
        return rotated_image

    def draw(self, screen):
        # draw every frame
        screen.blit(self.sprite.image, self.position)
        self.draw_radar(screen)

    def update(self, screen, action):
        # update every frame
        if not self.alive:
            return
        self.radars.clear()
        self.move_update_by_nn(action)
        self.update_corner()
        self.collision_detection(screen)
        self.radar_detection(screen, 0)
        self.radar_detection(screen, 45)
        self.radar_detection(screen, -45)
        self.radar_detection(screen, 90)
        self.radar_detection(screen, -90)
        self.update_reward()


"""
The class of neural network
Create a neural network for the autonomous vehicle
"""


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        # initialize the neural network
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE)

    def forward(self, x):
        # use ReLU activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
Genetic algorithm class
Using genetic algorithm to find the best neural network for the autonomous vehicle
"""


class GeneticAlgorithm:
    def __init__(self, input_size):
        self.population_size = POPULATION_SIZE
        self.mutation_rate = MUTATION_RATE
        self.crossover_rate = CROSSOVER_RATE
        self.population = [NeuralNetwork(input_size) for _ in range(POPULATION_SIZE)]
        self.best_individual = None
        self.fitness = [0] * POPULATION_SIZE
        self.vehicles = []
        for _ in range(POPULATION_SIZE):
            self.vehicles.append(Vehicles())

    def get_action(self):
        # get the action for the vehicle based on the neural network
        action_list = []
        for vehicle, individual in zip(self.vehicles, self.population):
            state = torch.tensor([value for row in vehicle.radars for value in row], dtype=torch.float32)
            outputs = individual(state)
            outputs = torch.sigmoid(outputs)
            action = torch.argmax(outputs).item()
            action_list.append(action)
        return action_list

    def compute_fitness(self):
        total_reward = 0
        for i, vehicle in enumerate(self.vehicles):
            total_reward += vehicle.reward
            self.fitness[i] = vehicle.reward
        self.fitness = [max(1, f) for f in self.fitness]

    def selection(self):
        parent = sorted(self.population, key=lambda x: self.fitness[self.population.index(x)], reverse=True)[:2]
        return parent

    @staticmethod
    def crossover(parent1, parent2):
        alpha = random.random()
        child = NeuralNetwork(INPUT_SIZE)
        for param1, param2, param_child in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
            param_child.data = param1.data * alpha + param2.data * (1 - alpha)
        return child

    def mutate(self, individual):
        for param in individual.parameters():
            if random.random() < self.mutation_rate:
                param.data += torch.randn_like(param) * 0.1

    def next_generation(self):
        parent = self.selection()
        next_gen = [parent[0], parent[1]]
        for i in range(self.population_size - 2):
            if random.random() < self.crossover_rate:
                child = self.crossover(parent[0], parent[1])
            else:
                child = self.population[i]
            self.mutate(child)
            next_gen.append(child)
        self.population = next_gen


def run_simulation():
    # Initialize simulation
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Autonomous Vehicle")
    background = pygame.image.load("map.png").convert()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 36)

    ga = GeneticAlgorithm(INPUT_SIZE)
    for i in range(GENERATION):
        text_generation = font.render("Generation: " + str(i), True, 'black')
        timer = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            screen.blit(background, (0, 0))

            # update the vehicles and neural network
            num_alive = 0
            action_list = ga.get_action()
            for i, vehicle, individual in zip(range(POPULATION_SIZE), ga.vehicles, ga.population):
                if vehicle.alive:
                    action = action_list[i]
                    vehicle.update(screen, action)
                    num_alive += 1

            for vehicle in ga.vehicles:
                if vehicle.alive:
                    vehicle.draw(screen)

            # check if all the vehicles are dead
            if num_alive == 0:
                print("All the vehicles are dead")
                ga.compute_fitness()
                break

            # set a time limit for the simulation
            timer += 1
            if timer > TIME_LIMIT:
                print("Time limit exceeded")
                ga.compute_fitness()
                break

            screen.blit(text_generation, (10, 10))

            pygame.display.update()
            clock.tick(FPS)

        # get the next generation
        print(ga.fitness)
        ga.next_generation()

        for vehicle in ga.vehicles:
            vehicle.reset()


if __name__ == "__main__":
    run_simulation()
