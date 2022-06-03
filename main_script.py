import pygame as pg
import numpy as np
import random as rn
import math
import sys

## INITILIZATION ##
pg.init()

## VARIABLE ##
STANDARD_SCREEN_DIMENSIONS = [800, 800]
STANDARD_DIMENSIONS = [20, 20]
WINDOW = pg.display.set_mode(tuple(STANDARD_SCREEN_DIMENSIONS)); pg.display.set_caption("Calculus-tool : main-script.py : ln 13 : Graphing tool")
WINDOW.fill((255, 255, 255))
CENTRE_POINT = [int(STANDARD_DIMENSIONS[0] / 2), int(STANDARD_DIMENSIONS[1] / 2)]
pg.draw.line(WINDOW, (0, 0, 0), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), STANDARD_SCREEN_DIMENSIONS[1]), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), 0))
pg.draw.line(WINDOW, (0, 0, 0), (0, int(STANDARD_SCREEN_DIMENSIONS[1] / 2)), (STANDARD_SCREEN_DIMENSIONS[0], int(STANDARD_SCREEN_DIMENSIONS[1] / 2)))
pg.transform.scale(WINDOW, (1, 1))


## FUNCTIONS ##

def transform_coordinates(point, centre=CENTRE_POINT, dims_screen=STANDARD_SCREEN_DIMENSIONS, dims=STANDARD_DIMENSIONS): # transform point in cartesian coords with respec to centre of screen to pygame coords
    return [dims_screen[0]/dims[0] * (centre[0] + point[0]), dims_screen[1]/dims[1] * (centre[1] - point[1])]

def plot(point_array, color=(255, 0, 0), disp=WINDOW):
    for point in point_array:
        real_point = transform_coordinates(point)
        pg.draw.circle(disp, color, real_point, 1)

    return True

def show():
    pg.display.update()
    running = 1
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = 0
                pg.quit()

def create_points(function, domain, n=1000):
    x_points = np.linspace(domain[0], domain[1], 10000)
    y_points = [function(i) for i in x_points]
    points = list(zip(x_points, y_points))
    return points

def plotF(f, domain, n=1000, color=(255, 0, 0)):
    plot(create_points(f, domain, n=n), color=color)

def sgn(x):
    return -1 if x < 0 else 1

def D(f, dx=0.0001):
    return lambda x : (f(x + dx) - f(x - dx)) / (2 * dx)

def diff(f, n):
    if n == 0:
        return f
    if n > 0:
        return diff(D(f), n - 1)

def taylorExpansion(f, a, n):
    def func(x):
        s = 0
        for i in range(n + 1):
            s += ((x - a) ** i) / (math.factorial(i)) * diff(f, i)(x)

        return s

    return lambda x : func(x / 2)

def I(f, C, dx=0.001): # INDEFINITE INTEGRAL
    def integral(domain):
        a = domain[0]
        b = domain[1]
        if b < a:
             x_space = np.linspace(b, a, round((a - b) / dx))
             area_space = [-f(i) * dx for i in x_space]
             return sum(area_space)

        x_space = np.linspace(a, b, round((b - a) / dx))
        area_space = [f(i) * dx for i in x_space]
        return sum(area_space)

    return lambda x : integral((0, x)) + C

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def newtons_method(f, x, n):
    if n == 1:
        return x

    if n > 1:
        return newtons_method(f, x - (f(x)) / (D(f)(x)), n - 1)

def lagrange_interpolation(input_data):
    
    def little_l(selected_key):
        def l_j(x):
            p = 1
            for i, j in input_data:
                if i != selected_key:
                    p *= (x-i) / (selected_key-i)

            return p
        return l_j

    def interpolation(x):
        s = 0
        for i, j in input_data:
            s += j * little_l(i)(x)

        return s

    return interpolation

## OBJECTS ##
class polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs # coeffs = [(c1, p1), (c2, p2), ...]
        self.deg = max([j for i, j in coeffs])

    def __call__(self, x):
        s = 0
        for i, j in self.coeffs:
            s += i * x ** j

        return s

    def func(self):
        return lambda x : self(x)

    def __add__(self, other):
        s = []
        for i, j in self.coeffs:
            for c, p in other.coeffs:
                if p == j :
                    s += [(i + c, j)]
                    break
                if p not in [b for a, b in self.coeffs]:
                    s += [(c, p)]
            else:
                s += [(i, j)]

        return polynomial(s)

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            s = []
            for i, j in self.coeffs:
                for c, p in other.coeffs:
                    s += [(i * c, j + p)]
            power_dict = {}
            for i, j in s:
                if j not in power_dict.keys():
                    power_dict.update({j : i})

                else:
                    power_dict[j] += i
                    
            s_prime = []
            for i, j in power_dict.items():
                s_prime += [(j, i)]
                
            return polynomial(s_prime)
        
        else:
            return polynomial([(i * other, j) for i, j in self.coeffs])


    def __sub__(self, other):
        return self + (other) * (-1)

    def root(self, n=10):
        return newtons_method(self.func, 0, n)

