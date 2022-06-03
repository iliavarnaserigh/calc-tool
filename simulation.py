import pygame as pg
## INITILIZATION ##
pg.init()
## VARIABLES ##
g = -9.8
mass = 20
dt = .01
running = 1
STANDARD_SCREEN_DIMENSIONS = [800, 800]
STANDARD_DIMENSIONS = [100, 100]
WINDOW = pg.display.set_mode(tuple(STANDARD_SCREEN_DIMENSIONS)); pg.display.set_caption("Calculus-tool : main-script.py : ln 13 : Graphing tool")
WINDOW.fill((255, 255, 255))
CENTRE_POINT = [int(STANDARD_DIMENSIONS[0] / 2), int(STANDARD_DIMENSIONS[1] / 2)]
pg.draw.line(WINDOW, (0, 0, 0), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), STANDARD_SCREEN_DIMENSIONS[1]), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), 0))
pg.draw.line(WINDOW, (0, 0, 0), (0, int(STANDARD_SCREEN_DIMENSIONS[1] / 2)), (STANDARD_SCREEN_DIMENSIONS[0], int(STANDARD_SCREEN_DIMENSIONS[1] / 2)))
pg.transform.scale(WINDOW, (1, 1))

def transform_coordinates(point, centre=CENTRE_POINT, dims_screen=STANDARD_SCREEN_DIMENSIONS, dims=STANDARD_DIMENSIONS): # transform point in cartesian coords with respec to centre of screen to pygame coords
    return [dims_screen[0]/dims[0] * (centre[0] + point[0]), dims_screen[1]/dims[1] * (centre[1] - point[1])]



## OBJECTS ##
class Vector:
	def __init__(self, val): # val is a tuple
		self.val = val[:]

	def __add__(self, other):
		res = []
		for i in range(len(self.val)):
			res.append(self.val[i] + other.val[i])

		return Vector(tuple(res[:]))

	def __sub__(self, other):
		res = []
		for i in range(len(self.val)):
			res.append(self.val[i] - other.val[i])

		return Vector(tuple(res[:]))

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return Vector([other * i for i in self.val])

		else:
			return sum([self.val[i] * other.val[i] for i in range(len(self.val))])


class Body:
	def __init__(self, init_pos, init_vel, mass, dt=dt):
		self.pos = init_pos
		self.vel = init_vel # a tuple => vector of velocity
		self.mass = mass
		self.dt = dt

	def applyForce(self, force):
		# force is a tuple => vector
		self.vel += force * (1 / self.mass) * self.dt

	def applyVel(self):
		self.pos += self.vel * self.dt

## MAINLOOP ##
body = Body(Vector([0, 0]), Vector([10, 10]), mass)
force = Vector([0, -9.8]) * mass
while running:

	for ev in pg.event.get():
		if ev.type == pg.QUIT:
			running = 0

	body.applyForce(force)
	body.applyVel()
	pg.draw.circle(WINDOW, (0, 0, 255), transform_coordinates(body.pos.val[:]), 5)
	pg.display.update()

pg.quit()
