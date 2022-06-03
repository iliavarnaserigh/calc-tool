import CONST

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

	def __abs__(self):
		return (self.val[0] ** 2 + self.val[1] ** 2) ** .5

class ElecField:
	def __init__(self, body):
		self.body = body
		self.abs = lambda x : (x - body.pos) * (-(body.q * CONST.k) / abs(x - body.pos) ** 3)

	def update(self):
		self.abs = lambda x : (x - self.body.pos) * (-(self.body.q * CONST.k) / abs(x - self.body.pos) ** 3)

class GravityField:
	def __init__(self, body):
		self.body = body
		self.abs = lambda x : (x - body.pos) * (body.mass * -CONST.G / abs(x - body.pos) ** 3)

	def update(self):
		self.abs = lambda x : (x - self.body.pos) * (self.body.mass * -CONST.G / abs(x - self.body.pos) ** 3)		

class GravForce:
	def __init__(self, body, field):
		self.field = field
		self.body = body
		self.force = self.field.abs(self.body.pos) * self.body.mass

	def update(self):
		self.force = self.field.abs(self.body.pos) * self.body.mass

class ElecForce:
	def __init__(self, body, field):
		self.field = field
		self.body = body
		self.force = field.abs(self.body.pos)

	def update(self):
		self.force = self.field.abs(self.body.pos)



class Body:
	def __init__(self, init_pos, init_vel, mass, dt=CONST.dt, q=1):
		self.pos = init_pos
		self.vel = init_vel # a tuple => vector of velocity
		self.mass = mass
		self.dt = dt
		self.q = q

	def applyForce(self, force):
		# force is a tuple => vector
		self.dt = CONST.dt
		self.vel += force.force * (1 / self.mass) * self.dt

	def applyVel(self):
		self.dt = CONST.dt
		self.pos += self.vel * self.dt
