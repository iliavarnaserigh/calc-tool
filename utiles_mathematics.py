import math 
import matplotlib.pyplot as plt


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
class Matrix:
	def __init__(self, array, identity=False):
		self.array = array
		self.columns = [[row[i] for row in self.array] for i in range(len(self.array[0]))]
		self.identity =[]
		if identity:
			for i in range(len(self.array)):
				row = [0] * len(self.columns)
				row[i] = 1
				self.identity.append(row[:])

	def __str__(self):
		string = [" "]
		for row in self.array:
			for item in range(len(row)):
				string += ["%d%s "%(row[item], "\n" if item == len(row) - 1 else "")]

		return "".join(string)

	def __add__(self, other):
		try:
			result = []
			for row in range(len(self.array)):
				new_row = []
				for column in range(len(self.array[row])):
					new_row += [self.array[row][column] + other.array[row][column]]
				result += [new_row]
			return Matrix(result)

		except:
			raise (ValueError,"the dimensions or types are incompatible.")

	def __sub__(self, other):
		return self + (other * (-1))

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return Matrix([[j * other for j in i] for i in self.array])

		elif isinstance(other, Matrix):
			def mulRow(row, column):
				return sum([row[i] * column[i] for i in range(len(row))])

			result = []
			for row in self.array:
				row_result = []
				for column in other.columns:
					row_result.append(mulRow(row, column))
				result.append(row_result)

			return Matrix(result)

	def det(self):
		if len(self.array) != len(self.columns):
			raise (ValueError, "matrix is not square.")

		else:
			dim = len(self.array)
			if dim == 1:
				return self.array[0][0]
				
			if dim == 2:
				return self.array[0][0] * self.array[1][1] - self.array[0][1]*self.array[1][0]

			def minor(matrix, row, column):
				new_matrix = matrix[0:row-1] + matrix[row:]
				
				for i in range(len(new_matrix)):
					new_row = new_matrix[i][0:column-1] + new_matrix[i][column:]
					new_matrix[i] = new_row
				return Matrix(new_matrix)
	
			return sum([self.minor(1, i+1).det() * ((-1) ** (i)) * self.array[0][i] for i in range(dim)])

	def minor(self, row, column):
		new_matrix = self.array[0:row-1] + self.array[row:]
		
		for i in range(len(new_matrix)):
			new_row = new_matrix[i][0:column-1] + new_matrix[i][column:]
			new_matrix[i] = new_row
		return Matrix(new_matrix)

	def transpose(self):		
		return Matrix([[self.array[i][j]for i in range(len(self.array))]for j in range(len(self.columns))])
	
	def adjugate(self):
		return Matrix([[(-1)**(i+j) * self.minor(i+1, j+1).det() for j in range(len(self.columns))] for i in range(len(self.array))]).transpose()

	def inverse(self):
		return self.adjugate() * (1/self.det())

	def char_polynomial(self):
		if len(self.array) != len(self.columns):
			raise (ValueError, "matrix is not square.")

		else:
			return lambda x : (self.identity * x - self).det()

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

class Var(object):
	def __init__(self, name):
		self.name = name
		self.func = lambda x : x[0] # x is a list typically consisting of only one element

	def __add__(self, other):
		new_var = Var("%s+%s"%(self.name, other.name))
		new_var.func = lambda x : self.func(x[0]) + other.func(x[1]) # x is a list [x, y]

	def __mul__(self, other):
		new_var = Var("%s*%s"%(self.name, other.name))
		new_var.func = lambda x : self.func(x[0]) * other.func(x[1]) # x is a list [x, y]

	def __div__(self, other):
		new_var = Var("%s/%s"%(self.name, other.name))
		new_var.func = lambda x : self.func(x[0]) / other.func(x[1]) # x is a list [x, y]

	def __sub__(self, other):
		new_var = Var("%s-%s"%(self.name, other.name))
		new_var.func = lambda x : self.func(x[0]) - other.func(x[1]) # x is a list [x, y]
	
def det(matrix):
	if len(matrix.array) != len(matrix.columns):
		raise (ValueError, "matrix is not square.")

	else:
		dim = len(matrix.array)
		if dim == 1:
			return matrix.array[0][0]
			
		if dim == 2:
			return matrix.array[0][0] * matrix.array[1][1] - matrix.array[0][1]*matrix.array[1][0]

		def minor(matrix, row, column):
			new_matrix = matrix[0:row-1] + matrix[row:]
			
			for i in range(len(new_matrix)):
				new_row = new_matrix[i][0:column-1] + new_matrix[i][column:]
				new_matrix[i] = new_row
			return Matrix(new_matrix)

		return sum([matrix.minor(1, i+1).det() * ((-1) ** (i)) * matrix.array[0][i] for i in range(dim)])

def minor(matrix, row, column):
	new_matrix = matrix.array[0:row-1] + matrix.array[row:]
	
	for i in range(len(new_matrix)):
		new_row = new_matrix[i][0:column-1] + new_matrix[i][column:]
		new_matrix[i] = new_row
	return Matrix(new_matrix)

def transpose(matrix):		
	return Matrix([[matrix.array[i][j]for i in range(len(matrix.array))]for j in range(len(matrix.columns))])

def adjugate(matrix):
	return Matrix([[(-1)**(i+j) * matrix.minor(i+1, j+1).det() for j in range(len(matrix.columns))] for i in range(len(matrix.array))]).transpose()

def inverse(matrix):
	return matrix.adjugate() * (1/matrix.det())

def char_polynomial(matrix):
	if len(matrix.array) != len(matrix.columns):
		raise (ValueError, "matrix is not square.")

	else:
		return lambda x : (matrix.identity * x - matrix).det()

def find_eigenvalue(matrix, x, n):
	return newtons_method(char_polynomial(matrix), x, n)

class Transformations2D:
	@staticmethod
	def strech_x(k):
		return Matrix([[k, 0], [0, 1]])

	@staticmethod
	def strech_y(k):
		return Matrix([[1, 0], [0, k]])

	@staticmethod
	def squeeze_x(k):
		return Matrix([[k, 0], [0, 1/k]])

	@staticmethod
	def rotation(theta):
		return Matrix([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
	
	@staticmethod
	def shear(self, k):
		return Matrix([[1, k], [0, 1]])

	@staticmethod
	def reflect(vect):
		return Matrix([[vect[0]**2 - vect[1]**2, 2*vect[0]*vect[1]], [2*vect[0]*vect[1], vect[1]**2 - vect[0]**2]]) * (1 / (vect[0] ** 2 + vect[1] ** 2))

	@staticmethod
	def orthogonal_projection(vect):
		return Matrix([[vect[0]**2, vect[0]*vect[1]], [vect[0]*vect[1], vect[1]**2]]) * (1 / (vect[0] ** 2 + vect[1] ** 2))
i = Matrix([[1], [0]])
j = Matrix([[0], [1]])
i_prime = Transformations2D.rotation(math.pi / 3) * i
j_prime = Transformations2D.rotation(-math.pi / 12) * j	
plt.arrow(0, 0, i.array[0][0], i.array[1][0])
plt.arrow(0, 0, j.array[0][0], j.array[1][0])
plt.arrow(0, 0, i_prime.array[0][0], i_prime.array[1][0], color="red")
plt.arrow(0, 0, j_prime.array[0][0], j_prime.array[1][0], color="red")
plt.show()