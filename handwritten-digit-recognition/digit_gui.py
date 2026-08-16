from tkinter import messagebox as tm
from tkinter import Canvas, Button, Frame, PhotoImage, W, Tk
import pickle
import json
from collections import Counter
import cv2
import numpy as np


class Login(Frame):
	def __init__(self, abc):
		super().__init__(abc)
		self.img = None
		self.btnlogin = Button(self, padx=10, pady=10, bd=10, fg='black',
			font=('arial', 17, 'bold'), width=10, text="Load", bg='powder blue',
			command=self.load)
		self.btnpredict = Button(self, padx=10, pady=10, bd=10, fg='black',
			font=('arial', 17, 'bold'), width=10, text="predict", bg='powder blue',
			command=self.predict)
		self.btnmetrics = Button(self, padx=10, pady=10, bd=10, fg='black',
			font=('arial', 17, 'bold'), width=10, text="metrics", bg='powder blue',
			command=self.show_metrics)
		self.btnpredict.grid(row=0, column=1)
		self.btnlogin.grid(row=0, column=0)
		self.btnmetrics.grid(row=0, column=2)
		self.canvas = Canvas(abc, width=400, height=200)
		self.canvas.pack()
		self.pack()

	def load(self):
		try:
			self.img = PhotoImage(file='img.png')
			self.canvas.create_image(200, 100, anchor=W, image=self.img)
			tm.showinfo('completion box', 'loaded')
		except FileNotFoundError:
			tm.showinfo('error', 'img.png not found in this folder.')

	def predict(self):
		try:
			x = cv2.imread('img.png')
			if x is None:
				raise FileNotFoundError('img.png')
			x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
			# Invert so the digit is white-on-black, matching MNIST.
			# Remove this line if your image is already MNIST-style (white digit).
			x = cv2.bitwise_not(x)
			x = cv2.resize(x, (28, 28))
			with open('training.pkl', 'rb') as f:
				k = pickle.load(f)
			x = k[0].transform(x.reshape(1, -1).astype(np.float64))
			x_pca = k[1].transform(x)
			l = [int(model.predict(x_pca.reshape(1, -1))) for model in k[2:5]]
			p = Counter(l).most_common()[0][0]
			tm.showinfo('completion box', f'prediction={p}')
		except FileNotFoundError:
			tm.showinfo('error',
				'training.pkl or img.png missing. Run train.py and add img.png.')
		except Exception as e:
			tm.showinfo('error', str(e))

	def show_metrics(self):
		try:
			with open('metrics.json') as f:
				metrics = json.load(f)
			msg = '\n'.join(f'{name}: {acc:.4f}' for name, acc in metrics.items())
		except FileNotFoundError:
			msg = 'metrics.json not found. Run train.py first.'
		tm.showinfo('Validation accuracy', msg)


root = Tk()
obj = Login(root)
root.title('Digit Recognition')
root.geometry('800x400')
root.mainloop()
