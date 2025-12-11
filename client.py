import tkinter as tk
import socket
import struct

class MNISTClient:
    def __init__(self, master):
        self.master = master
        master.title("mnist client")

        # config
        self.cell_size = 30
        self.grid_size = 28
        self.canvas_size = self.grid_size * self.cell_size

        self.pixels = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        self.canvas = tk.Canvas(master, width = self.canvas_size, height = self.canvas_size, bg = "black")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.label_pred = tk.Label(master, text = "draw a digit", font = ("Helvetica", 16))
        self.label_pred.pack()

        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack()

        self.btn_predict = tk.Button(self.btn_frame, text = "predict", command = self.send_prediction)
        self.btn_predict.pack(side = tk.LEFT)

        self.btn_clear = tk.Button(self.btn_frame, text = "clear", command = self.clear_canvas)
        self.btn_clear.pack(side = tk.LEFT)

    def draw(self, event) :
        x, y = event.x // self.cell_size, event.y // self.cell_size
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            x1, y1 = x * self.cell_size, y * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size

            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "white", outline = "white")

            self.pixels[y][x] = 1.0

            # bleed effect

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx , ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.pixels[ny][nx] = max(self.pixels[ny][nx], 0.5)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pixels = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.label_pred.config(text = "draw a digit")

    def send_prediction(self):
        flat_pixels = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                flat_pixels.append(self.pixels[r][c])
        
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(('127.0.0.1', 8080))

            # pack floats to binary
            data = struct.pack(f'{len(flat_pixels)}f', *flat_pixels)
            client.sendall(data)

            # recieve response
            response = client.recv(1024).decode('utf-8')
            self.label_pred.config(text = f"prediction: {response}")

            client.close()
        except ConnectionRefusedError:
            self.label_pred.config(text = "error: server offline")

root = tk.Tk()
client = MNISTClient(root)
root.mainloop()




