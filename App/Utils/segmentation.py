import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Replace with custom-trained model if available

class SandPileApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sand Pile Volume Estimator")
        self.material_type = tk.StringVar(value="Sand")

        self.top_image_path = None
        self.front_image_path = None
        self.tk_images = []

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        ttk.Label(frame, text="Select Material:").grid(row=0, column=0, sticky='w')
        ttk.Combobox(frame, textvariable=self.material_type, values=["Sand", "10mm Gravel", "20mm Gravel"]).grid(row=0, column=1)

        ttk.Button(frame, text="Upload Top View", command=self.upload_top).grid(row=1, column=0, pady=5)
        ttk.Button(frame, text="Upload Front View", command=self.upload_front).grid(row=1, column=1, pady=5)
        ttk.Button(frame, text="Process", command=self.process).grid(row=2, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(frame, text="Detection Results Will Appear Below", font=("Arial", 10, "bold"))
        self.result_label.grid(row=3, column=0, columnspan=2)

        self.canvas = tk.Canvas(self.root, width=1800, height=1100)
        self.canvas.pack()

    def upload_top(self):
        self.top_image_path = filedialog.askopenfilename()
        messagebox.showinfo("Top View", f"Loaded: {self.top_image_path}")

    def upload_front(self):
        self.front_image_path = filedialog.askopenfilename()
        messagebox.showinfo("Front View", f"Loaded: {self.front_image_path}")

    def preprocess(self, path):
        img = cv2.imread(path)
        return cv2.resize(img, (640, 640))

    def background_subtract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

    def extract_foreground(self, img):
        mask = self.background_subtract(img)
        inverted_mask = cv2.bitwise_not(mask)  # 🔄 Invert the mask
        foreground = np.zeros_like(img)
        for c in range(3):
            foreground[:, :, c] = img[:, :, c] * (inverted_mask // 255)
        return foreground

    def segment(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        mask = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        return cv2.bitwise_and(img, img, mask=filtered_mask)

    def detect_objects(self, img):
        results = model(img)
        annotated = results[0].plot()
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        return annotated, labels

    def show_image(self, img, x, y, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((280, 280))
        tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.create_image(x, y, anchor="nw", image=tk_img)
        self.canvas.create_text(x + 140, y - 10, text=label, font=("Arial", 10, "bold"))
        self.tk_images.append(tk_img)

    def show_raw_input(self, path, x, y, label):
        img = Image.open(path).resize((280, 280))
        tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(x, y, anchor="nw", image=tk_img)
        self.canvas.create_text(x + 140, y - 10, text=label, font=("Arial", 10, "bold"))
        self.tk_images.append(tk_img)

    def process(self):
        if not self.top_image_path or not self.front_image_path:
            messagebox.showerror("Error", "Please upload both views.")
            return

        self.canvas.delete("all")
        self.tk_images.clear()

        self.show_raw_input(self.top_image_path, 50, 50, "Input - Top View")
        self.show_raw_input(self.front_image_path, 50, 400, "Input - Front View")

        # Top View Processing
        top_img = self.preprocess(self.top_image_path)
        top_fg = self.extract_foreground(top_img)
        top_seg = self.segment(top_fg)
        top_detected, top_labels = self.detect_objects(top_seg)

        # Front View Processing
        front_img = self.preprocess(self.front_image_path)
        front_fg = self.extract_foreground(front_img)
        front_seg = self.segment(front_fg)
        front_detected, front_labels = self.detect_objects(front_seg)

        # Display Results
        self.show_image(top_fg, 350, 50, "Top - Foreground")
        self.show_image(top_seg, 650, 50, "Top - Segmented")
        self.show_image(top_detected, 950, 50, "Top - Detection")

        self.show_image(front_fg, 350, 400, "Front - Foreground")
        self.show_image(front_seg, 650, 400, "Front - Segmented")
        self.show_image(front_detected, 950, 400, "Front - Detection")

        result_text = (
            f"Material: {self.material_type.get()}\n\n"
            f"Top View Detected: {top_labels}\n"
            f"Front View Detected: {front_labels}"
        )
        self.result_label.config(text=result_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = SandPileApp(root)
    root.mainloop()
