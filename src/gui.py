import os
import time
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image
from ultralytics.models.yolo import YOLO

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class YOLOApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print("Model YOLO załadowany pomyślnie.")
        else:
            self.model = None
            print(f"Ostrzeżenie: Nie znaleziono pliku modelu w {self.model_path}")

        self.current_image_path = None

        self.setup_main_window()
        self.create_sidebar()
        self.create_main_panel()
        self.create_status_bar()

    def setup_main_window(self):
        self.title("Detektor Odcinków - YOLO v11")
        self.geometry("1000x650")

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(
            self.sidebar, text="MENU", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(padx=20, pady=20)

        self.btn_load = ctk.CTkButton(
            self.sidebar, text="Wczytaj Obraz", command=self.load_image
        )
        self.btn_load.pack(padx=20, pady=10)

        self.btn_detect = ctk.CTkButton(
            self.sidebar,
            text="URUCHOM DETEKCJĘ",
            fg_color="green",
            hover_color="darkgreen",
            command=self.run_detection,
        )
        self.btn_detect.pack(padx=20, pady=30)

    def create_main_panel(self):
        self.main_panel = ctk.CTkFrame(self)
        self.main_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.img_container = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self.img_container.pack(fill="both", expand=True, padx=20, pady=20)

        self.lbl_orig = ctk.CTkLabel(
            self.img_container,
            text="[ Podgląd Oryginału ]",
            width=350,
            height=350,
            fg_color="#2B2B2B",
            corner_radius=8,
        )
        self.lbl_orig.pack(side="left", fill="both", expand=True, padx=10)

        self.lbl_res = ctk.CTkLabel(
            self.img_container,
            text="[ Wynik Detekcji YOLO ]",
            width=350,
            height=350,
            fg_color="#2B2B2B",
            corner_radius=8,
        )
        self.lbl_res.pack(side="right", fill="both", expand=True, padx=10)

    def create_status_bar(self):
        self.status_panel = ctk.CTkFrame(self.main_panel, height=80)
        self.status_panel.pack(fill="x", side="bottom", padx=20, pady=10)

        self.lbl_counter = ctk.CTkLabel(
            self.status_panel,
            text="Wykryto odcinków: --",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.lbl_counter.pack(side="left", padx=20, pady=10)

        self.lbl_time = ctk.CTkLabel(
            self.status_panel, text="Czas operacji: -- ms", font=ctk.CTkFont(size=14)
        )
        self.lbl_time.pack(side="right", padx=20, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp")]
        )

        if file_path:
            self.current_image_path = file_path

            # --- POPRAWKA 1: Wyciągamy prawdziwą liczbę linii z nazwy pliku ---
            filename = os.path.basename(file_path) # np. "line66_9.png"
            try:
                # Wyciągamy to, co jest między "_" a "."
                self.true_count = filename.split("_")[1].split(".")[0]
            except IndexError:
                self.true_count = "?" # Jeśli plik nie ma standardowej nazwy

            pil_img = Image.open(file_path)

            self.img_orig_ref = ctk.CTkImage(
                light_image=pil_img, dark_image=pil_img, size=(350, 350)
            )

            self.lbl_orig.configure(image=self.img_orig_ref, text="")

            # --- POPRAWKA 2: Całkowite czyszczenie prawego panelu ---
            self.lbl_res.configure(image="", text="[ Oczekiwanie na detekcję ]")
            
            # Resetujemy dolny pasek, pokazując już prawdziwą liczbę linii z pliku
            self.lbl_counter.configure(text=f"Wykryto: --  |  Prawdziwe: {self.true_count}", text_color="white")
            self.lbl_time.configure(text="Czas operacji: -- ms")

    def run_detection(self):
        if not self.current_image_path:
            self.lbl_res.configure(text="[ Najpierw wczytaj obraz! ]")
            return

        if not self.model:
            self.lbl_res.configure(text="[ Błąd: Brak modelu best.pt! ]")
            return

        start_time = time.time()
        results = self.model(self.current_image_path, conf=0.3, verbose=False)
        end_time = time.time()
        elapsed_time_ms = int((end_time - start_time) * 1000)

        predicted_count = len(results[0].boxes)

        # Rysowanie ramek (wyłączamy napisy klas i conf, żeby obrazek był czysty)
        bgr_array = results[0].plot(labels=False, conf=False)
        rgb_array = bgr_array[:, :, ::-1]

        pil_res_img = Image.fromarray(rgb_array)

        self.img_res_ref = ctk.CTkImage(
            light_image=pil_res_img, dark_image=pil_res_img, size=(350, 350)
        )
        self.lbl_res.configure(image=self.img_res_ref, text="")

        # --- POPRAWKA 1 cd: Porównanie wyników i kolorowy status ---
        if self.true_count != "?":
            if int(predicted_count) == int(self.true_count):
                status_text = f"Wykryto: {predicted_count}  |  Prawdziwe: {self.true_count}  [OK]"
                text_color = "green"
            else:
                status_text = f"Wykryto: {predicted_count}  |  Prawdziwe: {self.true_count}  [BŁĄD]"
                text_color = "red"
        else:
            status_text = f"Wykryto odcinków: {predicted_count}"
            text_color = "white"

        self.lbl_counter.configure(text=status_text, text_color=text_color)
        self.lbl_time.configure(text=f"Czas operacji: {elapsed_time_ms} ms")


if __name__ == "__main__":
    app = YOLOApp()
    app.mainloop()
